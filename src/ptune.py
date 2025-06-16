import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import (
    PromptTuningConfig,
    get_peft_model,
    PromptTuningInit
)
from sklearn.metrics import f1_score


class MultiLabelClassifier(nn.Module):
    def __init__(self, peft_model, hidden_size, num_labels):
        super().__init__()
        self.peft = peft_model
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, **kwargs):
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("output_hidden_states", True)
        outputs = self.peft(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, H]
        logits = self.classifier(last_hidden)              # [B, num_labels]
        return logits


def prepare_ptune(model,
                  model_name: str,
                  genres,
                  device: str):
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=f"""    You are an expert music genre classifier. Analyze these song lyrics features:
        1. Themes (love, party, rebellion)
        2. Language style (slang, poetic)
        3. Rhythmic patterns
        4. Cultural context
        
        Possible genres: {', '.join(genres)}
        
        Example 1:
        Lyrics: "We’re rolling down the street with the bass turned up, neon lights flashing..."
        Genre: Hip-Hop
        
        Example 2:
        Lyrics: "Sweet child o' mine, you're the only one that's on my mind"
        Genre: Rock
        
        Note: Be concise. Classify the following:
        Lyrics: {{input_lyrics}}
        Genre:""".replace("{", "{{").replace("}", "}}"),  # Подставляем реальные жанры из датасета
        num_virtual_tokens=30,
        tokenizer_name_or_path=model_name,
        inference_mode=False
    )
    peft_model = get_peft_model(model, prompt_config)
    peft_model.gradient_checkpointing_enable()
    peft_model.print_trainable_parameters()    
    peft_model.to(device)
    return peft_model

def train(model: AutoModelForCausalLM, 
          idx2genre,
          learning_rate: float, 
          num_epochs: int,
          tokenizer: AutoTokenizer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: str):

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Map genres to token IDs
    label_token_ids = {
        genre: tokenizer.encode(' ' + genre, add_special_tokens=False)[0] 
        for genre in idx2genre.values()
    }
    genre_token_ids = torch.tensor(list(label_token_ids.values()), device=device)  # [num_genres]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()  # [B, num_genres]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                # outputs = model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     return_dict=True
                # )
                # Get logits at last token
                # lm_logits = outputs.logits  # [B, T, vocab]
                # last_logits = lm_logits[:, -1, :]  # [B, vocab]
                
                # Extract logits for genre token ids
                genre_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = criterion(genre_logits, labels)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).float()
                genre_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = (torch.sigmoid(genre_logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        label_accuracy = (all_preds == all_labels).mean()
        print(f"Epoch {epoch+1}/{num_epochs} - Val macro F1: {macro_f1:.4f}, Label acc: {label_accuracy:.4f}")
    return model