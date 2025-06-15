import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import (
    PromptTuningConfig,
    get_peft_model,
    PromptTuningInit
)

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
        Genre:""".replace("{", "{{").replace("}", "}}"),  
        num_virtual_tokens=30,
        tokenizer_name_or_path=model_name,
        inference_mode=False
    )
    model = get_peft_model(model, prompt_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()    
    model.to(device)
    return model

def train(model: AutoModelForCausalLM, 
          idx2genre,
          learning_rate: float, 
          num_epochs: int,
          tokenizer: AutoTokenizer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: str):

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    label_token_ids = {g: tokenizer.encode(' ' + g, add_special_tokens=False)[0] for g in idx2genre.values()}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                lm_logits = outputs.logits  # [B, T, vocab]
                last_logit = lm_logits[:, -1, :]  # [B, vocab]
                target_ids = torch.tensor(
                    [label_token_ids[idx2genre[i[0].item()]] for i in labels],
                    device=device,
                    dtype=torch.long
                )
                loss = loss_fn(last_logit, target_ids)

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
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]
                preds = logits.argmax(dim=-1)
                target_ids = torch.tensor(
                    [label_token_ids[idx2genre[i[0].item()]] for i in labels],
                    device=device,
                    dtype=torch.long
                )
                correct += (preds == target_ids).sum().item()
                total += labels.size(0)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        print(f"Epoch {epoch+1}/{num_epochs} - Val accuracy: {correct/total:.4f}")
    return model
