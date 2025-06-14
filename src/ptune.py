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


def label_smoothed_nll_loss(logits, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    return (confidence * nll_loss + smoothing * smooth_loss).mean()


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
        num_virtual_tokens=50,
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
            labels = batch['labels'].to(device)  # genre indices

            # Build prompt: virtual tokens + input text
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # dummy for LM
                return_dict=True
            )
            lm_logits = outputs.logits  # [batch, seq_len, vocab]

            # Use last token logits to predict genre token
            last_logit = lm_logits[:, -1, :]  # [batch, vocab]
            target_ids = torch.tensor(
                [label_token_ids[idx2genre[i.item()]] for i in labels],
                device=device
            )

            loss = label_smoothed_nll_loss(logits, labels, smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        torch.cuda.empty_cache()
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
                    [label_token_ids[idx2genre[i.item()]] for i in labels],
                    device=device
                )
                correct += (preds == target_ids).sum().item()
                total += labels.size(0)
                torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{num_epochs} - Val accuracy: {correct/total:.4f}")
    return model