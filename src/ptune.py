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
from sklearn.metrics import f1_score, precision_score, recall_score
from src.metrics import GenrePredictorWrapper, evaluate_model
from src.utils import plot_metrics


class MultiLabelClassifier(nn.Module):
    def __init__(self, peft_model, hidden_size, num_labels):
        super().__init__()
        self.peft = peft_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.peft(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()  
        return self.classifier(last_hidden)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        pt = targets * probas + (1 - targets) * (1 - probas)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def prepare_ptune(model,
                  model_name: str,
                  genres,
                  device: str):
    prompt_text = f"""You are an expert in music genre classification. Given the lyrics of a song, list **all applicable genres**. 
        Genres may include: {', '.join(genres)}.

        Analyze based on:
        1. Themes (e.g., love, rebellion, spirituality)
        2. Vocabulary (e.g., slang, poetic language)
        3. Rhythm and repetition
        4. Cultural/scene references

        Examples:

        Lyrics: "We’re rolling down the street with the bass turned up, neon lights flashing..."
        Genres: Hip-Hop, Electronic

        Lyrics: "Sweet child o' mine, you're the only one that's on my mind"
        Genres: Rock, Ballad

        Now classify this:
        Lyrics: {{input_lyrics}}
        Genres:""".replace("{", "{{").replace("}", "}}")
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=prompt_text,
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
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    wrapper = GenrePredictorWrapper(model, device)
    history = {
        'train_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    best_val_f1 = 0.0
    patience_counter = 0


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        metrics = evaluate_model(wrapper, val_loader, device=device)
        val_f1 = metrics['f1']
        val_precision = metrics['precision']
        val_recall = metrics['recall']
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f}")

        # if val_f1 > best_val_f1:
        #     best_val_f1 = val_f1
        #     patience_counter = 0
        #     torch.save(model.state_dict(), '../experiments/best_qwen_prune.pt')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= 10:
        #         print("Early stopping triggered.")
        #         break
    plot_metrics(history)
    return model