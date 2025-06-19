import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
import torch


class LoraClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_genres):
        super().__init__()
        self.base_model = base_model  
        self.classifier = nn.Linear(hidden_size, num_genres)  

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        cls_hidden = last_hidden[:, 0, :]  # возьмем [BOS] токен или первый
        logits = self.classifier(cls_hidden)  # [batch_size, num_genres]
        return logits


def prepare_lora(model, r=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"]):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,  
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def validate_model(model, val_loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_labels.append(labels)
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    f1 = f1_score(all_labels, all_preds, average='micro')
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    print(f"Validation Metrics → F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return {"f1": f1, "precision": precision, "recall": recall}

