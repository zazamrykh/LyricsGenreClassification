import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW


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
