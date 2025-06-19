import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        tokenizer,
        model_type="lora",             # ["zero-shot", "ptune", "lora", "finetune"]
        strategy="cls",                # ["cls", "mean", "last"]- strategies got extracting embeddings
        num_labels=1,                  # 1 — for single genre, N — for multi-label
        dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.strategy = strategy
        self.num_labels = num_labels

        if model_type in ["lora", "ptune", "finetune"]:
            hidden_size = self._get_hidden_size()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)