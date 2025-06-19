import json
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score



def get_pretrained(model_name: str, 
                   device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    return tokenizer, model

def parse_model_response(response: str) -> int:
    try:
        match = re.search(r'\{[^}]*"predict"\s*:\s*(0|1)[^}]*\}', response)
        if match:
            data = json.loads(match.group(0))
            return int(data['predict'])
    except Exception as e:
        print(f"Parsing error: {e}")
    raise ValueError("Could not parse prediction from model response.")


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