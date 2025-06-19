import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from abc import ABC, abstractmethod

# ModelEvalInterface - interface for model evaluation. Model should get row of df as input and then return vector of predictions (prediction if row belongs to some class of not).
class GenrePredictorInterface(ABC):
    @abstractmethod
    def predict(self, batch_features: dict) -> np.array:
        """
        Get batched input that contains 'input_ids', 'labels', 'features'
        Returns prediction in binary format [batch_size, num_classes] as numpy array
        """
        pass

def evaluate_model(model: GenrePredictorInterface, dataloader, device='cpu'):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            targets = batch['labels']
            preds = model.predict(batch)
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # confusion_matrix — многоклассовая, тут нужна "ошибочная матрица" в multilabel стиле
    # Мы сделаем aggregated confusion-like матрицу:
    num_classes = y_true.shape[1]
    error_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.where(y_pred[i] == 1)[0]

        for pred in pred_labels:
            for true in true_labels:
                error_matrix[pred, true] += 1

    metrics = {
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'error_matrix': error_matrix
    }

    return metrics


class GenrePredictorWrapper(GenrePredictorInterface):
    def __init__(self, model, device, treshold = None):
        self.model = model
        self.device = device
        self.treshold = treshold

    def predict(self, batch_features: dict) -> np.array:
        self.model.eval()
        with torch.no_grad():
            input_ids = batch_features['input_ids'].to(self.device)
            attention_mask = batch_features['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            if not self.treshold:
                self.treshold = 0.1
            preds = (probs > self.treshold).astype(int)
        return preds
