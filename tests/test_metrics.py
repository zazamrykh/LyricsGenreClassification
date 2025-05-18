# Test if our metrics evaluation is working. Implement model that will only return 1 for any input and class
import sys
import os
import pytest
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from metrics import GenrePredictorInterface, evaluate_model
from utils import DatasetTypes
from data import get_datasets, get_dataloaders

def test_evaluate_model():
    dataset_path = r'data\spotify_480k_with_features.csv' 
    if not os.path.exists(dataset_path):
        pytest.skip("Dataset file 480k with features not exists in data directory")
    
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    datasets_dict = get_datasets(dataset_path, tokenizer, dataset_type=DatasetTypes.small, debug=True)
    
    train_loader, val_loader, test_loader = get_dataloaders(datasets_dict['train_dataset'], datasets_dict['val_dataset'], datasets_dict['test_dataset'], 16)
    
    class DummyClassifier(GenrePredictorInterface):
        def predict(self, batch_features: dict) -> np.array:
            return np.ones_like(batch_features['labels'])
            
    dummy_model = DummyClassifier()

    metrics = evaluate_model(dummy_model, test_loader)

    assert metrics
    
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1-score:", metrics['f1'])
