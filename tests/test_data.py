import sys
import os
import pytest
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data import get_datasets
from utils import DatasetTypes

def test_get_datasets():
    dataset_path = r'data\spotify_480k_with_features.csv' 
    if not os.path.exists(dataset_path):
        pytest.skip("Dataset file 480k with features not exists in data directory")
    
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    datasets_dict = get_datasets(dataset_path, tokenizer, dataset_type=DatasetTypes.small, debug=True)
    
    train_dataset = datasets_dict['train_dataset']
    val_dataset = datasets_dict['val_dataset']
    test_dataset = datasets_dict['test_dataset']
    
    # Try get some item from datasets:
    assert train_dataset[0], 'Cannot take item from train dataset!'
    assert val_dataset[0], 'Cannot take item from val dataset!'
    assert test_dataset[0], 'Cannot take item from test dataset!'
    
    genres = datasets_dict['genres']
    assert len(genres) != 0, 'genres is empty!'
    
    idx2genre = datasets_dict['idx2genre']
    assert idx2genre, 'idx2genre is empty!'
    
    genre2idx = datasets_dict['genre2idx']
    assert genre2idx, 'genre2idx is empty!'
    