import pandas as pd
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from utils import DatasetTypes

class LyricsGenreDataset(Dataset):
    def __init__(self, lyrics_list, features, labels, tokenizer, max_length=512):
        self.lyrics = lyrics_list
        self.features = features  # pandas df
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        text = self.lyrics[idx]
        label = self.labels[idx]
        features = self.features.iloc[idx].to_dict()  # <--- превращаем в словарь

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [seq_len]
            'labels': torch.tensor(label, dtype=torch.float),  # [num_labels]
            'features': features  # <--- теперь это dict, collate будет работать
        }


def one_hot_encoded_to_genre_list(predictions, idx2genre: dict = None):
    ''' Predictions is array on n_genres size, where 1 if lyrics belongs to that genre and 0 if not'''    
    genre_list = []
    for i, value in enumerate(predictions):
        if value == 1:
            genre_list.append(idx2genre[i])
    
    return genre_list


def get_datasets(df_path, tokenizer, dataset_type=DatasetTypes.whole, debug=False, train_size=0.7, test_size=0.15, val_size=0.15, random_seed=1337):
    ''' Params:
            df_path - path to .csv format file. Expected that it have lyrics and genre fields as base. Other fields will go to feature field of dataset
        Returns:
            dicts with torch datasets and some extra objects'''
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    df = pd.read_csv(df_path)
    df = df[:1000] if dataset_type == DatasetTypes.small else df  # For experiments use

    if debug: 
        logger.info(str(df.head()))
        
    target = df['genre'].unique()
    
    if debug:
        logger.info(str(sorted(target)))

    all_genre_strings = df['genre'].unique()

    # Разделяем по запятой и складываем в множество (чтобы получить только уникальные жанры)
    all_genres = set()

    for genre_string in all_genre_strings:
        genres = genre_string.split(',')
        all_genres.update(genres)

    # Преобразуем в отсортированный список (по желанию)
    all_genres_list = sorted(all_genres)

    if debug:
        logger.info(str(all_genres_list))
        logger.info(str(f'Genres length: {len(all_genres_list)}'))
        
    df['genre_list'] = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')])

    # Используем MultiLabelBinarizer для преобразования в one-hot
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genre_list'])
    genres = mlb.classes_

    if debug:
        logger.info(f"Unique genres number: {len(mlb.classes_)}")
        
    idx2genre = {i: genre for i, genre in enumerate(genres)}
    genre2idx = {genre: i for i, genre in enumerate(genres)}
    
    if debug:
        genres_count = {genre_name: 0 for genre_name in genres}
        for index, row in df.iterrows():
            for genre in genres:
                if genre in row['genre_list']:
                    genres_count[genre] += 1

        logger.info('Genres count')
        for key, value in genres_count.items():
            logger.info(f"{key}: {value}")

    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_shuffled['genre_list'] = df_shuffled['genre'].apply(lambda x: [g.strip() for g in x.split(',')])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_shuffled['genre_list'])

    X_temp, X_test, y_temp, y_test = train_test_split(
        df_shuffled, y, test_size=test_size, random_state=random_seed
    )

    # Отношение валидационной выборки к оставшемуся (train + val)
    val_ratio_of_temp = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=random_seed
    )

    if debug:
        logger.info('Dataset sizes:')
        logger.info('Train size:', len(X_train))
        logger.info('Val size:', len(X_val))
        logger.info('Test size:', len(X_test))
        
    train_dataset = LyricsGenreDataset(X_train['lyrics'].tolist(), X_train, y_train, tokenizer)
    val_dataset = LyricsGenreDataset(X_val['lyrics'].tolist(), X_val, y_val, tokenizer)
    test_dataset = LyricsGenreDataset(X_test['lyrics'].tolist(), X_test, y_test, tokenizer)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'genres': genres,
        'idx2genre': idx2genre,
        'genre2idx': genre2idx}
    
    
def get_dataloaders(train_dataset: LyricsGenreDataset, val_dataset: LyricsGenreDataset, test_dataset: LyricsGenreDataset, batch_size):
    # We eill use custion collate fn because we want features dict be in our dataset
    def custom_collate_fn(batch):
        batch_dict = default_collate([
            {k: v for k, v in item.items() if k != 'features'}
            for item in batch
        ])

        # Собираем features отдельно
        if 'features' in batch[0]:
            feature_dicts = [item['features'] for item in batch]
            batch_dict['features'] = feature_dicts  # просто список словарей

        return batch_dict
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, test_loader
