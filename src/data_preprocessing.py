import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import re
import os


class PhishingEmailDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataPreprocessor:
    def __init__(self, tokenizer_name: str = 'distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'http\S+', ' [URL] ', text)
        text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(data_path)
        
        df['text_combined'] = df['text_combined'].apply(self.clean_text)
        
        df = df.dropna(subset=['text_combined', 'label'])
        
        df['label'] = df['label'].astype(int)
        
        X = df['text_combined']
        y = df['label']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        return train_df, val_df, test_df
    
    def create_data_loaders(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        batch_size: int = 16,
        max_length: int = 512
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        train_dataset = PhishingEmailDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        val_dataset = PhishingEmailDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        test_dataset = PhishingEmailDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_label_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        label_counts = train_df['label'].value_counts().sort_index()
        total = len(train_df)
        weights = total / (len(label_counts) * label_counts)
        return torch.tensor(weights.values, dtype=torch.float32)