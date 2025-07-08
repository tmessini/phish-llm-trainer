import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

from .model import DistilBertForPhishingDetection, ModelConfig
from .data_preprocessing import DataPreprocessor


class Trainer:
    def __init__(
        self,
        model: DistilBertForPhishingDetection,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "./models"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.best_val_f1 = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            if (batch_idx + 1) % self.config.logging_steps == 0:
                self.logger.info(f'Epoch: {epoch}, Step: {batch_idx + 1}, Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self):
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            self.logger.info(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            self.training_history['train_loss'].append(train_loss)
            
            val_metrics = self.evaluate()
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_precision'].append(val_metrics['precision'])
            self.training_history['val_recall'].append(val_metrics['recall'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            self.logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            self.logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
            self.logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
            self.logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_model(f'best_model_epoch_{epoch}')
                self.logger.info(f"New best model saved with F1: {self.best_val_f1:.4f}")
        
        self.save_training_history()
        self.logger.info("\nTraining completed!")
        self.logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
    
    def save_model(self, model_name: str):
        model_path = os.path.join(self.output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }, os.path.join(model_path, 'checkpoint.pt'))
        
        self.model.distilbert.save_pretrained(model_path)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def save_training_history(self):
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")


def main():
    config = ModelConfig(
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=512,
        dropout_rate=0.3,
        warmup_steps=500,
        logging_steps=50
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = DataPreprocessor()
    
    import os
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "phishing_email.csv")
    train_df, val_df, test_df = preprocessor.load_and_preprocess_data(data_path)
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_df, val_df, test_df,
        batch_size=config.batch_size,
        max_length=config.max_length
    )
    
    model = DistilBertForPhishingDetection(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout_rate=config.dropout_rate,
        freeze_bert=config.freeze_bert
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=f"./models/phishing_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    trainer.train()


if __name__ == "__main__":
    main()