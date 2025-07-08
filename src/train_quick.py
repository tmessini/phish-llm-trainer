import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

import torch
from datetime import datetime
from .model import DistilBertForPhishingDetection, ModelConfig
from .data_preprocessing import DataPreprocessor
from .train import Trainer

def main():
    # Quick training configuration
    config = ModelConfig(
        num_epochs=1,  # Just 1 epoch for quick results
        batch_size=16,
        learning_rate=2e-5,
        max_length=512,
        dropout_rate=0.3,
        warmup_steps=100,  # Reduced warmup
        logging_steps=10,  # Log every 10 steps
        save_steps=100,  # Save every 100 steps instead of 500
        eval_steps=200   # Evaluate every 200 steps
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = DataPreprocessor()
    
    # Load and split data
    import os as os_module
    data_path = os_module.path.join(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))), "datasets", "phishing_email.csv")
    train_df, val_df, test_df = preprocessor.load_and_preprocess_data(data_path)
    
    # Use smaller validation set for faster evaluation
    val_df = val_df.sample(n=min(1000, len(val_df)), random_state=42)
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)} (reduced for quick training)")
    print(f"Test size: {len(test_df)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        train_df, val_df, test_df,
        batch_size=config.batch_size,
        max_length=config.max_length
    )
    
    # Initialize model
    model = DistilBertForPhishingDetection(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout_rate=config.dropout_rate,
        freeze_bert=config.freeze_bert
    )
    
    # Create trainer with custom output directory
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=f"./models/phishing_detector_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print("\nStarting quick training...")
    print(f"Model will be saved every {config.save_steps} steps")
    print(f"First save will happen in approximately {config.save_steps * config.batch_size / 60:.1f} minutes\n")
    
    trainer.train()
    
    print("\nTraining completed! Check the models directory for saved checkpoints.")

if __name__ == "__main__":
    main()