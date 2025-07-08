import os
os.environ['OTEL_SDK_DISABLED'] = 'true'

from flask import Flask, render_template, request, jsonify
import torch
from datetime import datetime
import re
from pathlib import Path

from src.inference import PhishingDetector
from src.model import DistilBertForPhishingDetection
from transformers import DistilBertTokenizer

app = Flask(__name__)

# Global variable to store the detector
detector = None

def get_latest_model():
    """Find the latest trained model in the models directory"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Look for model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        return None
    
    # Sort by modification time and get the latest
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    
    # Look for checkpoint files
    checkpoint_files = list(latest_dir.rglob("checkpoint.pt"))
    if checkpoint_files:
        return str(checkpoint_files[0])
    
    return None

def load_detector():
    """Load the phishing detector model"""
    global detector
    
    # Try to find a trained model
    checkpoint_path = get_latest_model()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained model from: {checkpoint_path}")
        detector = PhishingDetector(checkpoint_path)
    else:
        # Fallback: use base DistilBERT model without fine-tuning
        print("No trained model found. Using base DistilBERT model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple detector with base model
        class SimpleDetector:
            def __init__(self):
                self.device = device
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.model = DistilBertForPhishingDetection()
                self.model.to(device)
                self.model.eval()
            
            def analyze_email(self, email_text):
                # Simple heuristic-based detection for demo
                suspicious_keywords = [
                    'urgent', 'verify', 'suspend', 'click here', 'act now',
                    'limited time', 'winner', 'prize', 'congratulations',
                    'update your', 'confirm your', 'validate your'
                ]
                
                email_lower = email_text.lower()
                suspicious_count = sum(1 for keyword in suspicious_keywords if keyword in email_lower)
                
                # Calculate risk based on keywords
                if suspicious_count >= 3:
                    risk_score = 0.9
                    prediction = 'phishing'
                    risk_level = 'High Risk'
                elif suspicious_count >= 1:
                    risk_score = 0.6
                    prediction = 'suspicious'
                    risk_level = 'Medium Risk'
                else:
                    risk_score = 0.2
                    prediction = 'legitimate'
                    risk_level = 'Low Risk'
                
                return {
                    'prediction': prediction,
                    'confidence': 1.0 - abs(0.5 - risk_score) * 2,  # Convert to confidence
                    'probabilities': {
                        'legitimate': 1 - risk_score,
                        'phishing': risk_score
                    },
                    'risk_level': risk_level,
                    'suspicious_patterns': [kw for kw in suspicious_keywords if kw in email_lower][:5],
                    'recommendation': self._get_recommendation(risk_score)
                }
            
            def _get_recommendation(self, risk_score):
                if risk_score > 0.8:
                    return "HIGH RISK: This email appears to be phishing. Do not click any links or provide personal information."
                elif risk_score > 0.5:
                    return "CAUTION: This email shows signs of potential phishing. Verify the sender before taking any action."
                else:
                    return "LIKELY SAFE: This email appears to be legitimate, but always exercise caution with unexpected emails."
        
        detector = SimpleDetector()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze email for phishing"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text.strip():
            return jsonify({'error': 'Please provide email text to analyze'}), 400
        
        # Analyze the email
        result = detector.analyze_email(email_text)
        
        # Format the response
        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': f"{result['confidence']:.1%}",
            'risk_level': result['risk_level'],
            'phishing_probability': f"{result['probabilities']['phishing']:.1%}",
            'legitimate_probability': f"{result['probabilities']['legitimate']:.1%}",
            'suspicious_patterns': result.get('suspicious_patterns', []),
            'recommendation': result['recommendation']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Phishing Email Detector UI...")
    load_detector()
    print("Model loaded. Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)