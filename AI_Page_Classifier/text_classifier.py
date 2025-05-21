import torch
import torch.nn as nn
import json
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, use_gpu=True):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("GPU initialized for training.")
        else:
            print("Using CPU for training.")
        
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
    def _initialize_weights(self):
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        nn.init.normal_(self.layer1.weight, mean=0, std=scale)
        nn.init.zeros_(self.layer1.bias)
        
        scale = np.sqrt(2.0 / (self.hidden_size + self.output_size))
        nn.init.normal_(self.layer2.weight, mean=0, std=scale)
        nn.init.zeros_(self.layer2.bias)
    
    def forward(self, x, training=False):
        x = self.layer1(x)
        x = self.relu(x)
        if training:
            x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    
    def train_step(self, inputs, targets):
        self.train()
        inputs = inputs.to(self.device)  # Already a tensor
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self(inputs, training=True)
        loss = nn.BCELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()
    
    def save(self, file_path):
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }, file_path)
    
    @staticmethod
    def load(file_path, learning_rate=0.01, use_gpu=True):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found.")
        
        checkpoint = torch.load(file_path, map_location='cpu')
        nn = NeuralNetwork(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            learning_rate=learning_rate,
            use_gpu=use_gpu
        )
        nn.load_state_dict(checkpoint['state_dict'])
        nn.to(nn.device)
        return nn

class TextClassifier:
    def __init__(self, use_gpu=True):
        self.model_file_path = "model.pt"
        self.training_data_file_path = "training_data.json"
        self.vocabulary_file_path = "vocabulary.json"
        
        self.vocabulary = self._load_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.nn = NeuralNetwork(
            input_size=self.vocab_size,
            hidden_size=16,
            output_size=1,
            learning_rate=0.1,
            use_gpu=use_gpu
        )
    
    def _load_vocabulary(self):
        if not os.path.exists(self.vocabulary_file_path):
            raise FileNotFoundError(f"Vocabulary file {self.vocabulary_file_path} not found.")
        
        try:
            with open(self.vocabulary_file_path, 'r') as f:
                vocab = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse vocabulary file: {e}")
        
        if not vocab or not isinstance(vocab, list):
            raise ValueError("Vocabulary must be a non-empty list of words.")
        
        return vocab
    
    def _text_to_features(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        matched_words = []
        
        features = np.zeros(self.vocab_size)
        total_words = len(words)
        
        if total_words == 0:
            return features, matched_words
        
        for i, vocab_word in enumerate(self.vocabulary):
            count = sum(1 for w in words if w.lower() == vocab_word.lower())
            if count > 0:
                matched_words.append(f"{vocab_word} ({count})")
            features[i] = count / total_words
        
        return features, matched_words
    
    def _load_training_data(self):
        if not os.path.exists(self.training_data_file_path):
            raise FileNotFoundError(f"Training data file {self.training_data_file_path} not found.")
        
        try:
            with open(self.training_data_file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse training data file: {e}")
        
        if not data or not isinstance(data, list):
            raise ValueError("Training data must be a non-empty list.")
        
        validated_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: Skipping invalid entry at index {i}: not a dictionary")
                continue
            if 'Text' not in item or 'Target' not in item:
                print(f"Warning: Skipping invalid entry at index {i}: missing 'text' or 'target'")
                continue
            if not isinstance(item['Text'], str) or not isinstance(item['Target'], (int, float)):
                print(f"Warning: Skipping invalid entry at index {i}: 'text' must be string, 'target' must be number")
                continue
            validated_data.append((item['Text'], np.array([float(item['Target'])])))
        
        if not validated_data:
            raise ValueError("No valid training data entries found after validation.")
        
        return validated_data
    
    def train(self, max_threads=1, batch_size=32):
        training_data = self._load_training_data()
        data_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                [self._text_to_features(d[0].lower())[0] for d in batch],
                [d[1] for d in batch]
            )
        )
        
        start_time = time.time()
        for epoch in range(1000):
            for batch_features, batch_targets in data_loader:
                features = torch.tensor(batch_features, dtype=torch.float32).to(self.nn.device)
                targets = torch.tensor(batch_targets, dtype=torch.float32).to(self.nn.device)
                self.nn.train_step(features, targets)
        elapsed_time = time.time() - start_time
        print(f"Training took {elapsed_time:.2f} seconds")
        self.nn.save(self.model_file_path)
    
    def load(self):
        if os.path.exists(self.model_file_path):
            self.nn = NeuralNetwork.load(self.model_file_path, learning_rate=0.1, use_gpu=self.nn.device.type == "cuda")
        else:
            raise FileNotFoundError(f"No saved model found at {self.model_file_path}. Please train the model first.")
    
    def predict(self, text):
        self.nn.eval()
        features, matched_words = self._text_to_features(text.lower())
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.nn.device)
        
        with torch.no_grad():
            output = self.nn(features_tensor)
            prob = output.item()
        
        return (prob > 0.5, prob, features, matched_words)

def main():
    retrain = '--retrain' in sys.argv
    max_threads = 1  # Default to 1 thread
    ocr_text = None
    use_gpu = '--gpu' in sys.argv or torch.cuda.is_available()  # Prefer GPU if available
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--threads' and i + 1 < len(sys.argv):
            try:
                max_threads = max(1, min(int(sys.argv[i + 1]), os.cpu_count()))
                i += 2
            except ValueError:
                i += 1
        elif sys.argv[i] not in ['--retrain', '--gpu'] and ocr_text is None:
            ocr_text = sys.argv[i]
            i += 1
        else:
            i += 1
    
    if not ocr_text:
        print("Error: Please provide OCR'd text as a command-line argument.")
        print("Example: python text_classifier.py \"404 Not Found Page Missing\" [--retrain] [--threads 4] [--gpu]")
        return
    
    try:
        classifier = TextClassifier(use_gpu)
        
        if retrain or not os.path.exists("model.pt"):
            print(f"Training model with {max_threads} thread{'s' if max_threads > 1 else ''} {'on GPU' if classifier.nn.device.type == 'cuda' else 'on CPU'}...")
            classifier.train(max_threads)
            print("Model trained and saved.")
        else:
            print("Loading saved model...")
            classifier.load()
            print("Model loaded.")
        
        is_error, probability, features, matched_words = classifier.predict(ocr_text)
        print(f"Text: {ocr_text}")
        print(f"Predicted: {'Error Page' if is_error else 'Not Error Page'}, Probability: {probability:.4f}")
        print(f"Matched Words: {', '.join(matched_words) if matched_words else 'None'}")
    
    except Exception as ex:
        print(f"Error: {str(ex)}")

if __name__ == "__main__":
    main()