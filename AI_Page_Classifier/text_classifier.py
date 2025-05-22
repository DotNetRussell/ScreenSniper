import torch
import torch.nn as nn
import json
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import base64

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, use_gpu=True, automation=False, retrain=True, model_exists=True):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if not automation and (retrain or not model_exists):
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
    def load(file_path, learning_rate=0.01, use_gpu=True, automation=False, retrain=True, model_exists=True):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found.")
        
        checkpoint = torch.load(file_path, map_location='cpu')
        nn = NeuralNetwork(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            automation=automation,
            retrain=retrain,
            model_exists=model_exists
        )
        nn.load_state_dict(checkpoint['state_dict'])
        nn.to(nn.device)
        return nn

class TextClassifier:
    def __init__(self, use_gpu=True, automation=False, retrain=True, model_exists=True):
        self.model_file_path = "./AI_Page_Classifier/model.pt"
        self.training_data_file_path = "./AI_Page_Classifier/training_data.json"
        self.vocabulary_file_path = "./AI_Page_Classifier/vocabulary.json"
        
        self.vocabulary = self._load_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.nn = NeuralNetwork(
            input_size=self.vocab_size,
            hidden_size=16,
            output_size=1,
            learning_rate=0.1,
            use_gpu=use_gpu,
            automation=automation,
            retrain=retrain,
            model_exists=model_exists
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
                features = torch.from_numpy(np.stack(batch_features)).float().to(self.nn.device)
                targets = torch.from_numpy(np.stack(batch_targets)).float().to(self.nn.device)
                self.nn.train_step(features, targets)
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def load(self, automation=False, retrain=True, model_exists=True):
        if os.path.exists(self.model_file_path):
            self.nn = NeuralNetwork.load(
                self.model_file_path,
                learning_rate=0.1,
                use_gpu=self.nn.device.type == "cuda",
                automation=automation,
                retrain=retrain,
                model_exists=model_exists
            )
        else:
            raise FileNotFoundError(f"No saved model found at {self.model_file_path}. Please train the model first.")
    
    def predict(self, text):
        self.nn.eval()
        features, matched_words = self._text_to_features(text.lower())
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.nn.device)
        
        with torch.no_grad():
            output = self.nn(features_tensor)
            prob = output.item()
        
        return (prob > 0.8, prob, features, matched_words)

def main():
    # Debug: Uncomment to inspect arguments
    # print(f"sys.argv: {sys.argv}")
    
    retrain = '--retrain' in sys.argv
    verbose = '--verbose' in sys.argv
    automation = '--automation' in sys.argv
    use_gpu = '--gpu' in sys.argv or torch.cuda.is_available()
    test_file = None
    test_string = None
    max_threads = 1  # Default to 1 thread
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--threads' and i + 1 < len(sys.argv):
            try:
                max_threads = max(1, min(int(sys.argv[i + 1]), os.cpu_count()))
                i += 2
            except ValueError:
                print(f"Warning: Invalid threads value '{sys.argv[i + 1]}', using default (1)")
                i += 2
        elif arg == '--test-file' and i + 1 < len(sys.argv):
            test_file = sys.argv[i + 1]
            i += 2
        elif arg not in ['--retrain', '--gpu', '--verbose', '--automation', '--threads'] and test_string is None:
            test_string = sys.argv[i]
            i += 1
        else:
            i += 1  # Skip known flags or invalid args
    
    if not test_string and not test_file:
        if automation:
            print(json.dumps({"error": "Please provide a test string for automation mode"}, indent=2))
        else:
            print("Error: Please provide either a test string or a test file using --test-file <file_path>.")
            print("Example: python text_classifier.py \"404 Not Found\" [--retrain] [--threads 4] [--gpu] [--verbose] [--automation]")
            print("Or: python text_classifier.py --test-file test_data.txt [--retrain] [--threads 4] [--gpu] [--verbose]")
        return
    
    if automation and (test_file or not test_string):
        print(json.dumps({"error": "--automation requires a single test string, not a test file"}, indent=2))
        return
    
    if test_file and not os.path.exists(test_file):
        if automation:
            print(json.dumps({"error": f"Test file '{test_file}' not found"}, indent=2))
        else:
            print(f"Error: Test file '{test_file}' not found.")
        return
    
    try:
        model_exists = os.path.exists("./AI_Page_Classifier/model.pt")
        classifier = TextClassifier(use_gpu, automation, retrain, model_exists)
        
        if retrain or not model_exists:
            if automation:
                print(json.dumps({"error": "Model file 'model.pt' missing. Training not allowed in automation mode"}, indent=2))
                return
            if not automation:
                print(f"Training model with {max_threads} thread{'s' if max_threads > 1 else ''} {'on GPU' if classifier.nn.device.type == 'cuda' else 'on CPU'}...")
            elapsed_time = classifier.train(max_threads)
            if not automation:
                print(f"Training took {elapsed_time:.2f} seconds")
            classifier.nn.save(classifier.model_file_path)
            if not automation:
                print("Model trained and saved.")
        else:
            if not automation:
                print("Loading saved model...")
            classifier.load(automation, retrain, model_exists)
            if not automation:
                print("Model loaded.")
        
        if automation:
            is_error, probability, _, _ = classifier.predict(test_string)
            result = {
                "text": base64.b64encode(test_string.encode('utf-8')).decode('utf-8'),
                "probability": probability,
                "is_interesting": is_error
            }
            print(json.dumps(result, indent=2))
            return
        
        positive_count = 0
        negative_count = 0
        total_tests = 0
        
        if test_file:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    test_text = line.strip()
                    if not test_text:
                        continue  # Skip empty lines
                    
                    total_tests += 1
                    is_error, probability, features, matched_words = classifier.predict(test_text)
                    
                    if verbose:
                        print(f"Text: {test_text}")
                        print(f"Predicted: {'Interesting Page' if is_error else 'Not Interesting Page'}, Probability: {probability:.4f}")
                        print(f"Matched Words: {', '.join(matched_words) if matched_words else 'None'}")
                        print("-" * 50)
                    
                    if is_error:
                        positive_count += 1
                    else:
                        negative_count += 1
        else:
            total_tests = 1
            is_error, probability, features, matched_words = classifier.predict(test_string)
            
            if verbose or not verbose:  # Always print for single string
                print(f"Text: {test_string}")
                print(f"Predicted: {'Interesting Page' if is_error else 'Not Interesting Page'}, Probability: {probability:.4f}")
                print(f"Matched Words: {', '.join(matched_words) if matched_words else 'None'}")
            
            if is_error:
                positive_count += 1
            else:
                negative_count += 1
        
        print(f"Test Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Positive Cases (Interesting): {positive_count}")
        print(f"Negative Cases (Not Interesting): {negative_count}")
    
    except Exception as ex:
        if automation:
            print(json.dumps({"error": str(ex)}, indent=2))
        else:
            print(f"Error: {str(ex)}")

if __name__ == "__main__":
    main()