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
import argparse

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.005, use_gpu=True, automation=False, retrain=True, model_exists=True):
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
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Increased from 0.3
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)
    
    def forward(self, x, training=False):
        x = self.layer1(x)
        x = self.relu(x)
        if training:
            x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x
    
    def train_step(self, inputs, targets, criterion):
        self.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self(inputs, training=True)
        loss = criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save(self, file_path):
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }, file_path)
    
    @staticmethod
    def load(file_path, learning_rate=0.005, use_gpu=True, automation=False, retrain=True, model_exists=True):
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
        self.model_file_path = "./AI_Page_Classifier/model.pt" if os.path.exists("./AI_Page_Classifier/model.pt") else "./model.pt"
        self.training_data_file_path = "./AI_Page_Classifier/training_data_v2.json" if os.path.exists("./AI_Page_Classifier/training_data_v2.json") else "./training_data_v2.json"
        self.vocabulary_file_path = "./AI_Page_Classifier/vocabulary.json" if os.path.exists("./AI_Page_Classifier/vocabulary.json") else "./vocabulary.json"
        
        self.vocabulary = self._load_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.nn = NeuralNetwork(
            input_size=self.vocab_size,
            hidden_size=64,
            output_size=1,
            learning_rate=0.005,
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
        text = text.replace('1', 'l').replace('0', 'o').replace('|', 'l')
        words = text.lower().split()
        matched_words = []
        features = np.zeros(self.vocab_size)
        
        if not words:
            return features, matched_words
        
        processed_words = []
        i = 0
        while i < len(words):
            if words[i] == "not" and i + 1 < len(words) and words[i + 1] == "interesting":
                processed_words.append("not_interesting")
                i += 2
            else:
                processed_words.append(words[i])
                i += 1
        
        for i, vocab_word in enumerate(self.vocabulary):
            count = sum(1 for w in processed_words if w.lower() == vocab_word.lower())
            if count > 0:
                matched_words.append(f"{vocab_word} ({count})")
                features[i] = 1.0
        if args.verbose:
            print(f"Processed words: {processed_words}")
            print(f"Non-zero feature indices: {np.nonzero(features)[0]}")
            print(f"Matched words (raw): {matched_words}")
        
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
                print(f"Warning: Skipping invalid entry at index {i}: missing 'Text' or 'Target'")
                continue
            if not isinstance(item['Text'], str) or not isinstance(item['Target'], (int, float)):
                print(f"Warning: Skipping invalid entry at index {i}: 'Text' must be string, 'Target' must be number")
                continue
            validated_data.append((item['Text'], np.array([float(item['Target'])])))
        
        if not validated_data:
            raise ValueError("No valid training data entries found after validation.")
        
        positive = sum(1 for _, target in validated_data if target[0] == 1.0)
        negative = sum(1 for _, target in validated_data if target[0] == 0.0)
        print(f"Training data stats: Positive examples (Interesting): {positive}, Negative examples (Not Interesting): {negative}")
        
        return validated_data
    
    def train(self, max_threads=1, batch_size=32):
        training_data = self._load_training_data()
        data_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max_threads,
            collate_fn=lambda batch: (
                [self._text_to_features(d[0].lower())[0] for d in batch],
                [d[1] for d in batch]
            )
        )
        
        criterion = nn.BCELoss(reduction='none')
        class_weights = torch.tensor([1.0, 2.5]).to(self.nn.device)
        
        start_time = time.time()
        best_loss = float('inf')
        patience = 15
        epochs_no_improve = 0
        
        for epoch in range(1000):
            epoch_loss = 0.0
            for batch_features, batch_targets in data_loader:
                features = torch.from_numpy(np.stack(batch_features)).float().to(self.nn.device)
                targets = torch.from_numpy(np.stack(batch_targets)).float().to(self.nn.device)
                weights = torch.where(targets == 0.0, class_weights[1], class_weights[0]).to(self.nn.device)
                outputs = self.nn(features, training=True)
                loss = criterion(outputs, targets)
                weighted_loss = (loss * weights).mean()
                self.nn.optimizer.zero_grad()
                weighted_loss.backward()
                self.nn.optimizer.step()
                epoch_loss += weighted_loss.item()
            
            epoch_loss /= len(data_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def load(self, automation=False, retrain=True, model_exists=True):
        if os.path.exists(self.model_file_path):
            self.nn = NeuralNetwork.load(
                self.model_file_path,
                learning_rate=0.005,
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
        if self.nn.device.type == "cuda":
            torch.cuda.empty_cache()
        if args.verbose:
            print(f"Feature vector sum: {sum(features)}")
        if not any(features):
            if args.verbose:
                print("No features matched, returning non-interesting")
            return (False, 0.0, features, matched_words)
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.nn.device)
        
        with torch.no_grad():
            output = self.nn(features_tensor)
            prob = output.item()
        
        return (prob > 0.5, prob, features, matched_words)

def main():
    global args
    parser = argparse.ArgumentParser(
        description="Text classifier for predicting interesting pages.",
        epilog="Example: python3 text_classifier.py \"404 Not Found\" --automation"
    )
    parser.add_argument(
        "text",
        type=str,
        nargs='?',
        help="Input text to classify (required unless --test-file is used)"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model even if model.pt exists"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--automation",
        action="store_true",
        help="Output JSON for automation (requires single text input)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to a file containing test texts (one per line)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for training (default: 1, max: 32)"
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if args.automation:
            print(json.dumps({"error": "Invalid arguments"}, indent=2))
        sys.exit(e.code)

    if not args.text and not args.test_file:
        error_msg = "Please provide either a test string or a test file using --test-file"
        if args.automation:
            print(json.dumps({"error": error_msg}, indent=2))
        else:
            print(f"Error: {error_msg}.")
            print("Example: python text_classifier.py \"404 Not Found\" [--retrain] [--threads 4] [--gpu] [--verbose] [--automation]")
            print("Or: python text_classifier.py --test-file test_data.txt [--retrain] [--threads 4] [--gpu] [--verbose]")
        sys.exit(1)
    
    if args.automation and (args.test_file or not args.text):
        print(json.dumps({"error": "--automation requires a single test string, not a test file"}, indent=2))
        sys.exit(1)
    
    if args.test_file and not os.path.exists(args.test_file):
        error_msg = f"Test file '{args.test_file}' not found"
        if args.automation:
            print(json.dumps({"error": error_msg}, indent=2))
        else:
            print(f"Error: {error_msg}.")
        sys.exit(1)
    
    max_threads = max(1, min(args.threads, 32))
    if args.threads != max_threads:
        if args.verbose:
            print(f"Warning: Threads adjusted to {max_threads} (requested: {args.threads}, max: 32)")
    
    try:
        model_exists = os.path.exists("./AI_Page_Classifier/model.pt") or os.path.exists("./model.pt")
        classifier = TextClassifier(
            use_gpu=args.gpu,
            automation=args.automation,
            retrain=args.retrain,
            model_exists=model_exists
        )
        
        if args.retrain or (not model_exists and not args.automation):
            if args.automation:
                print(json.dumps({"error": "Model file 'model.pt' missing. Training not allowed in automation mode"}, indent=2))
                sys.exit(1)
            if args.verbose:
                print(f"Training model with {max_threads} thread{'s' if max_threads > 1 else ''} {'on GPU' if classifier.nn.device.type == 'cuda' else 'on CPU'}...")
            elapsed_time = classifier.train(max_threads)
            if args.verbose:
                print(f"Training took {elapsed_time:.2f} seconds")
            classifier.nn.save(classifier.model_file_path)
            if args.verbose:
                print("Model trained and saved.")
        else:
            if args.verbose:
                print("Loading saved model...")
            classifier.load(args.automation, args.retrain, model_exists)
            if args.verbose:
                print("Model loaded.")
        
        if args.automation:
            if args.verbose:
                print(f"Predicting for text: {args.text}")
            is_error, probability, _, _ = classifier.predict(args.text)
            result = {
                "text": base64.b64encode(args.text.encode('utf-8')).decode('utf-8'),
                "probability": probability,
                "is_interesting": is_error
            }
            print(json.dumps(result, indent=2))
            return
        
        positive_count = 0
        negative_count = 0
        total_tests = 0
        
        if args.test_file:
            with open(args.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    test_text = line.strip()
                    if not test_text:
                        continue
                    
                    total_tests += 1
                    is_error, probability, features, matched_words = classifier.predict(test_text)
                    
                    if args.verbose:
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
            if args.verbose:
                print(f"Predicting for text: {args.text}")
            is_error, probability, features, matched_words = classifier.predict(args.text)
            
            if args.verbose or not args.automation:
                print(f"Text: {args.text}")
                print(f"Predicted: {'Interesting Page' if is_error else 'Not Interesting Page'}, Probability: {probability:.4f}")
                print(f"Matched Words: {', '.join(matched_words) if matched_words else 'None'}")
            
            if is_error:
                positive_count += 1
            else:
                negative_count += 1
        
        if args.verbose or not args.automation:
            print(f"Test Summary:")
            print(f"Total Tests: {total_tests}")
            print(f"Positive Cases (Interesting): {positive_count}")
            print(f"Negative Cases (Not Interesting): {negative_count}")
    
    except Exception as ex:
        if args.automation:
            print(json.dumps({"error": str(ex)}, indent=2))
        else:
            print(f"Error: {str(ex)}")
        sys.exit(1)

if __name__ == "__main__":
    main()