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
import logging
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Suppress Hugging Face warnings and set logging level
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

class TextClassifier:
    def __init__(self, use_gpu=True, automation=False, retrain=True, model_exists=True, verbose=False):
        self.model_file_path = "./AI_Page_Classifier/model.pt" if os.path.exists("./AI_Page_Classifier/model.pt") else "./model.pt"
        self.training_data_file_path = "./AI_Page_Classifier/training_data_v2.json" if os.path.exists("./AI_Page_Classifier/training_data_v2.json") else "./training_data_v2.json"
        self.verbose = verbose
        
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if self.verbose:
            if use_gpu and self.device.type != "cuda":
                print("Error: GPU requested but not available. Ensure NVIDIA drivers and Container Toolkit are installed.")
            else:
                print(f"Using {'GPU' if self.device.type == 'cuda' else 'CPU'} for training.")
        
        # Disable tqdm progress bars if not verbose
        if not verbose:
            try:
                from tqdm import tqdm
                # Create a dummy tqdm class to avoid AttributeError
                class DummyTqdm:
                    def __init__(self, *args, **kwargs):
                        pass
                    def update(self, n=1):
                        pass
                    def close(self):
                        pass
                    def set_description(self, desc=None):
                        pass
                    def set_postfix(self, **kwargs):
                        pass
                tqdm.tqdm = DummyTqdm
            except ImportError:
                pass
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', use_fast=False)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=1
            ).to(self.device)
        except Exception as e:
            sys.stdout = sys.__stdout__  # Restore stdout for error reporting
            print(f"Error: Failed to load tokenizer or model: {str(e)}")
            print("Ensure no local 'distilbert-base-uncased' directory exists and check network connectivity.")
            print("You can also try clearing the Hugging Face cache with: rm -rf ~/.cache/huggingface")
            sys.exit(1)
        
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        if self.verbose and not automation and (retrain or not model_exists):
            print(f"Initialized for {'GPU' if self.device.type == 'cuda' else 'CPU'} training.")

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
                if self.verbose:
                    print(f"Warning: Skipping invalid entry at index {i}: not a dictionary")
                continue
            if 'Text' not in item or 'Target' not in item:
                if self.verbose:
                    print(f"Warning: Skipping invalid entry at index {i}: missing 'Text' or 'Target'")
                continue
            if not isinstance(item['Text'], str) or not isinstance(item['Target'], (int, float)):
                if self.verbose:
                    print(f"Warning: Skipping invalid entry at index {i}: 'Text' must be string, 'Target' must be number")
                continue
            validated_data.append((item['Text'], float(item['Target'])))
        
        if not validated_data:
            raise ValueError("No valid training data entries found after validation.")
        
        if self.verbose:
            positive = sum(1 for _, target in validated_data if target == 1.0)
            negative = sum(1 for _, target in validated_data if target == 0.0)
            print(f"Training data stats: Positive examples (Interesting): {positive}, Negative examples (Not Interesting): {negative}")
        
        texts, targets = zip(*validated_data)
        return list(texts), list(targets)

    def train(self, max_threads=1, batch_size=16):
        texts, targets = self._load_training_data()
        dataset = TextDataset(texts, targets, self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max_threads
        )
        
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        
        start_time = time.time()
        best_loss = float('inf')
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(10):
            epoch_loss = 0.0
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits.squeeze()
                
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(data_loader)
            if self.verbose:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                self.save(self.model_file_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Fine-tuning took {elapsed_time:.2f} seconds")
        return elapsed_time

    def save(self, file_path):
        torch.save({
            'state_dict': self.model.state_dict(),
        }, file_path)

    def load(self, automation=False, retrain=True, model_exists=True):
        if os.path.exists(self.model_file_path):
            checkpoint = torch.load(self.model_file_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            self.model.to(self.device)
        else:
            raise FileNotFoundError(f"No saved model found at {self.model_file_path}. Please train the model first.")

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(
            text.lower(),
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits.squeeze()
            prob = torch.sigmoid(output).item()
        
        return (prob > 0.5, prob, None, None)

def main():
    global args
    parser = argparse.ArgumentParser(
        description="Text classifier using DistilBERT for predicting interesting pages.",
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
        help="Number of threads for training (default: 1)"
    )

    args = parser.parse_args()

    # Redirect stdout to /dev/null if not verbose or automation
    original_stdout = sys.stdout
    if not args.verbose and not args.automation:
        sys.stdout = open(os.devnull, 'w')

    try:
        if not args.text and not args.test_file:
            error_msg = "Please provide either a test string or a test file using --test-file"
            if args.automation:
                sys.stdout = original_stdout
                print(json.dumps({"error": error_msg}, indent=2))
            else:
                sys.stdout = original_stdout
                print(f"Error: {error_msg}.")
                print("Example: python text_classifier.py \"404 Not Found\" [--retrain] [--threads 4] [--gpu] [--verbose] [--automation]")
                print("Or: python text_classifier.py --test-file test_data.txt [--retrain] [--threads 4] [--gpu] [--verbose]")
            sys.exit(1)
        
        if args.automation and (args.test_file or not args.text):
            sys.stdout = original_stdout
            print(json.dumps({"error": "--automation requires a single test string, not a test file"}, indent=2))
            sys.exit(1)
        
        if args.test_file and not args.automation and not os.path.exists(args.test_file):
            sys.stdout = original_stdout
            error_msg = f"Test file '{args.test_file}' not found"
            print(f"Error: {error_msg}.")
            sys.exit(1)
        
        model_exists = os.path.exists("./AI_Page_Classifier/model.pt") or os.path.exists("./model.pt")
        classifier = TextClassifier(
            use_gpu=args.gpu,
            automation=args.automation,
            retrain=args.retrain,
            model_exists=model_exists,
            verbose=args.verbose
        )
        
        if args.retrain or (not model_exists and not args.automation):
            if args.automation:
                sys.stdout = original_stdout
                print(json.dumps({"error": "Model file 'model.pt' missing. Training not allowed in automation mode"}, indent=2))
                sys.exit(1)
            if args.verbose:
                print(f"Fine-tuning DistilBERT with {args.threads} thread{'s' if args.threads > 1 else ''} {'on GPU' if classifier.device.type == 'cuda' else 'on CPU'}...")
            elapsed_time = classifier.train(args.threads)
            if args.verbose:
                print("Model fine-tuned and saved.")
        else:
            if args.verbose:
                print("Loading fine-tuned model...")
            classifier.load(args.automation, args.retrain, model_exists)
            if args.verbose:
                print("Model loaded.")
        
        if args.automation:
            if args.verbose:
                print(f"Predicting for text: {args.text}")
            is_error, probability, _, _ = classifier.predict(args.text)
            sys.stdout = original_stdout
            print(json.dumps({
                "text": base64.b64encode(args.text.encode('utf-8')).decode('utf-8'),
                "probability": probability,
                "is_interesting": is_error
            }, indent=2))
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
                    is_error, probability, _, _ = classifier.predict(test_text)
                    
                    if args.verbose:
                        print(f"Text: {test_text}")
                        print(f"Predicted: {'Interesting Page' if is_error else 'Not Interesting Page'}, Probability: {probability:.4f}")
                        print("-" * 50)
                    
                    if is_error:
                        positive_count += 1
                    else:
                        negative_count += 1
        else:
            total_tests = 1
            if args.verbose:
                print(f"Predicting for text: {args.text}")
            is_error, probability, _, _ = classifier.predict(args.text)
            
            if not args.automation:
                sys.stdout = original_stdout
                print(f"Text: {args.text}")
                print(f"Predicted: {'Interesting Page' if is_error else 'Not Interesting Page'}, Probability: {probability:.4f}")
            
            if is_error:
                positive_count += 1
            else:
                negative_count += 1
        
        if not args.automation:
            sys.stdout = original_stdout
            print(f"Test Summary:")
            print(f"Total Tests: {total_tests}")
            print(f"Positive Cases (Interesting): {positive_count}")
            print(f"Negative Cases (Not Interesting): {negative_count}")
    
    except Exception as ex:
        sys.stdout = original_stdout
        if args.automation:
            print(json.dumps({"error": str(ex)}, indent=2))
        else:
            print(f"Error: {str(ex)}")
        sys.exit(1)
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()