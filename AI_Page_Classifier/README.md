AI Page Classifier
The AI_Page_Classifier directory contains a text classification system designed to identify web error pages (e.g., 404, 500, 403 errors) versus non-error pages using machine learning. It includes two implementations: a Python version using PyTorch and a C# version using Alea.CUDA, both leveraging GPU acceleration for training and inference. The classifiers analyze text extracted from web pages (e.g., via OCR) and predict whether the page indicates an error based on word frequency features.
Features

Dual Implementations:
Python (PyTorch): Modern, optimized for GPU acceleration, using a neural network with ReLU, dropout, and sigmoid activation.
C# (Alea.CUDA): Legacy implementation with manual CUDA kernel management, suitable for .NET environments.


GPU Support: Optimized for NVIDIA GPUs (tested with RTX 3090 Ti) using CUDA 12.4.
Text Classification: Detects error pages (e.g., "404 Not Found") vs. non-error pages (e.g., "Welcome to Our Site") using a vocabulary-based feature extraction approach.
Docker Integration: Runs in a containerized environment for easy setup and portability.
Vocabulary Generation: Automatically builds a vocabulary from training data for feature extraction.

Directory Contents

text_classifier.py: Python script implementing the text classifier using PyTorch.
TextClassifier.cs: C# source code for the text classifier using Alea.CUDA.
TextClassifier.csproj: C# project file defining dependencies (Alea, System.Text.Json).
training_data.json: Training dataset with text samples and labels (1 for error pages, 0 for non-error pages).
vocabulary.json: Vocabulary array of unique words used for feature extraction.
generate_vocabulary.py: Script to generate vocabulary.json from training_data.json.

Requirements

Hardware:
NVIDIA GPU (e.g., RTX 3090 Ti) with CUDA 11.1 or later.
Windows 10/11 with WSL 2 (for Docker Desktop).


Software:
Docker Desktop with WSL 2 backend and NVIDIA Container Toolkit.
NVIDIA drivers supporting CUDA 12.4 (e.g., driver version 545 or newer).
Git (for cloning the repository).


Dependencies (handled by Docker):
Python 3.10, PyTorch 2.3, NumPy (for Python implementation).
.NET 8 SDK, Alea.CUDA 3.0.4 (for C# implementation).



Setup
1. Clone the Repository
git clone https://github.com/DotNetRussell/ScreenSniper.git
cd ScreenSniper/AI_Page_Classifier

2. Prepare Docker Environment

Install Docker Desktop: Download from docker.com and enable WSL 2 backend (Settings > General > Use WSL 2 based engine).

Install NVIDIA Container Toolkit:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker


Verify GPU Access:
docker run --gpus all -it --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Ensure your RTX 3090 Ti is listed.


3. Build and Run the Container
Use the provided Dockerfile and docker-compose.yml in the parent directory to build a custom container with Python, PyTorch, .NET SDK, and Alea.CUDA.

Directory Structure:
ScreenSniper/
├── Dockerfile
├── docker-compose.yml
├── AI_Page_Classifier/
│   ├── text_classifier.py
│   ├── TextClassifier.cs
│   ├── TextClassifier.csproj
│   ├── training_data.json
│   ├── vocabulary.json
│   ├── generate_vocabulary.py


Build:
cd ScreenSniper
docker-compose build


Run:
docker-compose up

This starts an interactive shell in the container with AI_Page_Classifier mounted at /workspace.


Usage
Run Python Classifier
Train and predict with the PyTorch implementation:
python /workspace/text_classifier.py "404 Not Found" --retrain

Or use a pre-trained model:
python /workspace/text_classifier.py "404 Not Found"

Use --no-gpu to disable GPU acceleration if needed.
Run C# Classifier
Train and predict with the Alea.CUDA implementation:
dotnet /workspace/dotnet/bin/TextClassifier.dll "404 Not Found" --retrain --gpu

Or use a pre-trained model:
dotnet /workspace/dotnet/bin/TextClassifier.dll "404 Not Found" --gpu

The --threads N flag can specify the number of training threads.
Output
Both classifiers output:

Prediction (Error Page or Not Error Page).
Probability score.
Matched words from the vocabulary.

Example:
Text: 404 Not Found
Predicted: Error Page, Probability: 0.9876
Matched Words: 404 (1), not (1), found (1)

Training Data
The training_data.json contains labeled text samples:

Error Pages (Target: 1): HTTP errors (404, 500, 403, etc.), server issues, WAF blocks.
Non-Error Pages (Target: 0): Welcome pages, product catalogs, blogs, etc.

Example:
[
    {"Text": "404 Not Found Page Missing", "Target": 1},
    {"Text": "Welcome to Our Site", "Target": 0}
]

Notes

GPU Acceleration: Ensure your NVIDIA drivers are up-to-date for CUDA 12.4 compatibility. The 3090 Ti requires CUDA 11.1 or later.

File Updates: The volume mount (./AI_Page_Classifier:/workspace) ensures real-time synchronization between host and container files.

Alea.CUDA License: The C# implementation uses Alea.CUDA, which may require a license for commercial use. Verify at QuantAlea.

Git Configuration: Set your Git email to avoid push errors:
git config --global user.email "DotNetRussell@users.noreply.github.com"



Troubleshooting

Mount Issues:
Verify files in /workspace:
ls /workspace


Check mount path:
ls /mnt/c/Users/YourName/source/repos/ScreenSniper/AI_Page_Classifier


Add the path to Docker Desktop’s File Sharing (Settings > Resources > File Sharing).



GPU Errors:
Run nvidia-smi in the container to confirm GPU detection.
Reinstall NVIDIA Container Toolkit if needed.


C# Build Errors:
Check build logs:
cd /workspace/dotnet/TextClassifier
dotnet build -c Release




File Errors:
Ensure training_data.json and vocabulary.json are valid JSON.



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch.
Submit a pull request with clear descriptions.

License
This project is licensed under the MIT License. See the LICENSE file for details.
