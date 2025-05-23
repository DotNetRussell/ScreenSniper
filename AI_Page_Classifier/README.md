![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  

# AI Page Classifier

The `AI_Page_Classifier` directory contains the AI-based text classification component for [ScreenSniper](https://github.com/DotNetRussell/ScreenSniper), a Python tool for analyzing webpage screenshots. This component uses a trained neural network to predict whether a webpage screenshot contains "interesting" content, such as error pages, login pages, or sensitive information, based on text extracted via OCR (Optical Character Recognition).

The classifier is integrated into ScreenSniper via the `--ai` flag, enhancing its ability to categorize webpages for web reconnaissance, security assessments, and bug bounty hunting.

## Table of Contents
- [AI Page Classifier](#ai-page-classifier)
  - [Features](#features)
  - [Files](#files)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Integration with ScreenSniper](#integration-with-screensniper)
    - [Standalone Usage](#standalone-usage)
    - [Retraining the Model](#retraining-the-model)
  - [Dependencies](#dependencies)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Features
- Predicts if a webpage is "interesting" (e.g., error pages, login pages, sensitive content) using a trained neural network.
- Outputs a binary classification (`Interesting Page: True/False`) and a confidence score (`ClassifierProbability`).
- Processes text extracted from screenshots via Tesseract OCR.
- Supports model retraining with custom training data.
- Lightweight and optimized for integration with ScreenSniper.

## Files
- `text_classifier.py`: Main script for text classification and model training.
- `model.pt`: Trained PyTorch model for classification.
- `training_data.json`: JSON file containing training data for the classifier.
- `vocabulary.json`: JSON file defining the vocabulary for text feature extraction.

## Installation
The AI Page Classifier is a component of ScreenSniper and is included in its installation process. To set up the classifier, ensure the `AI_Page_Classifier` directory is present in the ScreenSniper root directory with all required files.

### Prerequisites
- Python 3.6+
- Python dependencies: `torch`, `numpy`
- ScreenSniper installed (see [ScreenSniper README](https://github.com/DotNetRussell/ScreenSniper/blob/main/README.md))

### Steps
1. Clone the ScreenSniper repository:
   ```bash
   git clone https://github.com/DotNetRussell/ScreenSniper.git
   cd ScreenSniper
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This includes `torch` and `numpy`, required for the AI classifier.

3. Verify the `AI_Page_Classifier` directory contains:
   - `text_classifier.py`
   - `model.pt`
   - `training_data.json`
   - `vocabulary.json`

4. If `model.pt` is missing, train the model:
   ```bash
   python3 AI_Page_Classifier/text_classifier.py --retrain "dummy text"
   ```

### Docker
If using ScreenSniper's Docker image, the `AI_Page_Classifier` directory and its dependencies are included automatically. Build the image:
```bash
docker build -t screensniper .
```

## Usage

### Integration with ScreenSniper
The AI classifier is activated in ScreenSniper with the `--ai` flag, processing OCR-extracted text from screenshots to predict if a page is interesting.

Example:
```bash
python3 ScreenSniper --output-format=json --ai testImages/aspx-stacktrace.png
```
**Output**:
```json
{
    "meta_tags": [
        "Interesting Page: True",
        "ClassifierProbability: 0.9485",
        "File Path: testImages/aspx-stacktrace.png"
    ]
}
```

Docker example:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --output-format=json --ai testImages/random-wp-page.png
```
**Output**:
```json
{
    "meta_tags": [
        "Interesting Page: False",
        "ClassifierProbability: 0.6768",
        "File Path: testImages/random-wp-page.png"
    ]
}
```

### Standalone Usage
The `text_classifier.py` script can be run independently to classify text or train the model.

Classify text:
```bash
python3 AI_Page_Classifier/text_classifier.py --classify "Sample error page text"
```
**Output**:
```plaintext
Interesting Page: True
ClassifierProbability: 0.85
```

### Retraining the Model
To improve accuracy or adapt to new data, update `training_data.json` with new text samples and labels, then retrain:
```bash
python3 AI_Page_Classifier/text_classifier.py --retrain "dummy text"
```
- `training_data.json` format:
  ```json
  [
      {"text": "Error 500 Internal Server Error", "label": 1},
      {"text": "Welcome to our homepage", "label": 0}
  ]
  ```
- `label`: 1 (interesting) or 0 (not interesting).

The retrained model is saved as `model.pt`, and the vocabulary is updated in `vocabulary.json`.

## Dependencies
- **Python**: `torch`, `numpy`
- **Files**: `model.pt`, `training_data.json`, `vocabulary.json`

Install dependencies:
```bash
pip install torch numpy
```

## Troubleshooting
- **Missing `model.pt`**: Run `python3 AI_Page_Classifier/text_classifier.py --retrain "dummy text"` to generate the model.
- **Vocabulary Error**: Ensure `vocabulary.json` exists and matches the format expected by `text_classifier.py`.
- **Low Accuracy**: Add more diverse samples to `training_data.json` and retrain the model.
- **PyTorch Issues**: Verify `torch` is installed correctly (`pip show torch`) and compatible with your system.

## Contributing
Contributions to the AI classifier are welcome! To contribute:
1. Fork the ScreenSniper repository.
2. Create a feature branch (`git checkout -b feature/ai-improvement`).
3. Commit changes (`git commit -m 'Improve AI classifier accuracy'`).
4. Push to the branch (`git push origin feature/ai-improvement`).
5. Open a pull request.

Please include tests or updated `training_data.json` for new features.

## License
MIT License. See `LICENSE` in the ScreenSniper root directory for details.


## Contact
- Author: ☣️ Mr. The Plague ☣️
- Twitter: [@DotNetRussell](https://twitter.com/DotNetRussell)
- Twitter: [@Squid_Sec](https://twitter.com/Squid_Sec)
- Website: [https://www.SquidHacker.com](https://www.SquidHacker.com)

For issues or feature requests, open an issue on [GitHub](https://github.com/DotNetRussell/ScreenSniper) or contact via Twitter.