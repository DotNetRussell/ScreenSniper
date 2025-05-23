
![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  

# ScreenSniper

ScreenSniper is a Python tool designed to analyze webpage screenshots by extracting text via OCR (Optical Character Recognition) and categorizing content using detection patterns and/or an AI text classifier. It generates meta tags to provide insights into technologies, security issues, and page types, making it useful for web reconnaissance, security assessments, and bug bounty hunting.

The tool supports:
- **Template-based detection** using JSON patterns in the `detectionPatterns` directory.
- **AI-based classification** using a trained neural network to identify interesting pages (e.g., error pages, login pages).
- Multiple output formats: normal (plain text), JSON, and XML.

Recent updates include integration with an AI classifier (`text_classifier.py`) for enhanced page analysis and new flags for flexible output control.

## Features
- Extracts text from screenshots using Tesseract OCR.
- Categorizes pages using detection patterns (e.g., identifying IIS, login pages).
- AI classification to predict if a page is "interesting" (e.g., error pages, sensitive content).
- Supports Base64-encoded extracted text output.
- Configurable output formats: normal, JSON, XML.
- Verbose mode for debugging OCR and classification steps.

## Installation

### Prerequisites
- Python 3.6+
- Tesseract OCR (system package)
- Python dependencies: `opencv-python`, `pytesseract`, `Pillow`, `numpy`, `torch` (for AI)

Install system dependencies (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Clone the repository:
```bash
git clone https://github.com/DotNetRussell/ScreenSniper.git
cd ScreenSniper
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

For AI classification (`--ai` flag), ensure the following files are in the `AI_Page_Classifier` directory:
- `text_classifier.py`
- `model.pt` (trained model)
- `training_data.json` (training data)
- `vocabulary.json` (vocabulary for text features)

If `model.pt` is missing, train the model:
```bash
python3 AI_Page_Classifier/text_classifier.py --retrain "dummy text"
```

### Directory Structure
```
ScreenSniper/
├── detectionPatterns/    # JSON files with detection patterns
├── AI_Page_Classifier/   # AI classifier script and data
│   ├── text_classifier.py
│   ├── model.pt
│   ├── training_data.json
│   ├── vocabulary.json
├── ScreenSniper         # Main script
├── requirements.txt     # Python dependencies
├── README.md            # This file
```

## Usage

Run `ScreenSniper` on a single screenshot:
```bash
python3 ScreenSniper --output-format=json --ai --include-extracted ./path/to/screenshot.png
```

Process multiple screenshots in a directory:
```bash
for image in $(ls testImages/); do
    python3 ScreenSniper --include-extracted --output-format=json --ai testImages/$image
done
```

### Command-Line Flags
- `--verbose`: Enable detailed debugging output (e.g., OCR steps, classifier logs).
- `--output-format [normal|json|xml]`: Specify output format (default: normal).
- `--ai`: Enable AI text classification using `text_classifier.py` to predict if a page is interesting.
- `--include-extracted`: Include the OCR’d text as a Base64-encoded meta tag (`ExtractedTextBase64`).
- `--detection-pattern`: Include meta tags from template-based detection patterns in `detectionPatterns`.

### Example Outputs

#### JSON Output with AI and Extracted Text
```bash
python3 ScreenSniper --output-format=json --ai --include-extracted ./starbucks.com.png
```
```json
{
    "meta_tags": [
        "Interesting Page: True",
        "ClassifierProbability: 0.6623",
        "ExtractedTextBase64: SUlTIGxvZ2lu",
        "File Path: ./starbucks.com.png"
    ]
}
```

#### XML Output with Detection Patterns
```bash
python3 ScreenSniper --output-format=xml --detection-pattern ./starbucks.com.png
```
```xml
<?xml version="1.0" ?>
<result>
    <meta_tags>
        <meta_tag>Technology: IIS</meta_tag>
        <meta_tag>PageType: Login Page</meta_tag>
        <meta_tag>File Path: ./starbucks.com.png</meta_tag>
    </meta_tags>
</result>
```

#### Normal Output with AI and Detection Patterns
```bash
python3 ScreenSniper --ai --detection-pattern ./starbucks.com.png
```
```
Technology: IIS
PageType: Login Page
Interesting Page: True
ClassifierProbability: 0.6623
File Path: ./starbucks.com.png
```

## AI Integration
The `--ai` flag enables the AI text classifier (`text_classifier.py`), which uses a trained neural network to predict if a page is "interesting" (e.g., error pages, login pages, or sensitive content). The classifier requires:
- `model.pt`: Trained model file.
- `training_data.json`: Training data in JSON format.
- `vocabulary.json`: Vocabulary for text feature extraction.

The classifier outputs:
- `Interesting Page: True/False`: Whether the page is deemed interesting.
- `ClassifierProbability: <float>`: Confidence score (0.0 to 1.0).

To update the training data or retrain the model, modify `training_data.json` and run:
```bash
python3 AI_Page_Classifier/text_classifier.py --retrain "dummy text"
```

Recent updates include a newly trained model for improved accuracy, with training data included in the `AI_Page_Classifier` directory.[](https://x.com/DotNetRussell/status/1925196652383469796)

## Detection Patterns
The `--detection-pattern` flag enables template-based detection using JSON files in the `detectionPatterns` directory. Each file specifies conditions (e.g., keywords like "IIS", "login") and corresponding meta tags (e.g., `Technology: IIS`, `PageType: Login Page`). Example:
```json
{
    "conditions": ["IIS"],
    "meta_tags": ["Technology: IIS"]
}
```

Without `--detection-pattern`, only AI classifier results (if `--ai` is set) and file path are included, unless `--include-extracted` is used.

## Dependencies
- **System**: `tesseract-ocr`
- **Python**: `opencv-python`, `pytesseract`, `Pillow`, `numpy`, `torch`
- **AI Classifier**: `text_classifier.py`, `model.pt`, `training_data.json`, `vocabulary.json`

Install Python dependencies:
```bash
pip install opencv-python pytesseract Pillow numpy torch
```

## Troubleshooting
- **Tesseract Error**: Ensure `tesseract-ocr` is installed and in PATH:
  ```bash
  tesseract --version
  ```
- **AI Classifier Error**: Verify `text_classifier.py` and required files are in `AI_Page_Classifier`. Check `model.pt` exists or retrain.
- **OCR Failure**: Use `--verbose` to inspect preprocessing steps (saved as `.debug_*.png`).
- **Gibberish in Text**: Some screenshots may produce noisy OCR output. Update `training_data.json` to improve classifier performance.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License
MIT License. See `LICENSE` for details.

## Contact
- Author: Anthony Russell
- Twitter: [@DotNetRussell](https://twitter.com/DotNetRussell)
- Blog: [https://www.DotNetRussell.com](https://www.DotNetRussell.com)
- Website: [https://www.SquidHacker.com](https://www.SquidHacker.com)

For issues or feature requests, open an issue on GitHub or contact via Twitter.