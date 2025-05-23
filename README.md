![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  

# ScreenSniper

ScreenSniper is a Python tool designed to analyze webpage screenshots by extracting text via OCR (Optical Character Recognition) and categorizing content using detection patterns and/or an AI text classifier. It generates meta tags to provide insights into technologies, security issues, and page types, making it useful for web reconnaissance, security assessments, and bug bounty hunting.

The tool supports:
- **Template-based detection** using JSON patterns in the `detectionPatterns` directory.
- **AI-based classification** using a trained neural network to identify interesting pages (e.g., error pages, login pages).
- Multiple output formats: normal (plain text), JSON, and XML.

Recent updates include integration with an AI classifier (`text_classifier.py`) for enhanced page analysis, new flags for flexible output control, and Docker support for easy deployment.

## Features
- Extracts text from screenshots using Tesseract OCR.
- Categorizes pages using detection patterns (e.g., identifying IIS, login pages).
- AI classification to predict if a page is "interesting" (e.g., error pages, sensitive content).
- Supports Base64-encoded extracted text output.
- Configurable output formats: normal, JSON, XML.
- Verbose mode for debugging OCR and classification steps.
- Docker support for consistent and portable execution.

## Installation

### Prerequisites
- Python 3.6+
- Tesseract OCR (system package)
- Python dependencies: `opencv-python`, `pytesseract`, `Pillow`, `numpy`, `torch` (for AI)
- Optional: Docker for containerized execution

#### Manual Installation
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

#### Docker Installation
ScreenSniper includes a Dockerfile for running the tool in a containerized environment, ensuring consistent dependencies and easy setup. The Docker image is based on `python:3.9-slim-bullseye` and includes:
- Tesseract OCR
- Chromium and Chromium Driver
- All required Python dependencies from `requirements.txt`
- Project files (`ScreenSniper`, `detectionPatterns`, `AI_Page_Classifier`, `testImages`)

To build the Docker image:
```bash
docker build -t screensniper .
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
├── ScreenSniper          # Main script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # This file
├── testImages/           # Sample images for testing
```

## Usage

### Manual Usage
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

### Docker Usage
Run ScreenSniper in a Docker container by mounting the current directory to `/app` in the container. The `-v $(pwd):/app` flag mounts the host directory, allowing access to local screenshots and ensuring output is saved to the host.

#### Example Commands
Analyze a WordPress page screenshot:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --output-format=json testImages/random-wp-page.png --ai
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

Analyze an ASPX stacktrace screenshot:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --output-format=json testImages/aspx-stacktrace.png --ai
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

**Notes**:
- The `--rm` flag ensures the container is removed after execution.
- The `-it` flag enables interactive mode with a TTY.
- The `-v $(pwd):/app` flag mounts the current directory to `/app`, making local files (e.g., `testImages/`) accessible in the container.
- The `--ai` flag enables AI classification, and `--output-format=json` specifies JSON output.

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
- **System**: `tesseract-ocr`, `chromium`, `chromium-driver` (included in Docker image)
- **Python**: `opencv-python`, `pytesseract`, `Pillow`, `numpy`, `torch`
- **AI Classifier**: `text_classifier.py`, `model.pt`, `training_data.json`, `vocabulary.json`

Install Python dependencies (manual installation):
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
- **Docker Issues**: Ensure the `-v $(pwd):/app` flag is used correctly to mount the current directory. Verify Docker image is built with `docker build -t screensniper .`.

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
- Author: ☣️ Mr. The Plague ☣️
- Twitter: [@DotNetRussell](https://twitter.com/DotNetRussell)
- Twitter: [@Squid_Sec](https://twitter.com/Squid_Sec)
- Website: [https://www.SquidHacker.com](https://www.SquidHacker.com)

For issues or feature requests, open an issue on GitHub or contact via Twitter.


### Changes Made
1. **Added Docker Section**:
   - Created a "Docker Installation" subsection under "Installation" to describe the Dockerfile and how to build the image.
   - Added a "Docker Usage" subsection under "Usage" to explain running ScreenSniper in a container.
   - Included the provided Docker run commands with their outputs, formatted consistently with existing examples.
   - Explained the `-v $(pwd):/app` flag, linking it to your earlier question about mounting paths instead of copying.

2. **Updated Features**:
   - Added "Docker support for consistent and portable execution" to the Features list.

3. **Updated Directory Structure**:
   - Added `Dockerfile` and `testImages/` to the directory structure for completeness.

4. **Updated Dependencies**:
   - Added `chromium` and `chromium-driver` to the system dependencies list, as they are included in the Dockerfile.
   - Noted that these are included in the Docker image for clarity.

5. **Troubleshooting**:
   - Added a Docker-specific troubleshooting tip about verifying the `-v` flag and image build.

6. **Maintained Style**:
   - Kept the formatting, tone, and structure consistent with the original README.
   - Ensured example outputs and commands align with existing examples (e.g., JSON, XML, normal output sections).

7. **Dockerfile Details**:
   - Highlighted key components of the Dockerfile (base image, system dependencies, Python dependencies, file copying, and environment setup) without reproducing the entire file, keeping the README concise.

