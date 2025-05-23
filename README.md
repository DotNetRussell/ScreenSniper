![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  


## FAST TRACK

### DOWNLOAD
`https://github.com/DotNetRussell/ScreenSniper.git`

### BUILD
```
cd ScreenSniper
docker build -t screensniper .
```

### RUN

- How to give it a directory of images
`docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --directory testImages/ --detection-pattern --output-format=json --ai`

- How to give it a single image
`docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/aspx-stacktrace.png --detection-pattern --output-format=json --ai`

- How to report
`docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/aspx-stacktrace.png --detection-pattern --output-format=json --ai --report`

- How to retrain
`docker run --rm -it -v $(pwd):/app screensniper python3 AI_Page_Classifier/text_classifier.py --retrain --threads=20 --verbose "test text"`



# ScreenSniper

ScreenSniper is a Python tool designed to analyze webpage screenshots by extracting text via OCR (Optical Character Recognition) and categorizing content using detection patterns and/or an AI text classifier. It generates meta tags to provide insights into technologies, security issues, and page types, making it useful for web reconnaissance, security assessments, and bug bounty hunting.

- [ScreenSniper](#screensniper)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Manual Installation](#manual-installation)
    - [Docker Installation](#docker-installation)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage)
    - [Manual Usage](#manual-usage)
    - [Docker Usage](#docker-usage)
    - [Command-Line Flags](#command-line-flags)
    - [Example Outputs](#example-outputs)
      - [JSON Output with AI and Extracted Text](#json-output-with-ai-and-extracted-text)
      - [XML Output with Detection Patterns and Report](#xml-output-with-detection-patterns-and-report)
      - [Normal Output with AI, Detection Patterns, and Directory](#normal-output-with-ai-detection-patterns-and-directory)
  - [AI Integration](#ai-integration)
  - [Detection Patterns](#detection-patterns)
  - [Dependencies](#dependencies)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

The tool supports:
- **Template-based detection** using JSON patterns in the `detectionPatterns` directory.
- **AI-based classification** using a trained neural network to identify interesting pages (e.g., error pages, login pages).
- Multiple output formats: normal (plain text), JSON, and XML.

Recent updates include:
- Integration with an AI classifier (`text_classifier.py`) for enhanced page analysis.
- New flags for flexible output control (`--report`, `--directory`).
- Base64-encoded extracted text output to prevent formatting issues.
- Docker support for easy deployment.

## Features
- Extracts text from screenshots using Tesseract OCR.
- Categorizes pages using detection patterns (e.g., identifying IIS, login pages).
- AI classification to predict if a page is "interesting" (e.g., error pages, sensitive content).
- Supports Base64-encoded extracted text output, included only with `--include-extracted` or `--verbose`.
- Configurable output formats: normal, JSON, XML.
- Single report file generation for all processed images with `--report`.
- Directory processing to analyze multiple screenshots with `--directory`.
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
python3 ScreenSniper --image_path ./path/to/screenshot.png --output-format=json --ai --include-extracted
```

Process a directory of screenshots:
```bash
python3 ScreenSniper --directory ./testImages --output-format=json --ai --report
```

Generate a report for a single screenshot:
```bash
python3 ScreenSniper --image_path ./path/to/screenshot.png --report --output-format=xml --detection-pattern
```

### Docker Usage
Run ScreenSniper in a Docker container by mounting the current directory to `/app` in the container. The `-v $(pwd):/app` flag mounts the host directory, allowing access to local screenshots and ensuring output is saved to the host.

#### Example Commands
Analyze a WordPress page screenshot:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/random-wp-page.png --output-format=json --ai
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
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/aspx-stacktrace.png --output-format=json --ai
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

Analyze a directory of screenshots with a report:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --directory testImages --report --output-format=json --ai
```

**Notes**:
- The `--rm` flag ensures the container is removed after execution.
- The `-it` flag enables interactive mode with a TTY.
- The `-v $(pwd):/app` flag mounts the current directory to `/app`, making local files (e.g., `testImages/`) accessible in the container.
- The `--ai` flag enables AI classification, and `--output-format=json` specifies JSON output.

### Command-Line Flags
- `--image_path <path>`: Path to a single screenshot image file (e.g., `screenshot.png`).
- `--directory <path>`: Path to a directory containing multiple screenshot image files.
- `--verbose`: Enable detailed debugging output (e.g., OCR steps, classifier logs) and include Base64-encoded extracted text.
- `--output-format [normal|json|xml]`: Specify output format (default: normal).
- `--ai`: Enable AI text classification using `text_classifier.py` to predict if a page is interesting.
- `--include-extracted`: Include the OCR’d text as a Base64-encoded meta tag (`ExtractedTextBase64`).
- `--detection-pattern`: Include meta tags from template-based detection patterns in `detectionPatterns`.
- `--report`: Generate a single report file (`screensniper_report_YYYYMMDD_HHMMSS.<format>`) with analysis results for all images.

### Example Outputs

#### JSON Output with AI and Extracted Text
```bash
python3 ScreenSniper --image_path ./starbucks.com.png --output-format=json --ai --include-extracted
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

#### XML Output with Detection Patterns and Report
```bash
python3 ScreenSniper --image_path ./starbucks.com.png --output-format=xml --detection-pattern --report
```
**Console and Report File (`screensniper_report_YYYYMMDD_HHMMSS.xml`)**:
```xml
<?xml version="1.0" ?>
<report>
    <generated>2025-05-23 17:28:00</generated>
    <output_format>xml</output_format>
    <total_images>1</total_images>
    <results>
        <result>
            <image_path>./starbucks.com.png</image_path>
            <meta_tags>
                <meta_tag>Technology: IIS</meta_tag>
                <meta_tag>PageType: Login Page</meta_tag>
                <meta_tag>File Path: ./starbucks.com.png</meta_tag>
            </meta_tags>
        </result>
    </results>
</report>
```

#### Normal Output with AI, Detection Patterns, and Directory
```bash
python3 ScreenSniper --directory ./testImages --ai --detection-pattern --verbose
```
**Output for each image**:
```
Analysis Results:
File Path: testImages/random-wp-page.png
Extracted Text (Base64): V29yZFByZXNzIExvZ2lu
Meta Tags:
Technology: WordPress
PageType: Login Page
Interesting Page: False
ClassifierProbability: 0.6768
File Path: testImages/random-wp-page.png
--------------------------------------------------
Analysis Results:
File Path: testImages/aspx-stacktrace.png
Extracted Text (Base64): QVNQLk5FVCBFcnJvcg==
Meta Tags:
Technology: ASP.NET
PageType: Error Page
Interesting Page: True
ClassifierProbability: 0.9485
File Path: testImages/aspx-stacktrace.png
--------------------------------------------------
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

Recent updates include a newly trained model for improved accuracy, with training data included in the `AI_Page_Classifier` directory.

## Detection Patterns
The `--detection-pattern` flag enables template-based detection using JSON files in the `detectionPatterns` directory. Each file specifies conditions (e.g., keywords like "IIS", "login") and corresponding meta tags (e.g., `Technology: IIS`, `PageType: Login Page`). Example:
```json
{
    "conditions": ["IIS"],
    "meta_tags": ["Technology: IIS"]
}
```

Without `--detection-pattern`, only AI classifier results (if `--ai` is set) and file path are included, unless `--include-extracted` or `--verbose` is used.

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
- **Report File Issues**: Ensure write permissions in the current directory for report files (`screensniper_report_*.`).

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
