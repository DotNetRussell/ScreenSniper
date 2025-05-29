![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  


# FAST TRACK

### DOWNLOAD

`git clone https://github.com/DotNetRussell/ScreenSniper.git`

### BUILD

```
cd ScreenSniper
docker build -t screensniper .
```

### RUN

- How to give it a directory of images
  
```
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --directory testImages/ --detection-pattern --output-format=json --ai
```

- How to give it a single image
  
```
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/aspx-stacktrace.png --detection-pattern --output-format=json --ai
```


- How to give it a single url and have it screen shot for you
  
```
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --url github.com --detection-pattern --output-format=json --ai
```

- How to report
  
```
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/aspx-stacktrace.png --detection-pattern --output-format=json --ai --report
```

- How to retrain
  
```
docker run --rm -it -v $(pwd):/app screensniper python3 AI_Page_Classifier/text_classifier.py --retrain --threads=20 --verbose "test text"
```


# ScreenSniper

![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)

ScreenSniper is a Python tool designed to analyze webpage screenshots by extracting text via OCR (Optical Character Recognition) and categorizing content using detection patterns and/or an AI text classifier. It generates meta tags to provide insights into technologies, security issues, and page types, making it ideal for web reconnaissance, security assessments, and bug bounty hunting.

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
      - [JSON Output with AI and Extracted Text (Single Image)](#json-output-with-ai-and-extracted-text-single-image)
      - [XML Output with Detection Patterns and Report (Directory)](#xml-output-with-detection-patterns-and-report-directory)
      - [Normal Output with AI and Detection Patterns (URL)](#normal-output-with-ai-and-detection-patterns-url)
      - [Empty JSON Output for Failed URL Screenshot](#empty-json-output-for-failed-url-screenshot)
  - [AI Integration](#ai-integration)
  - [Detection Patterns](#detection-patterns)
    - [Detection Pattern Properties](#detection-pattern-properties)
    - [Example Detection Pattern](#example-detection-pattern)
  - [Dependencies](#dependencies)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

The tool supports:
- **Template-based detection** using JSON patterns in the `detectionPatterns` directory.
- **AI-based classification** using a trained neural network to identify interesting pages (e.g., error pages, login pages).
- Multiple output formats: normal (plain text), JSON, and XML.
- Analysis of single images, directories of images, or screenshots captured from URLs.

Recent updates include:
- **URL support**: Capture screenshots from a single URL using `ScreenShotter` with the `--url` flag, with automatic cleanup of generated screenshot files.
- **Enhanced detection patterns**: Support for complex AND/OR conditions in `conditions` to create precise matching rules.
- **Failure handling**: Returns an empty array in the specified format if `ScreenShotter` fails to capture a screenshot for a URL.
- **Improved AI classifier**: Newly trained model for better accuracy.
- **Docker support**: Consistent and portable execution environment.

## Features
- Extracts text from screenshots using Tesseract OCR.
- Categorizes pages using detection patterns (e.g., identifying IIS, login pages, error pages).
- AI classification to predict if a page is "interesting" (e.g., error pages, sensitive content).
- Supports Base64-encoded extracted text output, included with `--include-extracted` or `--verbose`.
- Configurable output formats: normal, JSON, XML (with results always in an array for JSON/XML).
- Single report file generation for all processed images with `--report`.
- Directory processing to analyze multiple screenshots with `--directory`.
- URL-based screenshot capture using `ScreenShotter` with `--url`, with automatic cleanup.
- Verbose mode for debugging OCR, classification, and screenshot capture steps.
- Advanced detection patterns with AND/OR logic for precise matching.
- Docker support for easy deployment.

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

For URL-based screenshot capture (`--url` flag), ensure `ScreenShotter` is in the project directory and executable.

#### Docker Installation
ScreenSniper includes a Dockerfile for running the tool in a containerized environment, ensuring consistent dependencies. The Docker image is based on `python:3.9-slim-bullseye` and includes:
- Tesseract OCR
- Chromium and Chromium Driver
- All required Python dependencies from `requirements.txt`
- Project files (`ScreenSniper`, `detectionPatterns`, `AI_Page_Classifier`, `ScreenShotter`, `testImages`)

To build the Docker image:
```bash
docker build -t screensniper .
```

## Directory Structure
```
ScreenSniper/
├── detectionPatterns/    # JSON files with detection patterns
├── AI_Page_Classifier/   # AI classifier script and data
│   ├── text_classifier.py
│   ├── model.pt
│   ├── training_data.json
│   ├── vocabulary.json
├── ScreenSniper          # Main script
├── ScreenShotter         # Script for capturing URL screenshots
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

Capture and analyze a screenshot from a URL:
```bash
python3 ScreenSniper --url https://example.com --output-format=json --detection-pattern --ai
```

Generate a report for a single screenshot:
```bash
python3 ScreenSniper --image_path ./path/to/screenshot.png --report --output-format=xml --detection-pattern
```

### Docker Usage
Run ScreenSniper in a Docker container by mounting the current directory to `/app` in the container using `-v $(pwd):/app`. This allows access to local screenshots and saves output to the host.

#### Example Commands
Analyze a WordPress page screenshot:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/random-wp-page.png --output-format=json --ai
```

Analyze a URL screenshot:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --url https://example.com --output-format=json --ai --detection-pattern
```

Analyze a directory of screenshots with a report:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --directory testImages --report --output-format=json --ai
```

**Notes**:
- The `--rm` flag removes the container after execution.
- The `-it` flag enables interactive mode with a TTY.
- The `-v $(pwd):/app` flag mounts the current directory to `/app`, making local files (e.g., `testImages/`) accessible.
- The `--url` flag captures a screenshot using `ScreenShotter`, which is deleted after analysis.
- If `ScreenShotter` fails, an empty array is returned in the specified format.

### Command-Line Flags
- `--image_path <path>`: Path to a single screenshot image file (e.g., `screenshot.png`).
- `--directory <path>`: Path to a directory containing multiple screenshot image files.
- `--url <url>`: URL to capture a screenshot from using `ScreenShotter` (e.g., `https://example.com`). The screenshot (`<url>.png`) is deleted after analysis.
- `--verbose`: Enable detailed debugging output (e.g., OCR steps, classifier logs, screenshot capture) and include Base64-encoded extracted text.
- `--output-format [normal|json|xml]`: Specify output format (default: normal). JSON/XML outputs are always arrays.
- `--ai`: Enable AI text classification using `text_classifier.py` to predict if a page is interesting.
- `--include-extracted`: Include the OCR’d text as a Base64-encoded meta tag (`ExtractedTextBase64`).
- `--detection-pattern`: Include meta tags from template-based detection patterns in `detectionPatterns`.
- `--report`: Generate a single report file (`screensniper_report_YYYYMMDD_HHMMSS.<format>`) with analysis results for all images.

### Example Outputs

#### JSON Output with AI and Extracted Text (Single Image)
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --image_path testImages/starbucks.com.png --output-format=json --ai --include-extracted
```
```json
[
    {
        "meta_tags": [
            "Interesting Page: True",
            "ClassifierProbability: 0.6623",
            "ExtractedTextBase64: SUlTIGxvZ2lu",
            "File Path: testImages/starbucks.com.png"
        ]
    }
]
```

#### XML Output with Detection Patterns and Report (Directory)
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --directory testImages --output-format=xml --detection-pattern --report
```
**Console and Report File (`screensniper_report_YYYYMMDD_HHMMSS.xml`)**:
```xml
<?xml version="1.0" ?>
<reports>
    <report>
        <generated>2025-05-29 19:33:00</generated>
        <output_format>xml</output_format>
        <total_images>2</total_images>
        <results>
            <result>
                <image_path>testImages/random-wp-page.png</image_path>
                <meta_tags>
                    <meta_tag>Technology: WordPress</meta_tag>
                    <meta_tag>PageType: Login Page</meta_tag>
                    <meta_tag>File Path: testImages/random-wp-page.png</meta_tag>
                </meta_tags>
            </result>
            <result>
                <image_path>testImages/aspx-stacktrace.png</image_path>
                <meta_tags>
                    <meta_tag>Technology: ASP.NET</meta_tag>
                    <meta_tag>PageType: Error Page</meta_tag>
                    <meta_tag>File Path: testImages/aspx-stacktrace.png</meta_tag>
                </meta_tags>
            </result>
        </results>
    </report>
</reports>
```

#### Normal Output with AI and Detection Patterns (URL)
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --url https://example.com --output-format=normal --ai --detection-pattern
```
```
Analysis Results:
File Path: https://example.com.png
Meta Tags:
PageType: Unknown
Interesting Page: False
ClassifierProbability: 0.5000
File Path: https://example.com.png
```

#### Empty JSON Output for Failed URL Screenshot
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 ScreenSniper --url https://invalid-url.com --output-format=json --verbose
```
```
Running ScreenShotter command: echo "https://invalid-url.com" | python3 /app/ScreenShotter
ScreenShotter return code: 1
Error: ScreenShotter failed with exit code 1: [error message]
[]
```

## AI Integration
The `--ai` flag enables the AI text classifier (`text_classifier.py`), which uses a trained neural network to predict if a page is "interesting" (e.g., error pages, login pages, sensitive content). The classifier requires:
- `model.pt`: Trained model file.
- `training_data.json`: Training data in JSON format.
- `vocabulary.json`: Vocabulary for text feature extraction.

The classifier outputs:
- `Interesting Page: True/False`: Whether the page is deemed interesting.
- `ClassifierProbability: <float>`: Confidence score (0.0 to 1.0).

To retrain the model:
```bash
docker run --rm -it -v $(pwd):/app screensniper python3 AI_Page_Classifier/text_classifier.py --retrain --threads=20 --verbose "test text"
```

## Detection Patterns
The `--detection-pattern` flag enables template-based detection using JSON files in the `detectionPatterns` directory. Each file defines conditions to match text extracted from screenshots and assigns meta tags for technologies, page types, or security risks. Patterns support complex logic to ensure precise matching, reducing false positives.

### Detection Pattern Properties
Each JSON file in `detectionPatterns` can include the following properties:

- **`conditions`** (required):
  - **Type**: Array of strings or object.
  - **Description**: Specifies keywords or conditions to match in the extracted text (case-insensitive).
  - **Simple Array**: Treated as OR (any keyword matches).
    ```json
    "conditions": ["login", "sign in"]
    ```
  - **Structured Object**: Supports AND/OR logic with nesting.
    - `type`: `"AND"` or `"OR"` (case-insensitive).
    - `values`: Array of strings (keywords) or nested condition objects.
    ```json
    "conditions": {
      "type": "AND",
      "values": [
        {
          "type": "OR",
          "values": ["login", "sign in"]
        },
        "password"
      ]
    }
    ```
    Matches if "password" AND ("login" OR "sign in") are present.

- **`negative_conditions`** (optional):
  - **Type**: Array of strings.
  - **Description**: Keywords that, if present, prevent the pattern from matching (implicit AND NOT).
  - **Example**:
    ```json
    "negative_conditions": ["logout", "sign out"]
    ```

- **`meta_tags`** (required):
  - **Type**: Array of strings.
  - **Description**: Tags added to the output if conditions are met and negative conditions are not.
  - **Example**:
    ```json
    "meta_tags": ["PageType: Login Page", "Security: Authentication Page"]
    ```

- **`additional_checks`** (optional):
  - **Type**: Object.
  - **Sub-property**: `sensitive_extensions` (array of strings).
  - **Description**: Checks for specified file extensions in the text, adding security tags if found:
    - `Security: Exposed File Type ({ext})`
    - `SecurityRisk: Potential Sensitive File Exposure`
  - **Example**:
    ```json
    "additional_checks": {
      "sensitive_extensions": [".bak", ".sql"]
    }
    ```

### Example Detection Pattern
Filename: `detectionPatterns/login_page.json`
```json
{
  "name": "Login Page",
  "conditions": {
    "type": "AND",
    "values": [
      {
        "type": "OR",
        "values": ["login", "sign in", "log in", "signin"]
      },
      {
        "type": "OR",
        "values": ["username", "email", "phone"]
      },
      "password"
    ]
  },
  "negative_conditions": ["logout", "sign out"],
  "meta_tags": [
    "PageType: Login Page"
  ]
}
```
- Matches login pages with a login term, user identifier, and password field, excluding logout pages.
- Outputs `PageType: Login Page` if conditions are met.

## Dependencies
- **System**: `tesseract-ocr`, `chromium`, `chromium-driver` (included in Docker image)
- **Python**: `opencv-python`, `pytesseract`, `Pillow`, `numpy`, `torch`
- **AI Classifier**: `text_classifier.py`, `model.pt`, `training_data.json`, `vocabulary.json`
- **URL Screenshots**: `ScreenShotter` script

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
- **ScreenShotter Failure**: If `--url` fails, check `ScreenShotter` exists at `/app/ScreenShotter` and is executable. An empty array is returned in the specified format. Use `--verbose` to debug.
- **Invalid Filename for URL**: URLs like `https://example.com` generate `https://example.com.png`, which may cause filesystem issues. Verify `ScreenShotter` output with `--verbose`.
- **Gibberish in Text**: Noisy OCR output may occur. Update `training_data.json` to improve classifier performance.
- **Docker Issues**: Ensure `-v $(pwd):/app` mounts the correct directory. Verify Docker image is built with `docker build -t screensniper .`.
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
