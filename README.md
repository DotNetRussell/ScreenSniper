
![ScreenSniper Logo](https://i.imgur.com/yfJZLWm.png)  

# ScreenSniper

**A Penetration Testing Tool for Webpage Screenshot Analysis**

ScreenSniper is a Python-based tool designed for penetration testers to analyze webpage screenshots and identify security risks. It extracts text from screenshots using Tesseract OCR and matches the text against customizable templates to generate meta tags, such as `PageType: Login Page` or `SecurityRisk: Sensitive Information Exposure`. This tool is ideal for authorized testing scenarios where you need to quickly identify vulnerabilities like directory listings, default server pages, or exposed stack traces.

## Features

- **OCR-Powered Text Extraction**: Uses Tesseract OCR to extract text from webpage screenshots.
- **Template-Based Detection**: Matches extracted text against JSON templates in the `detectionPatterns/` directory to generate security-relevant meta tags.
- **Flexible Output Formats**: Supports `normal` (plain text), `json`, and `xml` output formats for easy integration into workflows.
- **Verbose Debugging**: Optional `--verbose` flag to display preprocessing steps and extracted text for troubleshooting.
- **Customizable**: Easily add or modify detection templates to suit your testing needs.
- **Lightweight and Free**: Built with open-source libraries, requiring no additional costs.

## Installation

### Prerequisites
- **Python 3.8+**
- **Tesseract OCR** installed on your system:
  - **Ubuntu**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki), then add to your PATH.

### Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/screensniper.git
   cd screensniper
   ```

2. **Install Dependencies**:
   Create a virtual environment (optional but recommended) and install the required Python packages:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify Tesseract Installation**:
   Ensure Tesseract is installed and accessible:
   ```
   tesseract --version
   ```

## Usage

### Basic Command
Analyze a webpage screenshot and output meta tags in the default `normal` format:
```
python screenSniper.py path/to/screenshot.png
```

### Example Output
For a screenshot `testImages/login-cms.png` containing a login page and CMS signature:
```
PageType: Login Page
Technology: amFOSS CMS
SecurityRisk: Check for Known CMS Vulnerabilities
File Path: testImages/login-cms.png
```

### Options
- **Verbose Output**: Display preprocessing steps and extracted text for debugging:
  ```
  python screenSniper.py path/to/screenshot.png --verbose
  ```
  Example verbose output:
  ```
  Starting image preprocessing...
  Image loaded successfully.
  Image resized and saved as debug_resized.png
  Converted to grayscale and saved as debug_grayscale.png
  Applied CLAHE and saved as debug_clahe.png
  Applied adaptive thresholding and saved as debug_threshold.png
  Applied dilation and saved as debug_dilated.png
  Final preprocessed image saved as preprocessed.png
  Extracted Text: Username: Password: Forgotten your password or username? Log In This is amFOSS CMS < By the CLUB | Of the CLUB | For the CLUB >
  Analysis Results:
  File Path: testImages/login-cms.png
  Extracted Text: Username: Password: Forgotten your password or username? Log In This is amFOSS CMS < By the CLUB | Of the CLUB | For the CLUB >
  Meta Tags:
  PageType: Login Page
  Technology: amFOSS CMS
  SecurityRisk: Check for Known CMS Vulnerabilities
  File Path: testImages/login-cms.png
  ```

- **Output Formats**: Choose between `normal`, `json`, or `xml` formats:
  - JSON:
    ```
    python screenSniper.py path/to/screenshot.png --output-format json
    ```
    Output:
    ```
    {
        "meta_tags": [
            "PageType: Login Page",
            "Technology: amFOSS CMS",
            "SecurityRisk: Check for Known CMS Vulnerabilities",
            "File Path: testImages/login-cms.png"
        ]
    }
    ```
  - XML:
    ```
    python screenSniper.py path/to/screenshot.png --output-format xml
    ```
    Output:
    ```
    <?xml version="1.0" ?>
    <result>
        <meta_tags>
            <meta_tag>PageType: Login Page</meta_tag>
            <meta_tag>Technology: amFOSS CMS</meta_tag>
            <meta_tag>SecurityRisk: Check for Known CMS Vulnerabilities</meta_tag>
            <meta_tag>File Path: testImages/login-cms.png</meta_tag>
        </meta_tags>
    </result>
    ```

## Directory Structure
- `screenSniper.py`: The main script for analyzing screenshots.
- `detectionPatterns/`: Directory containing JSON templates for detection patterns.
  - `login_page.json`: Template for detecting login pages.
  - `amfoss_cms.json`: Template for detecting amFOSS CMS.
  - Add more templates as needed (see [Customizing Templates](#customizing-templates)).
- `testImages/`: Directory for test screenshots (not included in the repo; create your own).
- `requirements.txt`: List of Python dependencies.
- `README.md`: This documentation file.

## Customizing Templates
ScreenSniper uses JSON templates in the `detectionPatterns/` directory to match extracted text and generate meta tags. Each template has the following structure:
- `name`: Descriptive name of the detection rule.
- `conditions`: List of keywords to match (case-insensitive).
- `negative_conditions` (optional): Keywords that, if present, prevent the template from matching.
- `meta_tags`: List of meta tags to apply if the conditions are met.
- `additional_checks` (optional): Extra logic, such as checking for sensitive file extensions.

### Example Template: `detectionPatterns/login_page.json`
```
{
  "name": "Login Page",
  "conditions": ["login", "sign in", "username", "password", "user", "pass", "forgot"],
  "negative_conditions": ["logout", "sign out"],
  "meta_tags": [
    "PageType: Login Page"
  ]
}
```

### Adding a New Template
To detect a new pattern (e.g., password reset pages):
1. Create a new file `detectionPatterns/password_reset.json`:
   ```
   {
     "name": "Password Reset Page",
     "conditions": ["forgot", "reset", "password"],
     "meta_tags": [
       "PageType: Password Reset Page",
       "SecurityRisk: Check for Insecure Reset Mechanism"
     ]
   }
   ```
2. Run ScreenSniper on a screenshot containing a password reset page to test the new template.

## Troubleshooting
- **OCR Errors**:
  - If text extraction fails (`OCR Error: No text extracted`), use `--verbose` to inspect preprocessing steps.
  - Check the debug images (e.g., `.debug_resized.png`, `.debug_threshold.png`) to identify issues.
  - Adjust preprocessing parameters in `preprocess_image` (e.g., `scale_factor`, `clipLimit`) if needed.
  - Ensure the screenshot is high-resolution and text is legible.
- **Tesseract Not Found**:
  - Verify Tesseract is installed and in your PATH:
    ```
    tesseract --version
    ```
  - On Windows, you may need to set the Tesseract path explicitly in the script:
    ```
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```
- **Template Not Matching**:
  - Check the extracted text (`--verbose`) to ensure expected keywords are present.
  - Add more keyword variations to the template’s `conditions` (e.g., “usrname” for “username”).
- **File Not Found**:
  - Ensure the screenshot file exists and has a valid extension (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

### Ideas for Contributions
- Add new detection templates for specific vulnerabilities or technologies.
- Enhance OCR accuracy with alternative libraries (e.g., PaddleOCR, EasyOCR).
- Implement batch processing for multiple screenshots.
- Integrate automated testing scripts for detected vulnerabilities (e.g., brute-force login pages).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


### Notes
- **Correct Code Block**: The README is now presented in a raw Markdown code block with triple backticks (```), ensuring proper formatting and avoiding escaping issues. You can copy this content directly into a `README.md` file.
- **Tool Name**: All references to the tool are updated to "ScreenSniper," including the script name (`screenSniper.py`), commands, and mentions throughout the document.
- **Formatting**: The content remains well-structured for GitHub, with clear sections, code blocks for examples, and proper Markdown syntax.
- **Copying**: To use this, copy the content within the code block (excluding the outer triple backticks) into a `README.md` file in your repository’s root directory.

If you need further adjustments or have additional files to generate, let me know!


------------------------------------------------------------------------

# Website Screenshot Tool

A Python script that captures screenshots of websites using Playwright. The script reads a list of URLs from standard input, navigates to each URL, and saves a screenshot of the webpage as a PNG file named after the website's domain.

## Features

- **Concurrent Processing**: Captures screenshots for multiple URLs concurrently, with a semaphore to limit resource usage.
- **Headless Browser**: Uses Playwright's Chromium browser in headless mode for reliable rendering.
- **Error Handling**: Gracefully handles timeouts and other errors, ensuring robust operation.
- **Customizable**: Configurable viewport size and user agent for consistent screenshots.

## Prerequisites

- Python 3.8 or higher
- [Playwright](https://playwright.dev/python/) for browser automation
- A list of URLs to process (provided via standard input)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs the required Python packages and Playwright browsers.

4. **Install Playwright Browsers**:
   ```bash
   playwright install
   ```

## Usage

1. **Prepare a List of URLs**:
   Create a file (e.g., `urls.txt`) with one URL per line. For example:
   ```
   google.com
   github.com
   example.com
   ```

2. **Run the Script**:
   Pipe the URLs into the script via standard input:
   ```bash
   cat urls.txt | python3 screenshot.py
   ```
   Alternatively, you can manually input URLs by running:
   ```bash
   python3 screenshot.py
   ```
   Then type each URL followed by Enter, and press Ctrl+D (or Ctrl+Z on Windows) to finish.

3. **Output**:
   - Screenshots are saved as PNG files in the current directory, named after the domain (e.g., `google.com.png`).
   - Console output indicates success or errors for each URL.

## Example

```bash
echo -e "google.com\ngithub.com" | python3 screenshot.py
```

**Output**:
```
Saved screenshot for https://google.com as google.com.png
Saved screenshot for https://github.com as github.com.png
```

## Requirements

See the `requirements.txt` file for dependencies:
- `playwright>=1.47.0`

## Notes

- **Concurrency**: The script limits concurrent browser instances to 5 to prevent resource exhaustion. Adjust the `semaphore` value in the script if needed.
- **Timeouts**: URLs that take longer than 10 seconds to load are skipped with a timeout message.
- **URL Formatting**: URLs without a scheme (e.g., `google.com`) are automatically prefixed with `https://`.
- **Screenshot Size**: Screenshots are taken at a 1280x720 viewport. Modify the `viewport` parameter in the script to change this.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
