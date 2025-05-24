# Use Go 1.24 image with Debian Bullseye
FROM golang:1.24-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies, Python 3.9, and others
RUN apt-get update && apt-get install -y \
	python3.9 \
	python3-pip \
	tesseract-ocr \
	chromium \
	chromium-driver \
	&& rm -rf /var/lib/apt/lists/*

# Install subfinder with Go modules enabled
ENV GO111MODULE=on
RUN go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest && \
	ln -s /go/bin/subfinder /usr/local/bin/subfinder

# Copy requirements file (to cache pip install)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps

# Copy repository files (adjust for potential filename variations)
COPY ScreenSniper* screensniper
COPY detectionPatterns/ detectionPatterns/
COPY AI_Page_Classifier/ AI_Page_Classifier/
COPY testImages/ testImages/

# Ensure scripts are executable and have correct shebang
RUN chmod +x screensniper && \
	sed -i 's|#!/usr/bin/python3|#!/usr/bin/env python3|' screensniper

# Set environment variables for Python and Tesseract
ENV PYTHONUNBUFFERED=1
ENV PATH="/app:/usr/local/bin:${PATH}"

# Default command (run screensniper with --help)
CMD ["screensniper", "--help"]