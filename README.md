# Portfolio Tracker

A Python-based tool that automatically captures and analyzes your cryptocurrency portfolio data from Debank.

## Features

- Automated screenshot capture of your Debank wallet
- AI-powered portfolio analysis using Phi-3.5 Vision
- Extracts token balances and USD values
- Clean and structured output in JSON format

## Prerequisites

- Python 3.8+
- Azure OpenAI API access

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd portfolio-tracker
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a .env file with the following variables:

```bash
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_API_KEY=your_api_key
INFERENCE_API_ENDPOINT=your_inference_endpoint
INFERENCE_API_VERSION=your_inference_version
INFERENCE_API_KEY=your_inference_key
```

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Capture screenshots from Debank
2. Analyze the screenshots using AI
3. Output the portfolio composition in JSON format

## Project Structure
- main.py: Main script
- screens/: Directory for captured screenshots
- .env: Environment variables
- .gitignore: Git ignore rules
