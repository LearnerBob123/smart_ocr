# Medical Report Parser

A Python-based OCR pipeline for extracting and analyzing text from medical reports using EasyOCR, OpenCV, and Google's Gemini API.

## Features

- OCR text extraction from medical reports
- PDF file support
- Image preprocessing for better OCR results
- Intelligent text analysis using Gemini API
- Structured JSON output

## Prerequisites

- Python 3.8 or higher
- Git

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd medical_report_parser
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Place your medical report (image or PDF) in the project directory
2. Run the parser:
```bash
python medical_report_parser.py
```

3. The script will:
   - Extract text from the report
   - Save preprocessing steps in the 'preprocessing_steps' directory
   - Generate a structured JSON output in 'analyzed_report.json'

## Project Structure

```
medical_report_parser/
├── .gitignore           # Git ignore rules
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in Git)
├── medical_report_parser.py  # Main script
└── preprocessing_steps/     # Generated preprocessing images (not in Git)
```

## Git Setup

The project is configured to ignore:
- Virtual environment directory (`venv/`)
- Environment variables (`.env`)
- Generated files (`preprocessing_steps/`, `analyzed_report.json`)
- Python cache files (`__pycache__/`)
- IDE-specific files (`.vscode/`, `.idea/`)

## Output Format

The script generates a JSON file with the following structure:
```json
{
    "Patient Name": "string or null",
    "Age": "string or null",
    "Gender": "string or null",
    "Date of Visit": "string or null",
    "Diagnosis": "string or null",
    "Symptoms": "string or null",
    "Prescription": "string or null",
    "Doctor's Name": "string or null",
    "Additional Notes": "string or null"
}
```

## Error Handling

- The script will raise an error if the Gemini API key is not found
- Invalid image files will be reported with appropriate error messages
- Missing fields in the analysis will be set to `null` in the output

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Notes

- The parser works best with clear, well-lit images
- For handwritten text, ensure the handwriting is relatively clear and consistent
- The accuracy of text extraction depends on image quality and preprocessing
- You may need to adjust preprocessing parameters based on your specific use case#   s m a r t _ o c r  
 