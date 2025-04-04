import cv2
import numpy as np
import easyocr
import json
from google import genai
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class MedicalReportParser:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        # Configure Gemini API
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=gemini_api_key)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better OCR results.
        """
        # Create output directory if it doesn't exist
        output_dir = Path("preprocessing_steps")
        output_dir.mkdir(exist_ok=True)
        
        # Save original image
        cv2.imwrite(str(output_dir / "1_original.jpg"), image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(output_dir / "2_grayscale.jpg"), gray)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        cv2.imwrite(str(output_dir / "3_denoised.jpg"), denoised)
        
        print("\nPreprocessing steps have been saved in the 'preprocessing_steps' directory:")
        print("1. original.jpg - Original input image")
        print("2. grayscale.jpg - Grayscale conversion")
        print("3. denoised.jpg - After noise reduction")
        
        return denoised
    
    def extract_text(self, image_path: str) -> List[Tuple[str, List[List[int]]]]:
        """
        Extract text from the image using EasyOCR.
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Perform OCR
        results = self.reader.readtext(processed_image)
        return results
    
    def analyze_with_gemini(self, text: str) -> Dict:
        """
        Use Gemini API to analyze and structure the medical text.
        """
        prompt = f"""
        Analyze the following medical report text and extract key information into a structured format.
        Group the information into these fields:
        - Patient Name
        - Age
        - Gender
        - Date of Visit
        - Diagnosis
        - Symptoms
        - Prescription
        - Doctor's Name
        - Additional Notes

        Medical Report Text:
        {text}

        Return the information in a JSON format with these exact field names. If a field is not found, set it to null.
        Only include the JSON output, no additional text.
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            # Extract JSON from response
            json_str = response.text.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            return {
                "Patient Name": None,
                "Age": None,
                "Gender": None,
                "Date of Visit": None,
                "Diagnosis": None,
                "Symptoms": None,
                "Prescription": None,
                "Doctor's Name": None,
                "Additional Notes": None
            }
    
    def extract_and_analyze(self, image_path: str):
        """
        Extract text from image and analyze it using Gemini API.
        """
        # Extract text from image
        ocr_results = self.extract_text(image_path)
        
        # Combine all extracted text
        full_text = "\n".join([result[1] for result in ocr_results if len(result) >= 2])
        
        print("\nExtracted Text from Image:")
        print("-------------------------")
        print(full_text)
        print("\n" + "="*50 + "\n")
        
        # Analyze with Gemini
        print("Analyzing text with Gemini API...")
        structured_data = self.analyze_with_gemini(full_text)
        
        # Save results
        output_file = "analyzed_report.json"
        with open(output_file, 'w') as f:
            json.dump(structured_data, f, indent=4)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        print("\nStructured Medical Report:")
        print(json.dumps(structured_data, indent=4))

def main():
    # Example usage
    parser = MedicalReportParser()
    
    # Process the image
    image_path = "image.png"  # Using the available image
    try:
        parser.extract_and_analyze(image_path)
    except Exception as e:
        print(f"Error processing medical report: {str(e)}")

if __name__ == "__main__":
    main() 