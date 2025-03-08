import cv2
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
from .config import TESSERACT_PATH, ENABLE_OCR

class FrameAnalyzer:
    def __init__(self):
        # Initialize OCR and NLP tools
        self.ocr_config = r'--oem 3 --psm 6'
        
        # Configure Tesseract path if available
        if ENABLE_OCR:
            if os.path.exists(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
                self.ocr_enabled = True
            else:
                print(f"Warning: Tesseract not found at {TESSERACT_PATH}. OCR will be disabled.")
                self.ocr_enabled = False
        else:
            self.ocr_enabled = False
            
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def analyze_frame(self, frame):
        # Main function to analyze a video frame
        processed_data = self.extract_data(frame)
        preprocessed_content = self.preprocess_content(processed_data)
        return preprocessed_content

    def extract_data(self, frame):
        # Extract various types of data from the frame
        data = {}
        
        # Extract text using OCR
        data['text'] = self.extract_text(frame)
        
        # Extract objects in the frame
        data['objects'] = self.detect_objects(frame)
        
        # Extract any other relevant information
        data['frame_info'] = self.extract_frame_info(frame)
        
        return data

    def extract_text(self, frame):
        """Extract text from the frame using OCR"""
        if not self.ocr_enabled:
            return ""
            
        try:
            # Convert frame to grayscale for better OCR results
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get better text contrast
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Convert the OpenCV image to PIL format for pytesseract
            pil_img = Image.fromarray(thresh)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(pil_img, config=self.ocr_config)
            
            return text.strip()
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""

    def detect_objects(self, frame):
        """Detect objects in the frame
        Note: For production, use pre-trained models like YOLO or SSD
        """
        # Placeholder for object detection
        objects = []
        
        # Simple object detection using contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'unknown',
                    'position': (x, y, w, h)
                })
        
        return objects

    def extract_frame_info(self, frame):
        """Extract general information about the frame"""
        height, width, channels = frame.shape
        brightness = np.mean(frame)
        
        # Detect if the frame likely contains a slide/presentation
        is_slide = brightness > 200
        
        return {
            'dimensions': (width, height),
            'brightness': brightness,
            'is_slide': is_slide,
        }

    def preprocess_content(self, data):
        """Preprocess the extracted data to make it suitable for Gemini"""
        processed_data = {}
        
        # Process extracted text
        if 'text' in data and data['text']:
            processed_data['text'] = self.preprocess_text(data['text'])
        
        # Process detected objects
        if 'objects' in data and data['objects']:
            processed_data['objects'] = data['objects']
            processed_data['object_summary'] = f"Detected {len(data['objects'])} objects in frame"
        
        # Include frame information
        if 'frame_info' in data:
            processed_data['frame_info'] = data['frame_info']
            
        return processed_data
    
    def preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Reconstruct text
        processed_text = ' '.join(filtered_tokens)
        
        return processed_text