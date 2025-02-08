"""
Implementation of Gemini Flash functionality for PDF processing
"""

import fitz
from PIL import Image
import io
import google.generativeai as genai
from typing import List, Dict, Any
from tqdm import tqdm
import base64
import json  # Added to parse JSON responses
import fitz
from PIL import Image
import io
import google.generativeai as genai
from typing import List, Dict, Any
from tqdm import tqdm
import json  # For JSON parsing

def extract_markdown(pdf_path: str, prompt: str) -> str:
    """Extract markdown text with semantic chunking."""
    doc = fitz.open(pdf_path)
    full_text = []
    
    # Create the model once outside the loop
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    for page in tqdm(doc, desc="Processing pages"):
        # Convert page to image
        pix = page.get_pixmap()
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Process with Gemini
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_byte_arr}
        ])
        
        # Safely extract text from response regardless of its type
        if isinstance(response, dict):
            text = response.get("text", "")
        elif hasattr(response, "text"):
            text = response.text
        elif isinstance(response, str):
            text = response
        else:
            text = ""
        
        if text:
            full_text.append(text)
    
    doc.close()
    return "\n\n".join(full_text)

def extract_text_nodes(pdf_path: str) -> List[str]:
    """Extract text nodes from PDF."""
    doc = fitz.open(pdf_path)
    text_nodes = []
    
    for page in tqdm(doc, desc="Extracting text nodes"):
        text = page.get_text("text")
        # Split into meaningful chunks (paragraphs)
        nodes = [node.strip() for node in text.split('\n\n') if node.strip()]
        text_nodes.extend(nodes)
    
    doc.close()
    return text_nodes


def get_bounding_boxes(text_nodes: List[str], prompt: str, pdf_path: str) -> Dict[str, Dict]:
    """Get bounding boxes for text nodes."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    bounding_boxes = {}
    
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    try:
        # Get the first page as an image
        page = doc[0]
        pix = page.get_pixmap()
        img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        img_data.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        for node in tqdm(text_nodes, desc="Getting bounding boxes"):
            try:
                formatted_prompt = prompt.format(nodes=node)
                response = model.generate_content([
                    formatted_prompt,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                if not response or not response.text:
                    print(f"Warning: No response for node: {node[:50]}...")
                    bounding_boxes[node] = {}
                    continue
                
                # Parse response - look for numbers in the text
                response_text = response.text.strip()
                import re
                # Find all floating point numbers in the text
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
                
                if len(numbers) >= 4:  # If we found at least 4 numbers
                    try:
                        # Take the first 4 numbers found
                        coords = numbers[:4]
                        bounding_boxes[node] = {
                            'x1': float(coords[0]),
                            'y1': float(coords[1]),
                            'x2': float(coords[2]),
                            'y2': float(coords[3])
                        }
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not convert coordinates to float: {coords}")
                        bounding_boxes[node] = {}
                else:
                    print(f"Warning: Not enough coordinates found in response: {response_text}")
                    bounding_boxes[node] = {}
                    
            except Exception as e:
                print(f"Error processing node: {str(e)}")
                bounding_boxes[node] = {}
    
    finally:
        doc.close()
    
    return bounding_boxes
