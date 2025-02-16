"""
pdf_parsers.py

PDF Parser Factory Module

This module provides a factory pattern implementation for different PDF parsers:
1. PyMuPDF4LLM (Fast, lightweight, best for simple PDFs)
2. Gemini Flash (Best for complex layouts, requires Google API)
3. Llama Parse (Strong structure preservation, requires Llama API)

Each parser implements a common interface for consistency and easy switching.
"""

import os
import io
import re
import json
import base64
import datetime  # <-- New import for date/time handling
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

import fitz  # PyMuPDF
import google.generativeai as genai
from llama_parse import LlamaParse
from openai import OpenAI

from pydantic import BaseModel, Field

# ---------------------------
# Global Debug Flag and Helper Function
# ---------------------------
DEBUG = True  # Set to False to disable debug output

def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

# ---------------------------
# Pydantic Models for Schema
# ---------------------------
class BoundingBox(BaseModel):
    """Model for bounding box coordinates."""
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

class Metadata(BaseModel):
    title: str
    pages: int
    parser: str
    text_nodes: List[str]
    bounding_boxes: Dict[str, BoundingBox] = {}

class ParsedPDF(BaseModel):
    metadata: Metadata
    content: str

# ---------------------------
# Custom Exception
# ---------------------------
class PDFParsingError(Exception):
    """Custom exception for PDF parsing errors."""
    pass

# ---------------------------
# Gemini Flash Helper Functions
# ---------------------------
def extract_markdown(pdf_path: str, prompt: str) -> str:
    """
    Extract markdown text with semantic chunking from a PDF.
    
    Converts each PDF page into an image and sends it along with the prompt to Gemini Flash.
    """
    debug(f"Starting extract_markdown for: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = []
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    for page in tqdm(doc, desc="Processing pages with Gemini Flash (Markdown extraction)"):
        try:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_bytes}
            ])
            
            debug(f"extract_markdown: response type {type(response)}")
            if isinstance(response, dict):
                text = response.get("text", "")
                debug(f"extract_markdown: received dict with text: {text[:100]}")
            elif hasattr(response, "text"):
                text = response.text
                debug(f"extract_markdown: response.text: {text[:100]}")
            elif isinstance(response, str):
                text = response
                debug(f"extract_markdown: response as string: {text[:100]}")
            else:
                text = ""
                debug("extract_markdown: Unknown response type")
            
            if text:
                full_text.append(text)
            else:
                debug("extract_markdown: No text found in response.")
        except Exception as e:
            debug(f"Error processing page {page.number}: {e}")
    
    doc.close()
    combined_text = "\n\n".join(full_text)
    debug(f"Completed extract_markdown, length of combined text: {len(combined_text)}")
    return combined_text

def extract_text_nodes(pdf_path: str) -> List[str]:
    """
    Extract text nodes from the PDF by splitting each page's text into paragraphs.
    """
    debug(f"Starting extract_text_nodes for: {pdf_path}")
    doc = fitz.open(pdf_path)
    text_nodes = []
    for page in tqdm(doc, desc="Extracting text nodes with Gemini Flash"):
        text = page.get_text("text")
        nodes = [node.strip() for node in text.split('\n\n') if node.strip()]
        debug(f"Page {page.number}: extracted {len(nodes)} nodes.")
        text_nodes.extend(nodes)
    doc.close()
    debug(f"Completed extract_text_nodes, total nodes: {len(text_nodes)}")
    return text_nodes

def get_bounding_boxes(text_nodes: List[str], prompt: str, pdf_path: str) -> Dict[str, Dict]:
    """
    Get bounding boxes for each text node using Gemini Flash.
    
    For each text node, the function sends a prompt and the first page image to Gemini Flash.
    Expects a response containing four numbers representing the bounding box.
    """
    debug("Starting get_bounding_boxes")
    model = genai.GenerativeModel('gemini-1.5-flash')
    bounding_boxes = {}
    doc = fitz.open(pdf_path)
    
    try:
        page = doc[0]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        for node in tqdm(text_nodes, desc="Extracting bounding boxes with Gemini Flash"):
            try:
                formatted_prompt = prompt.format(nodes=node)
                response = model.generate_content([
                    formatted_prompt,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                debug(f"get_bounding_boxes: Received response for node '{node[:50]}...' with type {type(response)}")
                if isinstance(response, dict):
                    response_text = response.get("text", "").strip()
                elif hasattr(response, "text"):
                    response_text = response.text.strip()
                elif isinstance(response, str):
                    response_text = response.strip()
                else:
                    response_text = ""
                    debug("get_bounding_boxes: Unknown response type")
                
                debug(f"get_bounding_boxes: Response text: {response_text}")
                
                if not response_text:
                    debug(f"Warning: No response text for node: {node[:50]}...")
                    bounding_boxes[node] = {}
                    continue
                
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
                debug(f"get_bounding_boxes: Found numbers: {numbers}")
                if len(numbers) >= 4:
                    coords = numbers[:4]
                    try:
                        bounding_boxes[node] = {
                            'x1': float(coords[0]),
                            'y1': float(coords[1]),
                            'x2': float(coords[2]),
                            'y2': float(coords[3])
                        }
                    except (ValueError, IndexError) as conv_err:
                        debug(f"Warning: Invalid coordinate conversion for: {coords} ({conv_err})")
                        bounding_boxes[node] = {}
                else:
                    debug(f"Warning: Insufficient coordinates found in response: {response_text}")
                    bounding_boxes[node] = {}
                    
            except Exception as inner_e:
                debug(f"Error processing node: {node[:50]}... Exception: {inner_e}")
                bounding_boxes[node] = {}
    
    finally:
        doc.close()
    
    debug("Completed get_bounding_boxes")
    return bounding_boxes

# ---------------------------
# Abstract PDFParser Class
# ---------------------------
class PDFParser(ABC):
    """Abstract base class for PDF parsers."""
    
    @abstractmethod
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and return structured content.
        """
        pass

# ---------------------------
# Gemini Flash Parser Class
# ---------------------------
class GeminiFlashParser(PDFParser):
    """Gemini Flash implementation of PDF parser using refined prompts and debugging."""
    
    CHUNKING_PROMPT = (
        "Please perform OCR on the attached PDF page image and output the text in Markdown format. "
        "Convert any tables into well-structured HTML tables without enclosing the result in triple backticks. "
        "Additionally, segment the text into semantically coherent chunks of approximately 250 to 1000 words each. "
        "Wrap each chunk with <chunk> and </chunk> tags."
    )
    
    GET_NODE_BOUNDING_BOXES_PROMPT = (
        "Given the attached image and the following text snippet:\n\n{nodes}\n\n"
        "determine the exact bounding box that encloses this text. "
        "Provide the coordinates as percentages (from 0 to 1) of the image dimensions in the order: x1, y1, x2, y2. "
        "Ensure the values are accurate and strictly in this order."
    )
    
    def __init__(self):
        debug("Initializing GeminiFlashParser")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def parse(self, pdf_path: str) -> dict:
        debug(f"GeminiFlashParser: Starting parse on {pdf_path}")
        try:
            markdown_text = extract_markdown(pdf_path, prompt=self.CHUNKING_PROMPT)
            debug("GeminiFlashParser: Markdown extraction completed.")
            
            text_nodes = extract_text_nodes(pdf_path)
            debug(f"GeminiFlashParser: Extracted {len(text_nodes)} text nodes.")
            
            raw_boxes = get_bounding_boxes(text_nodes, prompt=self.GET_NODE_BOUNDING_BOXES_PROMPT, pdf_path=pdf_path)
            debug("GeminiFlashParser: Raw bounding boxes extracted.")
            
            valid_boxes = {}
            for node, box in raw_boxes.items():
                if box:
                    try:
                        valid_boxes[node] = BoundingBox(**box)
                    except Exception as e:
                        debug(f"Warning: Skipping invalid bounding box for node: {node[:50]}... Error: {e}")
                        valid_boxes[node] = None
                else:
                    debug(f"Warning: Skipping empty bounding box for node: {node[:50]}...")
                    valid_boxes[node] = None
            
            title = os.path.basename(pdf_path)
            pages = len(fitz.open(pdf_path))
            
            metadata = Metadata(
                title=title,
                pages=pages,
                parser="gemini_flash",
                text_nodes=text_nodes,
                bounding_boxes=valid_boxes
            )
            
            parsed_pdf = ParsedPDF(
                metadata=metadata,
                content=markdown_text
            )
            
            # Flatten metadata so that keys like "title" exist at the top level.
            result = parsed_pdf.dict()
            debug(f"Before flattening, result keys: {list(result.keys())}")
            if "metadata" in result and isinstance(result["metadata"], dict):
                metadata_dict = result.pop("metadata")
                debug(f"Metadata keys: {list(metadata_dict.keys())}")
                result.update(metadata_dict)
            debug(f"After flattening, result keys: {list(result.keys())}")
            
            # NEW: Ensure that content is a list of dictionaries for consistency
            if isinstance(result.get("content"), str):
                result["content"] = [{"page": 1, "content": result["content"]}]

            debug("GeminiFlashParser: Parse completed successfully.")
            return result


        except Exception as e:
            debug(f"GeminiFlashParser: Error during parsing: {e}")
            raise PDFParsingError(f"Gemini Flash parsing failed: {str(e)}")

# ---------------------------
# Llama Parse Parser Class
# ---------------------------
class LlamaParser(PDFParser):
    """Llama Parse implementation of PDF parser."""
    
    def __init__(self):
        debug("Initializing LlamaParser")
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set")
        self.parser = LlamaParse(api_key=api_key)
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        debug(f"LlamaParser: Starting parse on {pdf_path}")
        documents = self.parser.load_data(pdf_path)
        parsed_content = []
        for doc in tqdm(documents, desc="Processing pages with Llama Parse"):
            parsed_content.append({
                'page': doc.metadata.get('page', 0) + 1,
                'content': doc.text,
                'type': 'text'
            })
        result = {
            'title': os.path.basename(pdf_path),
            'pages': len(documents),
            'content': parsed_content,
            'parser': 'llama_parse'
        }
        debug("LlamaParser: Parse completed successfully.")
        return result

# ---------------------------
# PyMuPDF4LLM Parser Class
# ---------------------------
class PyMuPDF4LLMParser(PDFParser):
    """
    PyMuPDF4LLM implementation of PDF parser that uses PyMuPDF for extraction
    and GPT-4 for text enhancement.
    """
    
    def __init__(self):
        debug("Initializing PyMuPDF4LLMParser")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def _enhance_text(self, text: str, model: str = "gpt-4o-mini") -> str:
        """
        Enhance extracted text using specified model.
        
        Args:
            text (str): Text to enhance
            model (str): Model to use for enhancement. Defaults to "gpt-4o-mini"
        """
        prompt = (
            "Please analyze and enhance the following extracted PDF text. "
            "Maintain the original information and structure, fix any OCR or formatting errors, "
            "and preserve important formatting such as lists and paragraphs. "
            "Here is the text to enhance:\n\n"
        )
        try:
            response = self.client.chat.completions.create(
                model=model,  # Use the provided model parameter
                messages=[
                    {"role": "system", "content": "You are a PDF text enhancement specialist."},
                    {"role": "user", "content": prompt + text}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            debug(f"PyMuPDF4LLMParser: Received model response: {response}")
            enhanced_text = response.choices[0].message.content
            return enhanced_text
        except Exception as e:
            debug(f"PyMuPDF4LLMParser: Error during text enhancement: {e}")
            return text  # Fallback to raw text if enhancement fails
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        debug(f"PyMuPDF4LLMParser: Starting parse on {pdf_path}")
        doc = fitz.open(pdf_path)
        parsed_content = []
        for page_num in tqdm(range(len(doc)), desc="Processing pages with PyMuPDF4LLM"):
            page = doc[page_num]
            raw_text = page.get_text()
            enhanced_text = self._enhance_text(raw_text)
            images = []
            for img in page.get_images():
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        images.append({
                            'type': 'image',
                            'format': base_image.get("ext"),
                            'data': base_image.get("image")
                        })
                except Exception as img_e:
                    debug(f"PyMuPDF4LLMParser: Error extracting image: {img_e}")
            parsed_content.append({
                'page': page_num + 1,
                'content': enhanced_text,
                'type': 'text',
                'images': images
            })
        result = {
            'title': os.path.basename(pdf_path),
            'pages': len(doc),
            'content': parsed_content,
            'parser': 'pymupdf4llm'
        }
        debug("PyMuPDF4LLMParser: Parse completed successfully.")
        return result

# ---------------------------
# PDF Parser Factory
# ---------------------------
class PDFParserFactory:
    """Factory class for creating PDF parser instances."""
    
    @staticmethod
    def get_parser(parser_type: str) -> PDFParser:
        debug(f"PDFParserFactory: Requested parser type: {parser_type}")
        parsers = {
            'gemini_flash': GeminiFlashParser,
            'llama_parse': LlamaParser,
            'pymupdf4llm': PyMuPDF4LLMParser
        }
        parser_class = parsers.get(parser_type.lower())
        if not parser_class:
            raise ValueError(f"Unknown parser type: {parser_type}")
        debug(f"PDFParserFactory: Instantiating parser: {parser_class.__name__}")
        return parser_class()

# ---------------------------
# Convenience Function
# ---------------------------
def parse_pdf(pdf_path: str, parser_type: str = 'pymupdf4llm') -> Dict[str, Any]:
    """
    Convenience function to parse a PDF file using the specified parser.
    """
    debug(f"parse_pdf: Using parser {parser_type} for file {pdf_path}")
    parser = PDFParserFactory.get_parser(parser_type)
    result = parser.parse(pdf_path)
    debug("parse_pdf: Processing complete.")
    return result

# ---------------------------
# New Helper Function: Save Parsed Content as Markdown
# ---------------------------
def save_parsed_pdf_as_markdown(parsed_result: Dict[str, Any], pdf_path: str, parser_type: str, output_dir: str = "output"):
    """
    Save the extracted text from the parsed PDF as a Markdown file under the specified output folder.
    The file name includes the input file base name, the parser type, and the current date/time.
    
    Args:
        parsed_result (Dict[str, Any]): The parsed PDF content
        pdf_path (str): Original PDF file path
        parser_type (str): Type of parser used
        output_dir (str): Directory to save the markdown file (default: "output")
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the file name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{base_name}_{parser_type}_{date_str}.md"
    file_path = os.path.join(output_dir, file_name)
    
    # Retrieve the text content. Depending on the parser, this might be a string or a list
    content = parsed_result.get("content", "")
    if isinstance(content, list):
        # Join page content if it's a list
        content = "\n\n".join(
            page.get("content", "") if isinstance(page, dict) else str(page)
            for page in content
        )
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    debug(f"Saved Markdown output to {file_path}")
    print(f"Output saved to: {file_path}")

# ---------------------------
# Module Test Code (if run as script)
# ---------------------------
if __name__ == "__main__":
    # Example usage for debugging purposes.
    sample_pdf = "sample.pdf"  # Replace with a valid PDF path for testing.
    chosen_parser = "gemini_flash"  # Change to "pymupdf4llm" or "llama_parse" as needed.
    
    try:
        result = parse_pdf(sample_pdf, parser_type=chosen_parser)
        print("Parsing result:")
        print(json.dumps(result, indent=2, default=str))
        
        # Save the parsed text as Markdown.
        save_parsed_pdf_as_markdown(result, sample_pdf, chosen_parser)
    except Exception as e:
        print(f"Error during parsing: {e}")
