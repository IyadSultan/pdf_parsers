"""
PDF Parser Factory Module

This module provides a factory pattern implementation for different PDF parsers:
- Gemini Flash
- Llama Parse
- PyMuPDF4LLM

Each parser implements a common interface for consistency and easy switching.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from llama_parse import LlamaParse
import fitz  # PyMuPDF
import json

# Load environment variables
load_dotenv()

class PDFParser(ABC):
    """Abstract base class for PDF parsers."""
    
    @abstractmethod
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and return structured content.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Parsed content with metadata
        """
        pass

class GeminiFlashParser(PDFParser):
    """Gemini Flash implementation of PDF parser."""
    
    def __init__(self):
        """Initialize Gemini Flash parser with API key."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using Gemini Flash capabilities."""
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Read PDF and convert to images
        doc = fitz.open(pdf_path)
        parsed_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes()
            
            # Process each page with Gemini
            response = model.generate_content([img_data])
            parsed_content.append({
                'page': page_num + 1,
                'content': response.text,
                'type': 'text'
            })
            
        return {
            'title': os.path.basename(pdf_path),
            'pages': len(doc),
            'content': parsed_content,
            'parser': 'gemini_flash'
        }

class LlamaParser(PDFParser):
    """Llama Parse implementation of PDF parser."""
    
    def __init__(self):
        """Initialize Llama Parse with API key."""
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set")
        self.parser = LlamaParse(api_key=api_key)
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using Llama Parse."""
        documents = self.parser.load_data(pdf_path)
        parsed_content = []
        
        for doc in documents:
            parsed_content.append({
                'page': doc.metadata.get('page', 0) + 1,
                'content': doc.text,
                'type': 'text'
            })
            
        return {
            'title': os.path.basename(pdf_path),
            'pages': len(documents),
            'content': parsed_content,
            'parser': 'llama_parse'
        }

class PyMuPDF4LLMParser(PDFParser):
    """PyMuPDF4LLM implementation of PDF parser."""
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        parsed_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract images if available
            images = []
            for img in page.get_images():
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    images.append({
                        'type': 'image',
                        'format': base_image["ext"],
                        'data': base_image["image"]
                    })
            
            parsed_content.append({
                'page': page_num + 1,
                'content': text,
                'type': 'text',
                'images': images
            })
            
        return {
            'title': os.path.basename(pdf_path),
            'pages': len(doc),
            'content': parsed_content,
            'parser': 'pymupdf4llm'
        }

class PDFParserFactory:
    """Factory class for creating PDF parsers."""
    
    @staticmethod
    def get_parser(parser_type: str) -> PDFParser:
        """
        Get a PDF parser instance based on the specified type.
        
        Args:
            parser_type (str): Type of parser ('gemini_flash', 'llama_parse', or 'pymupdf4llm')
            
        Returns:
            PDFParser: Instance of the specified parser
            
        Raises:
            ValueError: If parser_type is not recognized
        """
        parsers = {
            'gemini_flash': GeminiFlashParser,
            'llama_parse': LlamaParser,
            'pymupdf4llm': PyMuPDF4LLMParser
        }
        
        parser_class = parsers.get(parser_type.lower())
        if not parser_class:
            raise ValueError(f"Unknown parser type: {parser_type}")
            
        return parser_class()

def parse_pdf(pdf_path: str, parser_type: str = 'pymupdf4llm') -> Dict[str, Any]:
    """
    Convenience function to parse a PDF file using the specified parser.
    
    Args:
        pdf_path (str): Path to the PDF file
        parser_type (str): Type of parser to use (default: 'pymupdf4llm')
        
    Returns:
        Dict[str, Any]: Parsed content with metadata
    """
    parser = PDFParserFactory.get_parser(parser_type)
    return parser.parse(pdf_path) 