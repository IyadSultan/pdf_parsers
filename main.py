import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from src.pipeline import create_pipeline

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    required_vars = [
        'GOOGLE_API_KEY',
        'LLAMA_CLOUD_API_KEY', 
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print("Warning: The following API keys are missing:")
        for var in missing:
            print(f"- {var}")
        print("\nSome features may be limited.")

def select_parser() -> str:
    """Let user select the PDF parser to use"""
    print("\nAvailable PDF Parsers:")
    print("1. PyMuPDF4LLM (Fast, lightweight, best for simple PDFs)")
    print("2. Gemini Flash (Best for complex layouts, requires Google API)")
    print("3. Llama Parse (Strong structure preservation, requires Llama API)")
    
    while True:
        choice = input("\nSelect parser (1-3): ").strip()
        if choice == '1':
            return 'pymupdf4llm'
        elif choice == '2':
            return 'gemini_flash'
        elif choice == '3':
            return 'llama_parse'
        print("Invalid choice. Please select 1, 2, or 3.")

def get_input_path() -> Path:
    """Get input PDF file or directory path from user"""
    while True:
        path = input("\nEnter path to PDF file or directory: ").strip()
        path = Path(path)
        
        if not path.exists():
            print("Path does not exist. Please try again.")
            continue
            
        if path.is_file() and path.suffix.lower() != '.pdf':
            print("File must be a PDF. Please try again.")
            continue
            
        return path

def get_output_path() -> Path:
    """Get output directory path from user"""
    while True:
        path = input("\nEnter output directory path (default: ./output): ").strip()
        if not path:
            path = './output'
            
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            print(f"Error creating directory: {e}")
            print("Please try a different path.")

def process_documents(
    pipeline,
    input_path: Path,
    output_path: Path
) -> None:
    """Process PDF document(s) and save results"""
    
    try:
        if input_path.is_file():
            print(f"\nProcessing file: {input_path}")
            results = pipeline.process_pdf(str(input_path))
        else:
            print(f"\nProcessing directory: {input_path}")
            results = pipeline.process_directory(str(input_path))
            
        # Save knowledge graph
        graph_path = output_path / 'knowledge_graph'
        pipeline.save_knowledge_graph(str(graph_path))
        
        # Save processing results
        results_file = output_path / 'processing_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nProcessing completed successfully!")
        print(f"\nResults saved to:")
        print(f"- Knowledge Graph: {graph_path}")
        print(f"- Processing Results: {results_file}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")

def main():
    """Main application entry point"""
    print("\n=== PDF Knowledge Graph Builder ===\n")
    
    # Load environment variables
    load_environment()
    
    # Get user inputs
    parser_type = select_parser()
    input_path = get_input_path()
    output_path = get_output_path()
    
    # Create and configure pipeline
    print("\nInitializing pipeline...")
    pipeline = create_pipeline(
        parser_type=parser_type,
        graph_dir=str(output_path / 'knowledge_graph')
    )
    
    # Process documents
    process_documents(pipeline, input_path, output_path)

if __name__ == "__main__":
    main()
