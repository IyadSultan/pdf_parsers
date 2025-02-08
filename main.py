"""
Main Script

This script runs the PDF processing pipeline using all three available parsers:
1. PyMuPDF4LLM
2. Gemini Flash
3. Llama Parse

For each parser, it:
1. Parses PDFs and saves raw output
2. Creates a knowledge graph
3. Saves the knowledge graph
4. Runs evaluation
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.pipeline import create_pipeline
from src.pdf_parsers import parse_pdf, save_parsed_pdf_as_markdown

# Load environment variables
load_dotenv()

def setup_directories():
    """Create necessary directories for outputs."""
    directories = [
        'data/pdfs',
        'graphs/pymupdf4llm',
        'graphs/gemini_flash',
        'graphs/llama_parse',
        'output/parsed',
        'output/markdown',
        'evaluation'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def parse_and_save(parser_type: str, input_path: str) -> dict:
    """
    Parse PDF and save raw output before processing.
    
    Args:
        parser_type (str): Type of parser to use
        input_path (str): Path to PDF file
        
    Returns:
        dict: Parsed content
    """
    print(f"\nParsing with {parser_type}...")
    
    # Parse the PDF
    parsed_content = parse_pdf(input_path, parser_type=parser_type)
    
    # Save raw parsed content as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    json_filename = f"{base_name}_{parser_type}_{timestamp}.json"
    json_path = os.path.join('output/parsed', json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_content, f, indent=2, default=str)
    print(f"Saved parsed content to: {json_path}")
    
    # Save as markdown for readability
    save_parsed_pdf_as_markdown(
        parsed_content, 
        input_path, 
        parser_type,
        output_dir='output/markdown'
    )
    
    return parsed_content

def process_with_parser(parser_type: str, input_path: str):
    """
    Process PDFs using specified parser and save results.
    
    Args:
        parser_type (str): Type of parser to use
        input_path (str): Path to PDF file or directory
        
    Returns:
        tuple: (processing_results, evaluation_results)
    """
    print(f"\n{'='*50}")
    print(f"Processing with {parser_type}")
    print(f"{'='*50}")
    
    # Create pipeline with specified parser
    pipeline = create_pipeline(parser_type=parser_type)
    
    # Process PDF(s)
    if os.path.isfile(input_path):
        print(f"Processing single PDF: {input_path}")
        parsed_content = parse_and_save(parser_type, input_path)
        result = pipeline.process_parsed_content(parsed_content)
        results = [result]
    else:
        print(f"Processing directory: {input_path}")
        results = []
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_path, filename)
                parsed_content = parse_and_save(parser_type, pdf_path)
                result = pipeline.process_parsed_content(parsed_content)
                results.append(result)
    
    # Save knowledge graph
    graph_dir = f'graphs/{parser_type}'
    print(f"Saving knowledge graph to {graph_dir}")
    pipeline.save_knowledge_graph(graph_dir)
    
    # Run evaluation
    print(f"\nRunning evaluation for {parser_type}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_file = f'evaluation/{parser_type}_eval_{timestamp}.json'
    
    test_queries = get_test_queries()
    evaluation_results = pipeline.evaluate(
        test_queries=test_queries,
        output_file=eval_output_file
    )
    
    print(f"Evaluation results saved to: {eval_output_file}")
    
    return results, evaluation_results

def get_test_queries():
    """Define test queries for evaluation."""
    return [
        {
            'question': "What are the main topics discussed in the document?",
            'expected_answer': "The document discusses key topics related to its content. The exact topics should be identified from the actual document content."
        },
        {
            'question': "What are the key findings or conclusions?",
            'expected_answer': "The document presents specific findings and conclusions related to its main topics. These should be accurately extracted from the document content."
        },
        {
            'question': "Who are the main entities mentioned and what are their relationships?",
            'expected_answer': "The document mentions specific entities and describes relationships between them. These should be correctly identified from the document content."
        }
    ]

def main():
    """Main execution function."""
    # Create necessary directories
    setup_directories()
    
    # Define parsers to use
    parsers = ['pymupdf4llm', 'gemini_flash', 'llama_parse']
    
    # Get input path (file or directory)
    input_path = 'data/pdfs/sample.pdf'  # Default single file
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
    
    # Process with each parser
    all_results = {}
    evaluation_results = {}
    
    for parser_type in parsers:
        try:
            results, eval_results = process_with_parser(parser_type, input_path)
            all_results[parser_type] = results
            evaluation_results[parser_type] = eval_results
        except Exception as e:
            print(f"Error processing with {parser_type}: {str(e)}")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save processing results
    comparison_file = f'output/comparison_results_{timestamp}.json'
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nComparison results saved to: {comparison_file}")
    
    # Save evaluation comparison
    eval_comparison_file = f'evaluation/evaluation_comparison_{timestamp}.json'
    with open(eval_comparison_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"Evaluation comparison saved to: {eval_comparison_file}")
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    for parser_type, eval_results in evaluation_results.items():
        print(f"\n{parser_type}:")
        for metric, score in eval_results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.2f}")
            else:
                print(f"  {metric}: {score}")

if __name__ == "__main__":
    main()
