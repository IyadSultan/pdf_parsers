"""
Main Script

This script supports two modes:

1. Full Pipeline Mode:
   - Processes PDFs using all three available parsers:
     • pymupdf4llm
     • gemini_flash
     • llama_parse
   - Saves the resulting knowledge graphs.
   - Runs deepeval evaluations on each graph.
   - Saves individual and combined evaluation results into the "evaluation" folder.

2. Evaluation-Only Mode:
   - Loads the last prepared knowledge graphs from the three parser folders.
   - Runs deepeval evaluations using sample test queries.
   - Saves individual and combined evaluation results into the "evaluation" folder.

Usage Examples:
--------------
To run the full pipeline (processing PDFs and evaluation):
    python main.py --input data/pdfs/sample.pdf

To run evaluation only (using the already saved knowledge graphs):
    python main.py --eval-only
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.pipeline import create_pipeline
from src.pdf_parsers import save_parsed_pdf_as_markdown

def setup_directories():
    """Create necessary directories for outputs if they don't already exist."""
    directories = [
        'data/pdfs',
        'graphs/pymupdf4llm',
        'graphs/gemini_flash',
        'graphs/llama_parse',
        'output/markdown',
        'evaluation'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_test_queries():
    """Load test queries from the sample JSON file."""
    test_queries_file = os.path.join("data", "evaluations", "sample.json")
    with open(test_queries_file, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_all_knowledge_graphs():
    """
    Load each saved knowledge graph, evaluate it using deepeval,
    and save both individual and combined results.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    evaluation_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    test_queries = get_test_queries()
    parser_types = ['pymupdf4llm', 'gemini_flash', 'llama_parse']
    evaluation_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for parser in parser_types:
        graph_dir = os.path.join(base_dir, "graphs", parser)
        print(f"\nEvaluating knowledge graph for '{parser}' from: {graph_dir}")
        pipeline = create_pipeline(parser_type=parser, graph_dir=graph_dir)
        # Define an output file for individual evaluation results.
        eval_output_file = os.path.join(evaluation_dir, f"{parser}_eval_{timestamp}.json")
        results = pipeline.evaluate(
            test_queries=test_queries,
            output_file=eval_output_file,
            model="gpt-4o-mini"
        )
        evaluation_results[parser] = results
        print(f"Evaluation results for '{parser}' saved to: {eval_output_file}")
    
    # Save a combined evaluation comparison file.
    comparison_file = os.path.join(evaluation_dir, f"evaluation_comparison_{timestamp}.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"\nCombined evaluation results saved to: {comparison_file}")
    
    # Print a summary of the evaluations.
    print("\nEvaluation Summary:")
    print("=" * 50)
    for parser_type, eval_results in evaluation_results.items():
        print(f"\n{parser_type}:")
        # Convert to dictionary if it's a Pydantic model.
        if hasattr(eval_results, "dict"):
            eval_dict = eval_results.dict()
        else:
            eval_dict = eval_results
        for metric, score in eval_dict.items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.2f}")
            else:
                print(f"  {metric}: {score}")

def process_with_parser(parser_type: str, input_path: str):
    """
    Process PDFs using the specified parser, save the knowledge graph,
    and run evaluation on the results.
    
    Args:
        parser_type (str): Type of parser to use.
        input_path (str): Path to a PDF file or directory.
    
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
        result = pipeline.process_pdf(input_path)
        results = [result]
        print(f"Processed {result['title']} with {result['pages']} pages")
    else:
        print(f"Processing directory: {input_path}")
        results = pipeline.process_directory(input_path)
        print(f"Processed {len(results)} PDFs")
    
    # Save knowledge graph
    graph_dir = os.path.join("graphs", parser_type)
    print(f"\nSaving knowledge graph to {graph_dir}")
    pipeline.save_knowledge_graph(graph_dir)
    
    # Save parsed content as markdown (optional)
    for result in results:
        save_parsed_pdf_as_markdown(result, input_path, parser_type, os.path.join("output", "markdown"))
    
    # Run evaluation with specified model
    print(f"\nRunning evaluation for {parser_type}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_file = os.path.join("evaluation", f"{parser_type}_eval_{timestamp}.json")
    
    test_queries = get_test_queries()
    evaluation_results = pipeline.evaluate(
        test_queries=test_queries,
        output_file=eval_output_file,
        model="gpt-4o-mini"
    )
    
    print(f"Evaluation results saved to: {eval_output_file}")
    return results, evaluation_results

def main():
    """Main execution function."""
    load_dotenv()         # Load environment variables
    setup_directories()   # Ensure necessary folders exist
    
    # Parse command-line arguments to determine mode.
    parser = argparse.ArgumentParser(description="Run the PDF processing pipeline and/or evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only on existing knowledge graphs.")
    parser.add_argument("--input", type=str, default="data/pdfs/sample.pdf", help="Path to input PDF file or directory (used in full pipeline mode).")
    args = parser.parse_args()
    
    if args.eval_only:
        # Run evaluation only on the existing knowledge graphs.
        evaluate_all_knowledge_graphs()
    else:
        # Run full pipeline: process PDFs with all parsers and evaluate.
        parsers_list = ['pymupdf4llm', 'gemini_flash', 'llama_parse']
        all_results = {}
        evaluation_results = {}
        
        if not os.path.exists(args.input):
            print(f"Error: Input path '{args.input}' does not exist.")
            sys.exit(1)
        
        for parser_type in parsers_list:
            try:
                results, eval_results = process_with_parser(parser_type, args.input)
                all_results[parser_type] = results
                evaluation_results[parser_type] = eval_results
            except Exception as e:
                print(f"Error processing with {parser_type}: {str(e)}")
        
        # Save processing comparison results.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join("output", f"comparison_results_{timestamp}.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nProcessing comparison results saved to: {comparison_file}")
        
        # Save evaluation comparison.
        eval_comparison_file = os.path.join("evaluation", f"evaluation_comparison_{timestamp}.json")
        with open(eval_comparison_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"Evaluation comparison saved to: {eval_comparison_file}")
        
        # Print evaluation summary.
        print("\nEvaluation Summary:")
        print("=" * 50)
        for parser_type, eval_results in evaluation_results.items():
            print(f"\n{parser_type}:")
            if hasattr(eval_results, "dict"):
                eval_dict = eval_results.dict()
            else:
                eval_dict = eval_results
            for metric, score in eval_dict.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.2f}")
                else:
                    print(f"  {metric}: {score}")

if __name__ == "__main__":
    main()


# Run the full pipeline:
# python main.py --input data/pdfs/sample.pdf

# Run evaluation only:
# python main.py --eval-only
