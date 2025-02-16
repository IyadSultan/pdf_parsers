"""
Main Script

This script loads the last prepared knowledge graphs from the three parser folders,
runs deepeval evaluation on each using sample test queries, and then saves the 
individual and combined evaluation results into the evaluation folder.
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
# (Optional) If you want to save markdown outputs later:
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
    for parser, eval_results in evaluation_results.items():
        print(f"\n{parser}:")
        for metric, score in eval_results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.2f}")
            else:
                print(f"  {metric}: {score}")

def main():
    """Main execution function."""
    load_dotenv()         # Load environment variables
    setup_directories()   # Ensure necessary folders exist
    evaluate_all_knowledge_graphs()  # Run evaluation on the three knowledge graphs

if __name__ == "__main__":
    main()


