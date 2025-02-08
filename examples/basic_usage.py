"""
Basic Usage Example

This script demonstrates the core functionality of the PDF Knowledge Graph pipeline:
1. Creating a pipeline
2. Processing PDFs
3. Querying the knowledge graph
4. Running evaluation
"""

import os
import json
from dotenv import load_dotenv
from src.pipeline import create_pipeline

# Load environment variables
load_dotenv()

def main():
    # Create pipeline with default PyMuPDF parser
    pipeline = create_pipeline()
    
    # Process a single PDF
    print("Processing single PDF...")
    single_result = pipeline.process_pdf('data/pdfs/sample.pdf')
    print(f"Processed {single_result['title']} with {single_result['pages']} pages")
    
    # Process a directory of PDFs
    print("\nProcessing directory of PDFs...")
    dir_results = pipeline.process_directory('data/pdfs')
    print(f"Processed {len(dir_results)} PDFs")
    
    # Save the knowledge graph
    print("\nSaving knowledge graph...")
    pipeline.save_knowledge_graph('graphs/saved')
    
    # Query the knowledge graph
    print("\nQuerying knowledge graph...")
    queries = [
        "What are the main topics discussed in the documents?",
        "What are the key findings from the research?",
        "Who are the main authors and their affiliations?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = pipeline.query_knowledge_graph(query)
        print("Results:", json.dumps(results, indent=2))
    
    # Run evaluation
    print("\nRunning evaluation...")
    test_queries = [
        {
            'question': "What is the main conclusion of the research?",
            'expected_answer': "The research concludes that knowledge graphs significantly improve information retrieval from PDFs."
        },
        {
            'question': "What methods were used in the study?",
            'expected_answer': "The study used a combination of PDF parsing techniques and graph-based knowledge representation."
        }
    ]
    
    evaluation_results = pipeline.evaluate(
        test_queries,
        output_file='evaluation_results.json'
    )
    
    print("\nEvaluation Results:")
    print(json.dumps(evaluation_results, indent=2))

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/pdfs', exist_ok=True)
    os.makedirs('graphs/saved', exist_ok=True)
    
    # Run example
    main() 