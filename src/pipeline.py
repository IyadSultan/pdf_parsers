"""
Main Pipeline Module

This module implements the main pipeline that:
1. Parses PDFs using the selected parser
2. Builds a knowledge graph
3. Evaluates the results using deep_eval
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .pdf_parsers import parse_pdf, debug
from .knowledge_graph import KnowledgeGraphBuilder
from deepeval import evaluate
from deepeval.metrics import (
    GEval,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
import json

# Load environment variables
load_dotenv()

class Pipeline:
    """Main pipeline for PDF processing and knowledge graph building."""
    
    def __init__(self, parser_type: str = 'pymupdf4llm'):
        """
        Initialize the pipeline.
        
        Args:
            parser_type (str): Type of PDF parser to use
        """
        debug(f"Initializing pipeline with parser: {parser_type}")
        self.parser_type = parser_type
        self.graph_builder = KnowledgeGraphBuilder()
        
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Processing results
        """
        debug(f"Processing PDF: {pdf_path}")
        
        # Parse PDF
        parsed_content = parse_pdf(pdf_path, self.parser_type)
        debug("PDF parsing completed")
        
        # Add to knowledge graph
        debug("Building knowledge graph...")
        self.graph_builder.add_document(parsed_content)
        debug("Knowledge graph updated")
        
        return parsed_content
        
    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            dir_path (str): Path to directory containing PDFs
            
        Returns:
            List[Dict[str, Any]]: Processing results for each PDF
        """
        debug(f"Processing directory: {dir_path}")
        results = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(dir_path, filename)
                result = self.process_pdf(pdf_path)
                results.append(result)
        return results
        
    def save_knowledge_graph(self, output_dir: str) -> None:
        """
        Save the current knowledge graph.
        
        Args:
            output_dir (str): Directory to save the graph
        """
        debug(f"Saving knowledge graph to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.graph_builder.save_graph(output_dir)
        debug("Knowledge graph saved")
        
    def load_knowledge_graph(self, input_dir: str) -> None:
        """
        Load a previously saved knowledge graph.
        
        Args:
            input_dir (str): Directory containing the graph files
        """
        self.graph_builder.load_graph(input_dir)
        
    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            query (str): Natural language query
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        return self.graph_builder.query_graph(query)

    def evaluate(self, test_queries: List[Dict[str, str]], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the pipeline using test queries.

        Args:
            test_queries (List[Dict[str, str]]): List of test queries. Each query should include:
                - "question": the natural language query.
                - "expected_answer": the ground truth answer.
            output_file (Optional[str]): Path to a JSON file to save evaluation results.

        Returns:
            Dict[str, Any]: The evaluation results.
        """
        # Define evaluation metrics with explicit evaluation_steps.
        metrics = [
            GEval(
                name="Answer Correctness",
                model="gpt-4",
                evaluation_params=["input", "expected_output", "actual_output"],
                evaluation_steps=[
                    "Compare the actual answer with the expected answer and determine if the factual content is correct."
                ]
            ),
            FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4",
                evaluation_steps=[
                    "Evaluate whether the answer faithfully reflects the content of the retrieved context and source data."
                ]
            ),
            ContextualRelevancyMetric(
                threshold=0.7,
                model="gpt-4",
                evaluation_steps=[
                    "Assess whether the answer is contextually relevant to the question and the information contained in the knowledge graph."
                ]
            )
        ]
        
        # Create test cases based on the provided queries.
        test_cases = []
        for query in test_queries:
            # Retrieve results from the knowledge graph using the query.
            results = self.query_knowledge_graph(query['question'])
            # Format the retrieved results into a single answer string.
            actual_answer = self._format_answer(results)
            
            # Create a test case with input, expected output, and the actual output.
            test_case = LLMTestCase(
                input=query['question'],
                expected_output=query['expected_answer'],
                actual_output=actual_answer
            )
            test_cases.append(test_case)
        
        # Run the evaluation using the defined metrics.
        evaluation_results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )
        
        # Save the evaluation results to a JSON file if an output file path is provided.
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results


    def _format_answer(self, query_results: List[Dict[str, Any]]) -> str:
        """Format query results into a readable answer."""
        # Implementation of answer formatting logic
        return str(query_results)  # Placeholder

def create_pipeline(
    parser_type: str = 'pymupdf4llm',
    graph_dir: Optional[str] = None
) -> Pipeline:
    """
    Create and optionally initialize a pipeline with an existing knowledge graph.
    
    Args:
        parser_type (str): Type of PDF parser to use
        graph_dir (Optional[str]): Directory containing existing knowledge graph
        
    Returns:
        Pipeline: Initialized pipeline instance
    """
    pipeline = Pipeline(parser_type)
    
    if graph_dir and os.path.exists(graph_dir):
        pipeline.load_knowledge_graph(graph_dir)
        
    return pipeline 