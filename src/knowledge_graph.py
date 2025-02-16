"""
Knowledge Graph Builder Module

This module implements a knowledge graph builder using LangGraph framework.
It processes parsed PDF content and builds a semantic knowledge graph.
"""

import os
import re
from typing import Dict, List, Any, Optional
import urllib.parse
from dotenv import load_dotenv
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import json 

# Load environment variables
load_dotenv()

def clean_llm_output(llm_output: str) -> str:
    """
    Remove markdown code fences from the LLM output for valid JSON parsing.
    """
    llm_output = llm_output.strip()
    # Remove markdown code fences if present
    if llm_output.startswith("```"):
        lines = llm_output.splitlines()
        # Remove opening fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        llm_output = "\n".join(lines).strip()
    return llm_output

def create_valid_uri(text: str, prefix: str = "http://example.org/") -> str:
    """
    Create a valid URI from text by encoding special characters.
    """
    # Remove special characters and spaces, replace with underscores
    clean_text = re.sub(r'[^\w\s-]', '', text)
    clean_text = re.sub(r'[-\s]+', '_', clean_text)
    # URL encode the cleaned text
    encoded_text = urllib.parse.quote(clean_text)
    return f"{prefix}{encoded_text}"

class KnowledgeGraphBuilder:
    """Builds and manages a knowledge graph from parsed PDF content."""
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.graph = nx.DiGraph()
        self.rdf_graph = Graph()
        self.llm = ChatOpenAI(model="gpt-4-0125-preview")
        
    def _extract_entities_and_relations(self, content: str) -> Dict[str, Any]:
        """
        Extract entities and relations from text content using LLM.
        
        Args:
            content (str): Text content to analyze
            
        Returns:
            Dict[str, Any]: Extracted entities and relations
        """
        system_prompt = """
        Extract key entities and their relationships from the given text.
        Return only the JSON data without any markdown formatting or code blocks.
        Format:
        {
            "entities": [{"name": "entity_name", "type": "entity_type"}],
            "relations": [{"source": "entity1", "target": "entity2", "relation": "relation_type"}]
        }
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        response = self.llm.invoke(messages)
        cleaned_response = clean_llm_output(response.content)
        return eval(cleaned_response)  # Safe since we control the LLM prompt format
        
    def add_document(self, parsed_content: Dict[str, Any]) -> None:
        """
        Add a parsed document to the knowledge graph.
        
        Args:
            parsed_content (Dict[str, Any]): Parsed PDF content
        """
        document_title = parsed_content['title']
        
        # Add document node with valid URI
        self.graph.add_node(document_title, type='document')
        doc_uri = create_valid_uri(document_title, "http://example.org/doc/")
        self.rdf_graph.add((URIRef(doc_uri), RDF.type, URIRef("http://example.org/Document")))
        
        # Process each page's content
        for page in parsed_content['content']:
            content_text = page['content']
            extracted_info = self._extract_entities_and_relations(content_text)
            
            # Add entities
            for entity in extracted_info['entities']:
                entity_name = entity['name']
                entity_type = entity['type']
                entity_id = f"{entity_name}_{entity_type}"
                
                self.graph.add_node(entity_id, 
                                  type='entity',
                                  entity_type=entity_type,
                                  name=entity_name)
                
                # Connect entity to document
                self.graph.add_edge(document_title, entity_id, 
                                  type='contains',
                                  page=page['page'])
                
                # Add to RDF graph with valid URIs
                entity_uri = create_valid_uri(entity_id, "http://example.org/entity/")
                type_uri = create_valid_uri(entity_type, "http://example.org/")
                
                self.rdf_graph.add((URIRef(entity_uri), RDF.type, URIRef(type_uri)))
                self.rdf_graph.add((URIRef(entity_uri), 
                                  URIRef("http://example.org/name"), 
                                  Literal(entity_name)))
                self.rdf_graph.add((URIRef(doc_uri), 
                                  URIRef("http://example.org/contains"), 
                                  URIRef(entity_uri)))
            
            # Add relations
            for relation in extracted_info['relations']:
                source_id = f"{relation['source']}_entity"
                target_id = f"{relation['target']}_entity"
                
                self.graph.add_edge(source_id, target_id,
                                  type='relation',
                                  relation_type=relation['relation'])
                
                # Add to RDF graph with valid URIs
                source_uri = create_valid_uri(source_id, "http://example.org/entity/")
                target_uri = create_valid_uri(target_id, "http://example.org/entity/")
                relation_uri = create_valid_uri(relation['relation'], "http://example.org/relation/")
                
                self.rdf_graph.add((URIRef(source_uri), 
                                  URIRef(relation_uri),
                                  URIRef(target_uri)))

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using natural language.
        
        Args:
            query (str): Natural language query
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        system_prompt = (
            "Convert the following natural language query into a JSON object representing a graph pattern.\n"
            f"Query: {query}\n"
            "The JSON object should have the following keys:\n"
            "- node_patterns: a list of node patterns to match,\n"
            "- edge_patterns: a list of edge patterns to match,\n"
            "- conditions: any conditions to filter.\n"
            "Return only the JSON (do not include any extra text)."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        cleaned_response = clean_llm_output(response.content)
        try:
            query_pattern = json.loads(cleaned_response)
        except Exception as e:
            print("Error parsing LLM response into JSON. Response was:")
            print(cleaned_response)
            raise e
        
        matching_subgraphs = []
        for subgraph in nx.connected_components(self.graph.to_undirected()):
            subgraph_view = self.graph.subgraph(subgraph)
            if self._matches_pattern(subgraph_view, query_pattern):
                matching_subgraphs.append(subgraph_view)
                
        return self._format_results(matching_subgraphs)

    
    def _matches_pattern(self, subgraph: nx.DiGraph, pattern: Dict[str, Any]) -> bool:
        """Check if a subgraph matches the query pattern."""
        # Implementation of pattern matching logic
        return True  # Placeholder
        
    def _format_results(self, subgraphs: List[nx.DiGraph]) -> List[Dict[str, Any]]:
        """Format matching subgraphs into readable results."""
        results = []
        for subgraph in subgraphs:
            result = {
                'nodes': list(subgraph.nodes(data=True)),
                'edges': list(subgraph.edges(data=True))
            }
            results.append(result)
        return results
    
    def save_graph(self, output_dir: str) -> None:
        """
        Save the knowledge graph to files.
        
        Args:
            output_dir (str): Directory to save the graph files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save NetworkX graph
        nx.write_gexf(self.graph, os.path.join(output_dir, "knowledge_graph.gexf"))
        
        # Save RDF graph
        self.rdf_graph.serialize(
            destination=os.path.join(output_dir, "knowledge_graph.ttl"),
            format="turtle"
        )
        
    def load_graph(self, input_dir: str) -> None:
        """
        Load a previously saved knowledge graph.
        
        Args:
            input_dir (str): Directory containing the graph files
        """
        # Load NetworkX graph
        self.graph = nx.read_gexf(os.path.join(input_dir, "knowledge_graph.gexf"))
        
        # Load RDF graph
        self.rdf_graph = Graph()
        self.rdf_graph.parse(
            os.path.join(input_dir, "knowledge_graph.ttl"),
            format="turtle"
        ) 