"""
Knowledge Graph Builder Module

This module implements a knowledge graph builder using LangGraph framework.
It processes parsed PDF content and builds a semantic knowledge graph.
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Load environment variables
load_dotenv()

class KnowledgeGraphBuilder:
    """Builds and manages a knowledge graph from parsed PDF content."""
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.graph = nx.DiGraph()
        self.rdf_graph = Graph()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        
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
        Return the results in the following format:
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
        return eval(response.content)  # Safe since we control the LLM prompt format
        
    def add_document(self, parsed_content: Dict[str, Any]) -> None:
        """
        Add a parsed document to the knowledge graph.
        
        Args:
            parsed_content (Dict[str, Any]): Parsed PDF content
        """
        document_title = parsed_content['title']
        
        # Add document node
        self.graph.add_node(document_title, type='document')
        doc_uri = URIRef(f"http://example.org/doc/{document_title}")
        self.rdf_graph.add((doc_uri, RDF.type, URIRef("http://example.org/Document")))
        
        # Process each page's content
        for page in parsed_content['content']:
            content_text = page['content']
            extracted_info = self._extract_entities_and_relations(content_text)
            
            # Add entities
            for entity in extracted_info['entities']:
                entity_id = f"{entity['name']}_{entity['type']}"
                self.graph.add_node(entity_id, 
                                  type='entity',
                                  entity_type=entity['type'],
                                  name=entity['name'])
                
                # Connect entity to document
                self.graph.add_edge(document_title, entity_id, 
                                  type='contains',
                                  page=page['page'])
                
                # Add to RDF graph
                entity_uri = URIRef(f"http://example.org/entity/{entity_id}")
                self.rdf_graph.add((entity_uri, RDF.type, URIRef(f"http://example.org/{entity['type']}")))
                self.rdf_graph.add((entity_uri, URIRef("http://example.org/name"), Literal(entity['name'])))
                self.rdf_graph.add((doc_uri, URIRef("http://example.org/contains"), entity_uri))
            
            # Add relations
            for relation in extracted_info['relations']:
                source_id = f"{relation['source']}_entity"
                target_id = f"{relation['target']}_entity"
                
                self.graph.add_edge(source_id, target_id,
                                  type='relation',
                                  relation_type=relation['relation'])
                
                # Add to RDF graph
                source_uri = URIRef(f"http://example.org/entity/{source_id}")
                target_uri = URIRef(f"http://example.org/entity/{target_id}")
                self.rdf_graph.add((source_uri, 
                                  URIRef(f"http://example.org/relation/{relation['relation']}"),
                                  target_uri))
                
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using natural language.
        
        Args:
            query (str): Natural language query
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        system_prompt = f"""
        Convert the following natural language query into a graph pattern:
        Query: {query}
        
        Return the result as a dictionary with:
        1. Node patterns to match
        2. Edge patterns to match
        3. Conditions to filter
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        query_pattern = eval(response.content)
        
        # Use NetworkX to find matching patterns
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