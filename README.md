# PDF Knowledge Graph Builder

A powerful Python-based pipeline for parsing PDFs, building knowledge graphs, and evaluating the results using LangGraph and deep_eval.

## Features

- Multiple PDF parser options:
  - Gemini Flash: Google's advanced vision-language model
  - Llama Parse: Robust PDF parsing with structure preservation
  - PyMuPDF4LLM: Lightweight and fast PDF processing
- Knowledge Graph Building:
  - Entity and relationship extraction
  - Semantic graph construction
  - Support for both NetworkX and RDF formats
- Natural Language Querying:
  - Query the knowledge graph using plain English
  - Get structured responses with source tracking
- Evaluation Framework:
  - Answer correctness assessment
  - Faithfulness checking
  - Contextual relevancy metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-knowledge-graph.git
cd pdf-knowledge-graph
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```env
GOOGLE_API_KEY=your_gemini_api_key
LLAMA_CLOUD_API_KEY=your_llama_parse_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Basic Usage

1. Create a pipeline instance:
```python
from src.pipeline import create_pipeline

# Create pipeline with default PyMuPDF parser
pipeline = create_pipeline()

# Or specify a different parser
pipeline = create_pipeline(parser_type='gemini_flash')
```

2. Process a single PDF:
```python
results = pipeline.process_pdf('path/to/your/document.pdf')
```

3. Process multiple PDFs in a directory:
```python
results = pipeline.process_directory('path/to/pdf/directory')
```

4. Query the knowledge graph:
```python
query = "What are the main topics discussed in the documents?"
results = pipeline.query_knowledge_graph(query)
```

### Adding More PDFs to Existing Knowledge Graph

1. Save the current knowledge graph:
```python
pipeline.save_knowledge_graph('path/to/graph/directory')
```

2. Later, load the saved graph and add more documents:
```python
pipeline = create_pipeline(graph_dir='path/to/graph/directory')
pipeline.process_pdf('path/to/new/document.pdf')
```

### Evaluation

1. Prepare test queries:
```python
test_queries = [
    {
        'question': 'What is the main conclusion of the research?',
        'expected_answer': 'The research concludes that...'
    },
    # Add more test queries...
]
```

2. Run evaluation:
```python
evaluation_results = pipeline.evaluate(
    test_queries,
    output_file='evaluation_results.json'
)
```

## Building an Evaluation Pipeline

1. Create a test dataset:
```python
# test_queries.json
{
    "queries": [
        {
            "question": "What are the key findings?",
            "expected_answer": "The key findings include...",
            "context": "Section 4.2 of the paper..."
        }
    ]
}
```

2. Run comprehensive evaluation:
```python
import json
from src.pipeline import create_pipeline

# Load test queries
with open('test_queries.json', 'r') as f:
    test_data = json.load(f)

# Create pipeline
pipeline = create_pipeline('llama_parse')

# Process evaluation documents
pipeline.process_directory('evaluation_docs')

# Run evaluation
results = pipeline.evaluate(
    test_data['queries'],
    output_file='evaluation_results.json'
)

# Print summary
print(f"Average correctness score: {results['metrics']['correctness']['average']}")
print(f"Faithfulness score: {results['metrics']['faithfulness']['average']}")
print(f"Contextual relevancy: {results['metrics']['contextual_relevancy']['average']}")
```

## Parser Comparison

Each parser has its strengths:

- **Gemini Flash**:
  - Best for documents with complex layouts
  - Excellent at handling images and diagrams
  - Requires Google API key

- **Llama Parse**:
  - Strong structure preservation
  - Good table extraction
  - Requires Llama Cloud API key

- **PyMuPDF4LLM**:
  - Fast and lightweight
  - No API key required
  - Best for simple text-based PDFs

## Project Structure

```
pdf-knowledge-graph/
├── src/
│   ├── pdf_parsers.py      # PDF parsing implementations
│   ├── knowledge_graph.py  # Knowledge graph builder
│   └── pipeline.py         # Main pipeline
├── tests/
│   └── test_queries.json   # Evaluation test cases
├── data/
│   └── pdfs/              # PDF documents
├── graphs/
│   └── saved/             # Saved knowledge graphs
├── requirements.txt
├── .env
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph team for the graph-based LLM framework
- Anthropic for Claude API
- Google for Gemini API
- Llama Parse team for PDF parsing capabilities 