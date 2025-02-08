# PDF Knowledge Graph Builder

A powerful Python toolkit that transforms PDF documents into queryable knowledge graphs using multiple parsing strategies and LLM-powered graph construction.

## Overview

This project provides a flexible pipeline for:
1. Parsing PDFs using different strategies (PyMuPDF4LLM, Gemini Flash, Llama Parse)
2. Constructing knowledge graphs from the parsed content
3. Querying the graphs using natural language
4. Evaluating and comparing different parsing approaches

## Features

### Multiple PDF Parsing Strategies

- **PyMuPDF4LLM**
  - Fast and lightweight
  - Best for simple text-based PDFs
  - No API key required
  - Limited layout understanding

- **Gemini Flash**
  - Excellent for complex layouts
  - Strong image and diagram handling
  - Requires Google API key
  - Higher accuracy for visual content

- **Llama Parse**
  - Strong structure preservation
  - Excellent table extraction
  - Requires Llama Cloud API key
  - Best for technical documents

### Knowledge Graph Construction

- Entity extraction using GPT-4
- Relationship identification
- Automatic URI generation
- RDF format support
- NetworkX graph structure

### Query & Analysis

- Natural language querying
- Multi-metric evaluation
- Parser comparison tools
- Performance analytics

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

### Command Line Interface

Run the main script to process PDFs with all available parsers:

```bash
# Process a single PDF
python main.py path/to/your/document.pdf

# Process all PDFs in a directory
python main.py path/to/pdf/directory
```

The script will:
1. Create necessary output directories
2. Process the PDF(s) with each parser
3. Generate knowledge graphs
4. Run evaluations
5. Save comparison results

### Python API

```python
from src.pipeline import create_pipeline

# Initialize pipeline with specific parser
pipeline = create_pipeline(parser_type='pymupdf4llm')

# Process a PDF
results = pipeline.process_pdf('path/to/document.pdf')

# Save the knowledge graph
pipeline.save_knowledge_graph('output/graphs')

# Query the graph
answer = pipeline.query("What are the main topics discussed?")
```

## Output Structure

```
pdf-knowledge-graph/
├── data/
│   ├── pdfs/              # Input PDFs
│   └── evaluations/       # Evaluation queries
├── graphs/
│   ├── pymupdf4llm/      # Knowledge graphs by parser
│   ├── gemini_flash/
│   └── llama_parse/
├── output/
│   ├── markdown/         # Parsed content in markdown
│   └── comparison_results_{timestamp}.json
└── evaluation/
    ├── pymupdf4llm_eval_{timestamp}.json
    ├── gemini_flash_eval_{timestamp}.json
    ├── llama_parse_eval_{timestamp}.json
    └── evaluation_comparison_{timestamp}.json
```

## Evaluation Metrics

The evaluation framework assesses:
- Parsing accuracy
- Entity extraction quality
- Relationship identification
- Query response accuracy
- Processing speed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/ tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangGraph team for the graph-based LLM framework
- Anthropic for Claude API
- Google for Gemini API
- Llama Parse team for PDF parsing capabilities

## Support

For issues and feature requests, please use the GitHub issue tracker.

For usage questions, join our [Discord community](https://discord.gg/yourinvitelink). 