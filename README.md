# PDF Knowledge Graph Builder 🚀

Transform your PDF documents into dynamic, queryable knowledge graphs with our powerful, multi-strategy toolkit! This project leverages advanced PDF parsing and LLM-powered graph construction to deliver rich semantic analysis, natural language querying, and comprehensive evaluation. 🔍✨

## Overview

The PDF Knowledge Graph Builder enables you to:
- **Parse PDFs** using multiple strategies (PyMuPDF4LLM, Gemini Flash, Llama Parse) 📄
- **Construct Knowledge Graphs** with entities, relationships, and automatic URI generation 🌐
- **Query Graphs** using natural language for deep insights 🗣️
- **Evaluate & Compare** parsing methods with multi-metric assessments 📊

## Key Features

- **Diverse PDF Parsing:**  
  - **PyMuPDF4LLM:** Fast & lightweight; ideal for simple text PDFs.  
  - **Gemini Flash:** Excels with complex layouts and visual content (requires Google API key).  
  - **Llama Parse:** Preserves structure and excels at table extraction (requires Llama Cloud API key).

- **Robust Knowledge Graphs:**  
  - Uses GPT-4 for entity & relationship extraction.  
  - Generates valid URIs and supports both NetworkX and RDF formats.

- **Natural Language Querying & Evaluation:**  
  - Interact with your graphs using everyday language.  
  - Comprehensive evaluations on parsing accuracy, entity extraction, relationship mapping, and query response quality.

## Installation 🛠️

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-knowledge-graph.git
   cd pdf-knowledge-graph
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a .env file in the project root and add:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   LLAMA_CLOUD_API_KEY=your_llama_parse_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Command Line Interface

**Process a Single PDF:**
```bash
python main.py --input path/to/your/document.pdf
```

**Process a Directory of PDFs:**
```bash
python main.py --input path/to/pdf/directory
```

**Evaluation-Only Mode:**
```bash
python main.py --eval-only
```

The script automatically creates necessary directories, processes PDFs with all parsers, builds knowledge graphs, runs evaluations, and saves the results. ✅

### Python API

Integrate the pipeline directly into your Python code:

```python
from src.pipeline import create_pipeline

# Initialize the pipeline with a specific parser
pipeline = create_pipeline(parser_type='pymupdf4llm')

# Process a PDF document
results = pipeline.process_pdf('path/to/document.pdf')

# Save the generated knowledge graph
pipeline.save_knowledge_graph('output/graphs')

# Query the graph using natural language
answer = pipeline.query_knowledge_graph("What are the main topics discussed?")
```

## Directory Structure

```
pdf-knowledge-graph/
├── data/
│   ├── pdfs/              # Input PDF files 📄
│   └── evaluations/       # Evaluation queries (JSON)
├── graphs/
│   ├── pymupdf4llm/       # Graphs from PyMuPDF4LLM parser
│   ├── gemini_flash/      # Graphs from Gemini Flash parser
│   └── llama_parse/       # Graphs from Llama Parse parser
├── output/
│   ├── markdown/          # Parsed content in Markdown format
│   └── comparison_results_{timestamp}.json
└── evaluation/
    ├── pymupdf4llm_eval_{timestamp}.json
    ├── gemini_flash_eval_{timestamp}.json
    ├── llama_parse_eval_{timestamp}.json
    └── evaluation_comparison_{timestamp}.json
```

## Evaluation Metrics 📊

Our evaluation framework measures:

- Parsing Accuracy: How well the PDF content is extracted.
- Entity Extraction: Precision in identifying key entities.
- Relationship Mapping: Accuracy in connecting entities.
- Query Response Quality: Effectiveness of natural language answers.
- Processing Efficiency: Speed and resource usage across parsers.

## Contributing 🤝

We welcome contributions! To get started:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to your branch.
5. Submit a pull request.

For development:
```bash
pip install -r requirements-dev.txt  # Install dev dependencies
pytest tests/                        # Run tests
flake8 src/ tests/                  # Lint the code
```

## License

Licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments 🙏

- LangGraph: For the graph-based LLM framework.
- Anthropic: For the Claude API integration.
- Google: For powering the Gemini API.
- Llama Parse: For advanced PDF parsing capabilities.

## Support

For issues, feature requests, or usage questions, please open an issue on GitHub or join our Discord community.