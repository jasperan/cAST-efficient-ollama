# Comprehensive RAG Coding Assistant Project - AST vs Random Chunking Demonstration (Ollama Version)

## Overview

This project demonstrates the differences between random chunk-based RAG and cAST (Abstract Syntax Tree) based chunking for code retrieval using Python. It uses Oracle Database 26ai for vector storage, tree-sitter for AST parsing, CodeBERT for embeddings, BGE for reranking, and Ollama for local metadata enrichment.

## Technical Stack

- Python 3.10+
- Oracle Database 26ai with python-oracledb
- tree-sitter for AST
- sentence-transformers with microsoft/codebert-base
- FlagEmbedding for reranking
- Ollama for metadata
- argparse for CLI
- tabulate, matplotlib, pandas for visualization

## Installation

<!-- one-command-install -->
> **One-command install** — clone, configure, and run in a single step:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/jasperan/cAST-efficient-ollama/main/install.sh | bash
> ```
>
> <details><summary>Advanced options</summary>
>
> Override install location:
> ```bash
> PROJECT_DIR=/opt/myapp curl -fsSL https://raw.githubusercontent.com/jasperan/cAST-efficient-ollama/main/install.sh | bash
> ```
>
> Or install manually:
> ```bash
> git clone https://github.com/jasperan/cAST-efficient-ollama.git
> cd cAST-efficient-ollama
> # See below for setup instructions
> ```
> </details>


1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure the application:
   - Copy `config.yaml.example` to `config.yaml` and edit the values (Oracle credentials, Ollama settings, etc.).
   - Alternatively, set environment variables as in `.env.example` (deprecated but supported as fallback).
4. Install Ollama:
   - Download from https://ollama.ai or use `curl -fsSL https://ollama.ai/install.sh | sh` (Linux/macOS).
   - Pull a model: `ollama pull llama2`
   - Start the server: `ollama serve`
5. Ensure Oracle Database 26ai is running and accessible.

## Usage

CLI arguments override config.yaml values.

### Setup Database

```bash
python main.py --action setup
```
(Overrides from config.yaml will be used.)

### Vectorize Sample Code

```bash
python main.py --action vectorize --chunking-method both --sample-file examples/sample.py --enrich-metadata true
```

### Search and Compare

```bash
python main.py --action search --query "data validation" --top-k 20 --rerank true
```

## Testing

Run unit tests:

```bash
python -m unittest discover tests -v
```

Tests cover chunking, embedding, retrieval, and metadata extraction.

## Configuration

Edit `config.yaml` for settings. Example structure:

```yaml
oracle:
  user: your_oracle_user
  password: your_password
  dsn: your_dsn

ollama:
  endpoint: http://localhost:11434
  model: llama2

app:
  verbose: true
  chunk_size: 2000
  overlap_percentage: 10
  top_k_retrieval: 150
  top_k_rerank: 20
```

For Ollama, ensure the server is running and the model is pulled.

## Troubleshooting

- If metadata extraction fails, check Ollama server and model.
- For DB errors, verify Oracle connection details in config.yaml.
- Tests may require mocking for external services.

For more details, see `project_summary_ollama.md`.

## Project Structure

- `src/cast_ollama/`: Main package directory
  - `cli.py`: Entry point for main application
  - `chunking/`: Random and AST chunkers
  - `embedding/`: Embedder and reranker
  - `oracle_db/`: Oracle DB operations (Primary)
  - `chroma_db/`: ChromaDB operations (Fallback)
  - `metadata/`: Ollama extractor
  - `retrieval/`: Search pipeline
  - `comparison/`: Analyzer and reporter
- `demo_paper.py`: Standalone demo comparison script
- `examples/`: Sample code
- `tests/`: Unit tests

## Running Tests

python -m unittest discover tests
```

## Running the Retrieval Demo

This repository includes a standalone demo script `demo_paper.py` that replicates the methodology comparing Random Chunking vs cAST (Abstract Syntax Tree) chunking.

### Usage

```bash
python demo_paper.py
```

### Observed Results

Running the demo on a synthetic dataset yields the following comparison (using locally embedded vectors):

**Random Chunking:**
- **Accuracy**: ~40% (Top-5)
- **Issues**: Functions are frequently split across arbitrary character boundaries (e.g., regex patterns cut in half), leading to poor retrieval context.

**cAST Chunking:**
- **Accuracy**: ~95% (Top-5)
- **Advantages**: Preserves complete function and class definitions, ensuring the retrieved code is syntactically valid and semantically complete.

**Metrics:**
- **Accuracy Improvement**: +55% (cAST over Random)
- **Chunk Reduction**: ~60% fewer chunks generated (more efficient index)

*Note: If an Oracle Database connection is not available, the system automatically falls back to using ChromaDB for vector storage.*

## Ollama Setup

Ensure Ollama is running on the specified endpoint. No API keys required.

For more details, see the project summary in project_summary_ollama.md.
