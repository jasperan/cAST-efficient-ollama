# Summary: Code Vectorization for RAG with cAST vs Random Chunking (Ollama Version)

## Executive Summary

This comprehensive Cursor prompt provides a complete project specification for building a demonstration system that compares **random chunking** versus **AST-based chunking (cAST)** for code retrieval in RAG-powered coding assistants. 

**Key Change: Metadata enrichment now uses Ollama API endpoint instead of OpenAI, meaning NO API KEYS ARE REQUIRED.**

## Problem Statement

When storing Python code in vector databases for retrieval-augmented generation (RAG) systems:
- **Random chunking** breaks code at arbitrary character boundaries, often splitting functions mid-declaration
- **cAST (Abstract Syntax Tree) chunking** respects code structure, keeping functions and classes intact
- Result: cAST provides 20%+ better retrieval accuracy and semantic coherence

## What This Project Demonstrates

### 1. Code Chunking Comparison

**Random Chunking Issues:**
```python
# Chunk 1 ends here (500 chars)
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]

# Chunk 2 starts mid-regex
+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```
Problem: Function definition is fragmented across chunks—impossible to understand in isolation.

**AST-Based Chunking (cAST):**
```python
# Complete Chunk 1
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```
Benefit: Complete function preserved with full context.

### 2. Advanced Retrieval Pipeline

The project implements a multi-stage retrieval pipeline:

```
Query → Embed → Initial Retrieval (150) → Rerank (20) → Results
```

1. **Embedding**: CodeBERT transforms code into 384-dimensional vectors
2. **Initial Retrieval**: Vector similarity search returns top-150 candidates
3. **Reranking**: BGE Reranker scores these 150 and selects top-20
4. **Improvement**: Reranking typically improves relevance by 15-20%

### 3. Metadata Enrichment via Ollama (NO API KEYS!)

Uses **local Ollama API** to extract semantic information via structured outputs:

```python
{
    'chunk_type': 'function',
    'function_name': 'validate_email',
    'purpose': 'Validates email format using regex pattern',
    'input_params': 'email: str - email address to validate',
    'return_type': 'bool - True if valid email format',
    'dependencies': ['re'],
    'complexity': 'low'
}
```

**Ollama Benefits:**
- ✓ No API keys required
- ✓ Runs entirely locally (privacy preserved)
- ✓ Zero cost after model download
- ✓ Offline capability
- ✓ Configurable model selection
- ✓ Structured outputs with Pydantic schemas

### 4. Oracle Database 26ai Integration

Stores all chunks with vectors in Oracle's native vector column:

```sql
CREATE TABLE code_chunks (
    chunk_id VARCHAR2(255) PRIMARY KEY,
    chunk_content CLOB,
    embedding VECTOR(384, FLOAT32),  -- CodeBERT embeddings
    chunking_method VARCHAR2(20),     -- 'random' or 'ast'
    metadata_json CLOB,               -- Ollama-extracted metadata
    ...
);
```

Performance: Vector similarity search with indexes for sub-100ms retrieval on 10K+ chunks.

## Project Components

### Core Modules

1. **Chunking** (`chunking/`)
   - `random_chunker.py`: Fixed-size splitting with 10% overlap
   - `ast_chunker.py`: tree-sitter based AST extraction with recursive split-then-merge

2. **Embedding** (`embedding/`)
   - `embedder.py`: CodeBERT vectorization (384 dimensions)
   - `reranker.py`: BGE reranker for top-20 selection from top-150

3. **Storage** (`oracle_db/`)
   - `connection.py`: Connection pooling and lifecycle management
   - `schema.py`: Table creation with vector indexes
   - `operations.py`: CRUD operations for chunk storage/retrieval

4. **Intelligence** (`metadata/`)
   - `extractor.py`: **Ollama-based metadata extraction with structured outputs** (Pydantic models)
   - No OpenAI dependency, no API keys needed
   - Structured output support for guaranteed JSON schema

5. **Retrieval** (`retrieval/`)
   - `search.py`: Multi-stage pipeline orchestration

6. **Analysis** (`comparison/`)
   - `analyzer.py`: Comparative metrics computation
   - `reporter.py`: Console and export report generation

### Project Structure

```
code-rag-ast-demo/
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── .env.example                 # Example configuration
├── chunking/                    # Random vs AST implementations
├── embedding/                   # CodeBERT + BGE reranker
├── oracle_db/                   # Connection, schema, operations
├── metadata/                    # Ollama extraction
├── retrieval/                   # Multi-stage pipeline
├── comparison/                  # Analysis & reporting
└── examples/sample.py          # Representative test code
```

### Expected Results

Based on research, AST chunking demonstrates:
- **20%+ better retrieval accuracy** vs random
- **4.3-point Recall@5 improvement**
- **2.67-point generation Pass@1 improvement**
- **73% fewer chunks** to search (12 vs 45 for typical code)
- **15-20% improvement** from reranking stage

## Key Features of Ollama Integration

### 1. Structured Outputs with Pydantic

```python
from pydantic import BaseModel
from ollama import chat

class CodeMetadata(BaseModel):
    purpose: str
    input_params: str
    return_type: str
    docstring: Optional[str]
    dependencies: List[str]
    complexity: str

# Ollama will constrain output to match schema exactly
response = chat(
    model="llama2",
    messages=[{"role": "user", "content": "..."}],
    format=CodeMetadata.model_json_schema(),  # Enforce schema
    stream=False
)

metadata = CodeMetadata.model_validate_json(response.message.content)
```

### 2. Zero Configuration for API Keys

Create `.env` file:
```
ORACLE_USER=scott
ORACLE_PASSWORD=yourpassword
ORACLE_DSN=localhost:1521/freepdb1
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
```

No OpenAI API key needed!

### 3. Model Selection

Supports any Ollama model. Recommended options:

| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| **llama2** | Fast | Good | 4K | Default, balanced |
| **llama3.1** | Medium | Excellent | 8K | Better quality |
| **mistral** | Very Fast | Good | 8K | Speed priority |
| **neural-chat** | Fast | Good | 4K | Instruction following |

Download: `ollama pull llama2`

### 4. Ollama Setup

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a model
ollama pull llama2

# 3. Start server (runs on localhost:11434)
ollama serve

# 4. In another terminal, run the project
python main.py --action vectorize ...
```

## CLI Usage Examples

### Setup
```bash
python main.py --action setup \
  --db-user scott \
  --db-password password \
  --db-dsn localhost:1521/freepdb1
```

### Vectorize with Metadata Enrichment (via Ollama)
```bash
python main.py --action vectorize \
  --chunking-method both \
  --sample-file examples/sample.py \
  --enrich-metadata true \
  --ollama-endpoint http://localhost:11434 \
  --ollama-model llama2
```

### Search and Compare
```bash
python main.py --action search \
  --query "find data validation" \
  --top-k 20 \
  --rerank true
```

## Comparison Output Example

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CHUNKING STRATEGY COMPARISON REPORT                │
└─────────────────────────────────────────────────────────────────────┘

Query: "Find code that validates email"

RANDOM CHUNKING:
├─ Chunks: 45 | Avg Size: 487 chars
├─ Top-1 Accuracy: ✗ (Incorrect, score: 0.72)
├─ Top-5 Hit Rate: 40%
└─ Issue: Function split across 3 chunks

CASTS (AST-Based):
├─ Chunks: 12 | Avg Size: 1852 chars
├─ Top-1 Accuracy: ✓ (Correct, score: 0.94)
├─ Top-5 Hit Rate: 95%
└─ Improvement: +138% accuracy, -73% chunks to search

RERANKING IMPACT:
├─ Initial (top-150): 0.78 avg score
├─ After Rerank (top-20): 0.91 avg score
└─ Improvement: +17%

METADATA EXTRACTION (via Ollama):
├─ Model Used: llama2
├─ Chunks Processed: 12
├─ Avg Extraction Time: 0.34s per chunk
├─ Success Rate: 100%
└─ Cached Results: 8/12 (33% reduction on repeat queries)
```

## Technical Advantages of Ollama Approach

### vs. OpenAI API
- **No API keys**: Paste `.env` without credentials
- **No costs**: Free after model download (~4GB for llama2)
- **Privacy**: All data stays local
- **Offline**: Works without internet
- **Latency**: Faster for small batches
- **Control**: Choose model and parameters

### vs. Running Your Own LLM Server
- **Simplicity**: Just `ollama pull model && ollama serve`
- **Management**: Ollama handles all infrastructure
- **Compatibility**: Works on Mac, Linux, Windows
- **Structured Outputs**: First-class support with Pydantic

## Architecture Comparison

```
OPENAI API VERSION:
┌──────────────┐    HTTPS     ┌──────────────┐
│   Project    │──────────────│  OpenAI API  │
│  (Python)    │◄─────────────│ (Cloud)      │
└──────────────┘              └──────────────┘
• Requires API key
• Network dependency
• Per-token costs
• Response latency

OLLAMA VERSION (This Project):
┌──────────────┐    Local     ┌──────────────┐
│   Project    │──────────────│ Ollama       │
│  (Python)    │◄─────────────│ (localhost)  │
└──────────────┘              └──────────────┘
• No API key needed
• Fully offline
• Zero cost
• Sub-second responses
• Privacy-first
```

## How to Use This Project

### Step 1: Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### Step 2: Download Model
```bash
ollama pull llama2
```

### Step 3: Start Ollama Server
```bash
ollama serve
# Runs on http://localhost:11434
```

### Step 4: Use Cursor Prompt
1. Download `cursor_prompt_ollama.md` [125]
2. Paste entire content into **Cursor IDE**
3. Cursor generates full project
4. Configure `.env` with Oracle credentials
5. Run commands:
   ```bash
   python main.py --action setup ...
   python main.py --action vectorize ...
   python main.py --action search ...
   ```

## Key Differences from Original Prompt

| Aspect | Original | Ollama Version |
|--------|----------|---|
| **Metadata LLM** | OpenAI API | Local Ollama |
| **API Key** | Required | Not needed |
| **Cost** | Per-token fees | Free |
| **Privacy** | Cloud-based | 100% local |
| **Latency** | Network dependent | Sub-second |
| **Model Selection** | Limited | Any Ollama model |
| **Structured Output** | JSON mode | Pydantic schemas |
| **Offline Support** | No | Yes |

## Success Criteria

✓ Random chunking breaks code at arbitrary boundaries
✓ AST chunking preserves complete functions/classes
✓ **Ollama-based metadata extraction works without API keys**
✓ Oracle DB stores and retrieves vectors correctly
✓ Reranking improves top-20 relevance by 15%+
✓ AST method shows 20%+ higher accuracy than random
✓ Clear, actionable comparison reports generated
✓ All features accessible via intuitive CLI
✓ Code is production-quality with error handling
✓ Documentation is comprehensive and clear
✓ **Project works entirely locally (Ollama + Oracle)**
✓ **Zero external API dependencies**

## Next Steps

1. Download the updated **cursor_prompt_ollama.md** [125]
2. Install Ollama locally
3. Pull a model (`ollama pull llama2`)
4. Start Ollama server
5. Paste prompt into Cursor
6. Let Cursor generate project
7. Configure `.env` with Oracle credentials
8. Run the three main commands
9. Compare AST vs random chunking results with local LLM enrichment

This version provides the same powerful functionality but entirely privately and cost-free using local Ollama inference.