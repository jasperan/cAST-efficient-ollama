# Comprehensive RAG Coding Assistant Project - AST vs Random Chunking Demonstration

You are an expert Python developer. Create a complete, production-ready project that demonstrates the difference between random chunk-based RAG and cAST (Abstract Syntax Tree) based chunking for code retrieval. This project will be used by Oracle developers to understand optimal code vectorization strategies for coding assistants and LLM agents.

## Project Overview

Build a Python application that:
1. Implements both random chunking and AST-based chunking (cAST) methods
2. Stores code chunks in Oracle Database 26ai with vector embeddings
3. Performs semantic similarity search with cAST retrieval pipeline including reranking
4. Enriches metadata using Ollama API for local LLM inference (no API keys required)
5. Provides CLI interface with argparse for easy comparison
6. Visualizes the difference between both approaches with test queries

## Technical Stack

- **Language**: Python 3.10+
- **Vector Storage**: Oracle Database 26ai with python-oracledb
- **Code Parsing**: tree-sitter (for AST extraction)
- **Embeddings**: sentence-transformers with CodeBERT model (microsoft/codebert-base)
- **Reranking**: BAAI/bge-reranker-v2-m3 from FlagEmbedding
- **Local LLM Metadata Extraction**: Ollama API endpoint (no API keys needed, runs locally)
- **CLI**: argparse
- **Visualization**: pandas, tabulate for console output, matplotlib for charts

## Key Features to Implement

### 1. Chunking Strategies

#### Random Chunking
- Split code into fixed-size chunks (e.g., 500 characters)
- Ignore code structure completely
- Add random 10% overlap between chunks
- Track chunk boundaries without semantic awareness

#### AST-Based Chunking (cAST)
- Use tree-sitter to parse Python code into Abstract Syntax Tree
- Extract complete syntactic units (functions, classes, methods)
- Chunk size limit: 2000 non-whitespace characters
- Implement recursive split-then-merge algorithm:
  - Keep complete AST nodes intact when they fit
  - When nodes exceed limits, recursively break into child nodes
  - Greedily merge small sibling nodes
  - Add 10% overlap between chunks
- Preserve semantic boundaries

### 2. Metadata Enrichment via Ollama

Implement local LLM-based metadata extraction using Ollama API with structured outputs:

```python
metadata_fields = {
    'file_path': str,
    'chunk_id': str,
    'chunk_type': str,  # 'function', 'class', 'method', 'import_block'
    'parent_class': Optional[str],
    'function_name': Optional[str],
    'start_line': int,
    'end_line': int,
    'docstring': Optional[str],  # Ollama-extracted via structured output
    'purpose': str,  # Ollama-extracted
    'input_params': str,  # Ollama-extracted
    'return_type': str,  # Ollama-extracted
    'dependencies': List[str],  # Functions/classes it calls
    'complexity': str,  # 'low', 'medium', 'high'
    'has_tests': bool,
    'language': str,  # 'python'
}
```

Use Ollama with structured outputs (Pydantic models) to extract semantic metadata without any API keys.

**Ollama Setup Requirements:**
- Ollama running locally on `http://localhost:11434` (configurable via environment variable `OLLAMA_ENDPOINT`)
- Recommended models: `llama2`, `llama3.1`, `mistral`, or `neural-chat` (smaller models are faster)
- Structured outputs supported on latest Ollama version (v0.1.0+)

### 3. Oracle Database 26ai Integration

- Create table schema for storing code chunks with vectors:
  ```sql
  CREATE TABLE code_chunks (
    chunk_id VARCHAR2(255) PRIMARY KEY,
    file_path VARCHAR2(512),
    chunk_content CLOB,
    chunk_type VARCHAR2(50),
    start_line NUMBER,
    end_line NUMBER,
    function_name VARCHAR2(255),
    parent_class VARCHAR2(255),
    docstring VARCHAR2(2000),
    purpose VARCHAR2(2000),
    dependencies CLOB,  -- JSON array
    complexity VARCHAR2(20),
    embedding VECTOR(384, FLOAT32),
    chunking_method VARCHAR2(20),  -- 'random' or 'ast'
    created_at TIMESTAMP DEFAULT SYSDATE,
    metadata_json CLOB  -- Full metadata as JSON
  );
  ```

- Use python-oracledb to connect and store vectors
- Create vector indexes on both random and AST chunks
- Implement functions for insert, query, and retrieval

### 4. Vector Operations

- **Embedding Model**: microsoft/codebert-base (384-dimensional)
- Generate embeddings for each chunk using sentence-transformers
- Store embeddings in Oracle VECTOR column (FLOAT32)
- Implement cosine similarity search

### 5. Retrieval Pipeline with Reranking

Implement a sophisticated retrieval pipeline:

1. **Initial Retrieval**: 
   - Query the code chunk vector store
   - Retrieve top-150 chunks using cosine similarity
   - Track and store intermediate results

2. **Reranking**:
   - Use BAAI/bge-reranker-v2-m3 (FlagEmbedding)
   - Rerank top-150 results
   - Select top-20 most relevant chunks
   - Return scores and rankings

3. **Post-Processing**:
   - Remove duplicates
   - Add context expansion (retrieve parent chunks if child was matched)
   - Format results with metadata

### 6. Comprehensive Testing Framework

Create test queries that demonstrate advantages of AST chunking:

```python
test_queries = [
    # Query 1: Looking for data validation
    "Find code that validates email addresses and user input",
    
    # Query 2: Looking for error handling
    "Show me functions that handle database connection errors",
    
    # Query 3: Looking for a specific algorithm
    "Find the sorting or filtering algorithm implementation",
    
    # Query 4: Looking for API interaction
    "Where is code that makes HTTP requests to external APIs?",
    
    # Query 5: Context-dependent query
    "Find helper functions that process strings and format output"
]
```

For each query, perform retrieval using both methods and compare:
- Retrieval accuracy (is correct chunk in top-5?)
- Semantic relevance scores
- Chunk completeness (is function code intact or fragmented?)
- Processing time

### 7. CLI Interface (argparse)

```bash
python main.py \
  --action [setup|vectorize|search] \
  --chunking-method [random|ast|both] \
  --sample-file path/to/code.py \
  --query "search query text" \
  --top-k 20 \
  --db-user username \
  --db-password password \
  --db-dsn hostname:port/service \
  --ollama-endpoint http://localhost:11434 \
  --ollama-model llama2 \
  --enrich-metadata [true|false] \
  --rerank [true|false] \
  --verbose
```

Examples:
```bash
# Setup Oracle tables
python main.py --action setup --db-user scott --db-password password --db-dsn localhost:1521/freepdb1

# Vectorize with both methods and metadata enrichment via Ollama
python main.py --action vectorize --chunking-method both --sample-file examples/sample.py \
  --enrich-metadata true --ollama-endpoint http://localhost:11434 --ollama-model llama2

# Search and compare
python main.py --action search --query "data validation" --top-k 20 --rerank true
```

### 8. Comparison Output

Generate detailed comparison reports:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CHUNKING STRATEGY COMPARISON REPORT                │
└─────────────────────────────────────────────────────────────────────┘

Query: "Find code that validates email"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RANDOM CHUNKING RESULTS:
├─ Chunks Generated: 45
├─ Avg Chunk Size: 487 chars
├─ Processing Time: 1.23s
├─ Top-1 Result: ✗ Incorrect (score: 0.72)
├─ Top-5 Accuracy: 40%
└─ Issues: 
   • Function split across chunks
   • Missing context for complete understanding

CASTS (AST-BASED) RESULTS:
├─ Chunks Generated: 12
├─ Avg Chunk Size: 1852 chars
├─ Processing Time: 0.89s (with reranking)
├─ Top-1 Result: ✓ Correct (score: 0.94)
├─ Top-5 Accuracy: 95%
└─ Advantages:
   • Complete functions preserved
   • Better semantic relevance
   • Reduced chunks to search

RERANKING IMPACT (cAST only):
├─ Initial Retrieval (top-150): 0.78 avg score
├─ After Reranking (top-20): 0.91 avg score
├─ Improvement: +17.9%
└─ Processing Time: +0.12s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9. Sample Python File

Create a comprehensive example file (examples/sample.py) with:
- Multiple functions with different purposes
- Classes with methods
- Error handling
- Database operations
- API interactions
- String processing utilities
- Data validation functions

Example structure:
```python
"""
Sample code file for vectorization demonstration.
Contains various code patterns for testing retrieval.
"""

import requests
import re
from datetime import datetime

class DataValidator:
    """Validates user input data."""
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_username(self, username: str) -> bool:
        """Validate username length and characters."""
        return len(username) >= 3 and username.isalnum()


class DatabaseManager:
    """Manages database connections."""
    
    def connect_with_retry(self, max_retries: int = 3) -> bool:
        """Connect to database with retry logic."""
        for attempt in range(max_retries):
            try:
                # Connection logic here
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                continue
        return False


def process_api_response(response: dict) -> list:
    """Process API response and extract data."""
    try:
        data = response.get('data', [])
        return [item for item in data if item.get('valid')]
    except KeyError:
        raise ValueError("Invalid response format")


# ... more functions and classes ...
```

## Project Structure

```
code-rag-ast-demo/
├── main.py                 # Entry point with argparse CLI
├── requirements.txt        # Dependencies
├── config.py              # Configuration settings
├── .env.example           # Example environment variables
├── chunking/
│   ├── __init__.py
│   ├── random_chunker.py  # Random chunking implementation
│   └── ast_chunker.py     # cAST implementation with tree-sitter
├── embedding/
│   ├── __init__.py
│   ├── embedder.py        # CodeBERT embeddings
│   └── reranker.py        # BGE reranker implementation
├── oracle_db/
│   ├── __init__.py
│   ├── connection.py      # Oracle DB connection management
│   ├── schema.py          # Table creation and schema
│   └── operations.py      # CRUD operations for chunks
├── metadata/
│   ├── __init__.py
│   └── extractor.py       # Ollama-based metadata extraction with structured outputs
├── retrieval/
│   ├── __init__.py
│   └── search.py          # Retrieval and reranking pipeline
├── comparison/
│   ├── __init__.py
│   ├── analyzer.py        # Compare random vs AST results
│   └── reporter.py        # Generate comparison reports
├── examples/
│   └── sample.py          # Sample code file for testing
├── tests/
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   └── test_ollama_metadata.py
└── README.md              # Comprehensive documentation
```

## Requirements.txt

```
python-oracledb>=2.0.0
sentence-transformers>=2.2.0
torch>=2.0.0
tree-sitter>=0.20.0
FlagEmbedding>=1.5.0
ollama>=0.1.0
pydantic>=2.0.0
click>=8.0.0
tabulate>=0.9.0
matplotlib>=3.5.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.60.0
requests>=2.28.0
python-dotenv>=1.0.0
```

## .env.example

```
# Oracle Database Configuration
ORACLE_USER=scott
ORACLE_PASSWORD=password
ORACLE_DSN=localhost:1521/freepdb1

# Ollama Configuration
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2

# Application Settings
VERBOSE=false
CHUNK_SIZE=2000
OVERLAP_PERCENTAGE=10
TOP_K_RETRIEVAL=150
TOP_K_RERANK=20
```

## Detailed Implementation Requirements

### A. Chunking Module

**random_chunker.py**:
- `RandomChunker` class with `chunk(code: str) -> List[Dict]`
- Split at fixed character boundaries (500 chars)
- Add 10% overlap between chunks
- Return chunk with metadata: content, start_line, end_line, char_positions
- Track which characters belong to which chunk

**ast_chunker.py**:
- `ASTChunker` class with `chunk(code: str) -> List[Dict]`
- Use tree-sitter for Python parsing
- Implement recursive split-then-merge algorithm
- Extract node types (FunctionDef, ClassDef, etc.)
- Maintain 10% overlap using duplicate final lines
- Return: content, node_type, start_line, end_line, parent_nodes
- Calculate non-whitespace character count for sizing

### B. Embedding Module

**embedder.py**:
- `CodeEmbedder` class using microsoft/codebert-base
- `embed(text: str) -> np.ndarray` returns 384-dim vector
- Batch embedding support for efficiency
- Handle truncation for long chunks

**reranker.py**:
- `CodeReranker` class using BAAI/bge-reranker-v2-m3
- `rerank(query: str, candidates: List[str], top_k: int) -> List[Tuple[int, float]]`
- Return indices and scores of top-k reranked results
- Normalize scores to 0-1 range

### C. Oracle DB Module

**connection.py**:
- Manage connection pooling
- Singleton pattern for connection
- Error handling and reconnection logic
- Environment variable support for credentials

**schema.py**:
- `create_tables()` function to initialize schema
- Create code_chunks table with vector column
- Create indexes for performance
- Implement vector similarity indexing

**operations.py**:
- `insert_chunk(chunk_data, embedding, metadata)` - Store in DB
- `search_by_vector(query_embedding, limit=150)` - Vector similarity search
- `get_chunk_by_id(chunk_id)` - Retrieve specific chunk
- `delete_all_chunks()` - Clear data
- Bulk insert for efficiency

### D. Metadata Extraction via Ollama

**extractor.py**:
- `OllamaMetadataExtractor` class using local Ollama API
- Use Pydantic models for structured outputs
- Extract: purpose, input_params, return_type, docstring
- Handle parsing errors gracefully
- Implement retry logic for Ollama connectivity
- Cache extracted metadata to avoid redundant API calls
- Support configurable model and endpoint

```python
from pydantic import BaseModel
from typing import Optional, List
from ollama import chat

class CodeMetadata(BaseModel):
    purpose: str
    input_params: str
    return_type: str
    docstring: Optional[str]
    dependencies: List[str]
    complexity: str  # 'low', 'medium', 'high'

class OllamaMetadataExtractor:
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "llama2"):
        self.endpoint = endpoint
        self.model = model
        self.cache = {}
    
    def extract(self, code_chunk: str) -> CodeMetadata:
        """
        Extract metadata from code chunk using Ollama with structured outputs.
        Uses Pydantic schema for guaranteed JSON structure.
        """
        # Check cache first
        chunk_hash = hash(code_chunk)
        if chunk_hash in self.cache:
            return self.cache[chunk_hash]
        
        prompt = f"""
        Analyze this Python code and extract metadata in JSON format.
        
        Code:
        {code_chunk}
        
        Extract:
        1. purpose: One-line description of what this code does
        2. input_params: Description of input parameters
        3. return_type: What this function/class returns
        4. docstring: A clean docstring (optional)
        5. dependencies: List of imported modules or called functions
        6. complexity: 'low', 'medium', or 'high' based on code complexity
        
        Return only valid JSON matching the schema.
        """
        
        try:
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=CodeMetadata.model_json_schema(),
                stream=False
            )
            
            metadata = CodeMetadata.model_validate_json(response.message.content)
            self.cache[chunk_hash] = metadata
            return metadata
            
        except Exception as e:
            # Return defaults on failure
            return CodeMetadata(
                purpose="Unable to extract - check Ollama connection",
                input_params="",
                return_type="unknown",
                docstring=None,
                dependencies=[],
                complexity="medium"
            )
```

**Key differences from OpenAI approach:**
- No API keys required
- Runs entirely locally
- Uses Ollama's structured outputs with Pydantic models
- Configurable model selection (llama2, llama3.1, mistral, etc.)
- Endpoint URL configurable via environment variable

### E. Retrieval Pipeline

**search.py**:
- `SearchPipeline` class orchestrating:
  1. Query embedding
  2. Initial vector search (top-150)
  3. Reranking (top-20 from 150)
  4. Metadata attachment
  5. Parent chunk retrieval if needed

### F. Comparison & Analysis

**analyzer.py**:
- Compare results from random vs AST across multiple queries
- Track: accuracy, relevance scores, chunk integrity, processing time
- Detect when chunks are fragmented across results

**reporter.py**:
- Generate formatted console output with tables
- Create comparison report showing metrics
- Export results to CSV/JSON

## Execution Flow

### 1. Setup Phase
```bash
# Ensure Ollama is running first
ollama serve

# In another terminal:
python main.py --action setup --db-user scott --db-password pwd --db-dsn localhost:1521/freepdb1
# Creates Oracle tables and indexes
```

### 2. Vectorization Phase
```bash
python main.py --action vectorize --chunking-method both --sample-file examples/sample.py \
  --enrich-metadata true --ollama-endpoint http://localhost:11434 --ollama-model llama2
# Processes sample.py with both random and AST chunking
# Extracts metadata using Ollama (local, no API key)
# Generates embeddings
# Stores in Oracle DB
```

### 3. Retrieval & Comparison Phase
```bash
python main.py --action search --query "data validation" --top-k 20 --rerank true
# Searches using both methods
# Applies reranking to AST results
# Displays comparison report
```

## Expected Outputs

1. **Console Output**: Side-by-side comparison table showing:
   - Query text
   - Random chunking results (top-5)
   - AST chunking results (top-5)
   - Reranking impact (for AST)
   - Metrics (time, accuracy, chunk completeness)

2. **CSV Export** (optional):
   - Query, method, chunk_id, relevance_score, rank, metrics

3. **Visualization**:
   - Chart comparing retrieval quality (random vs AST vs AST+reranking)
   - Show improvement percentages

## Ollama Setup Instructions

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### 2. Download a Model
```bash
# Recommended: llama2 (balanced speed/quality)
ollama pull llama2

# Or other options:
ollama pull llama3.1      # Better quality, slightly slower
ollama pull mistral       # Fast
ollama pull neural-chat   # Good for instruction following
```

### 3. Start Ollama Server
```bash
ollama serve
# Runs on http://localhost:11434 by default
```

### 4. Verify Connection
```bash
# Test Ollama API
curl http://localhost:11434/api/tags
```

## Configuration

Create `.env` file in project root:
```
ORACLE_USER=scott
ORACLE_PASSWORD=yourpassword
ORACLE_DSN=localhost:1521/freepdb1
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
VERBOSE=true
```

## Key Implementation Tips

1. **Error Handling**: Gracefully handle Oracle connection issues, Ollama unavailability, malformed code
2. **Ollama Fallback**: If metadata extraction fails, use default values instead of crashing
3. **Performance**: Use batch operations for vectorization, optimize DB queries, cache Ollama responses
4. **Testing**: Include unit tests for chunking, embedding, retrieval, and Ollama integration
5. **Documentation**: Every function should have docstrings explaining parameters and returns
6. **Logging**: Implement logging for debugging and monitoring
7. **Configuration**: Use environment variables and config files for all settings

## Success Criteria

✓ Random chunking breaks code at arbitrary boundaries
✓ AST chunking preserves complete functions/classes
✓ Ollama-based metadata extraction provides semantic context without API keys
✓ Oracle DB stores and retrieves vectors correctly
✓ Reranking improves top-20 relevance by 15%+
✓ AST method shows 20%+ higher accuracy than random
✓ Clear, actionable comparison reports generated
✓ All features accessible via intuitive CLI
✓ Code is production-quality with error handling
✓ Documentation is comprehensive and clear
✓ Project works with local Ollama setup (no external API dependencies)