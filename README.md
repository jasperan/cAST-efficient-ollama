# cAST-efficient-ollama

A local-first demo project that compares random code chunking vs cAST/AST-aware chunking for code retrieval.

## What changed

This repo now supports a **real local walkthrough** without requiring Oracle, Ollama, or model downloads:

- `doctor` checks the runtime and shows which backends will be used
- `demo` runs an end-to-end AST-vs-random comparison and exports reports
- `vectorize` + `search` work with automatic local fallbacks
- the CLI prefers lightweight local backends when you use `--profile local`

Oracle/Ollama are still supported as advanced integrations when available.

## Quickstart

### Option 1: install locally

```bash
git clone https://github.com/jasperan/cAST-efficient-ollama.git
cd cAST-efficient-ollama
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install only the lightweight local mode above, or opt into richer integrations when you need them:

```bash
pip install -e ".[models,ollama,oracle]"
```

### Option 2: use the helper script

```bash
curl -fsSL https://raw.githubusercontent.com/jasperan/cAST-efficient-ollama/main/install.sh | bash
```

## Fast local walkthrough

This is the recommended README-reader flow.

### 1) Check the environment

```bash
cast-ollama --action doctor --profile local
```

JSON output is also available:

```bash
cast-ollama --action doctor --profile local --output json
```

### 2) Run the end-to-end demo

```bash
cast-ollama --action demo --profile local --sample-file examples/sample.py --report-dir ./reports
```

This writes:

- `reports/demo_report.csv`
- `reports/demo_report.json`

### 3) Run the tests

```bash
python -m unittest discover tests -v
```

## Core CLI commands

### Reset / setup storage

- Oracle mode:

```bash
cast-ollama --action setup
```

- Local Chroma fallback is used automatically when Oracle is unavailable.

### Vectorize a file

```bash
cast-ollama --action vectorize --profile local --chunking-method both --sample-file examples/sample.py
```

### Search indexed chunks

```bash
cast-ollama --action search --profile local --chunking-method both --query "validate email" --report-dir ./reports
```

This writes:

- `reports/search_report.csv`
- `reports/search_report.json`

## CLI profiles and backends

### Local profile

Use this when you want the most reliable no-surprises experience:

```bash
cast-ollama --action demo --profile local
```

It prefers:

- hash embeddings
- lexical reranking
- Chroma fallback storage

### Advanced backends

You can still request richer integrations explicitly:

```bash
cast-ollama --action search --embedding-backend sentence-transformers --reranker-backend flag --query "database connection"
```

## Configuration

Copy the example file if you want to pin config values:

```bash
cp config.yaml.example config.yaml
```

Example structure:

```yaml
oracle:
  user: scott
  password: password
  dsn: localhost:1521/freepdb1

ollama:
  endpoint: http://localhost:11434
  model: llama2
  embed_model: nomic-embed-text

embedding:
  model: microsoft/codebert-base
  backend: hash
  dimension: 768

reranker:
  model: BAAI/bge-reranker-v2-m3
  backend: lexical

chroma:
  persist_dir: .cast_ollama/chroma
  collection: code_chunks

reporting:
  report_dir: .

app:
  verbose: false
  chunk_size: 2000
  overlap_percentage: 10
  top_k_retrieval: 150
  top_k_rerank: 20
```

## Standalone demo script

You can also run:

```bash
python demo_paper.py
```

## Project structure

- `src/cast_ollama/cli.py` — CLI entrypoint
- `src/cast_ollama/chunking/` — random and AST-aware chunkers
- `src/cast_ollama/embedding/` — embedding + reranker backends with fallbacks
- `src/cast_ollama/retrieval/` — search pipeline
- `src/cast_ollama/chroma_db/` — local vector-store fallback
- `src/cast_ollama/oracle_db/` — Oracle integration
- `src/cast_ollama/demo.py` — reusable end-to-end demo flow
- `tests/` — unit + walkthrough-oriented tests

## Notes

- If Oracle is unavailable, the project falls back to Chroma automatically.
- If FlagEmbedding is unavailable, the project falls back to lexical reranking.
- If transformer embeddings are unavailable, the project can run with local hash embeddings.
