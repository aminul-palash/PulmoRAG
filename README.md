# PulmoRAG - Pulmonary Disease RAG System

RAG system for evidence-based pulmonary disease treatment information, including COPD, asthma, pneumonia, tuberculosis, and more.

## Quick Start

```bash
# 1. Setup environment
cp .env.example .env    # Edit with your API keys

# 2. Build and run
docker compose up -d

# 3. Access
# App: http://localhost:8501 | ChromaDB: http://localhost:8000
```

## Data Pipeline

### 1. Document Processing
Process raw documents (PDF, DOCX, TXT) into structured chunks:

```bash
# Process all documents in data/raw/
docker compose exec rag-app python scripts/setup_data.py

# Output: data/processed/chunks.jsonl (chunked documents)
#         data/processed/manifest.json (processing metadata)
```

### 2. Vector Database Setup
Build ChromaDB (dense) and BM25 (sparse) indexes:

```bash
# Build both indexes from processed chunks
docker compose exec rag-app python scripts/build_vector_db.py build

# Rebuild (clear existing and re-index)
docker compose exec rag-app python scripts/build_vector_db.py build --clear
```

**Storage:**
- ChromaDB (dense): `data/vectordb/`
- BM25 (sparse): `data/bm25_index/`

### 3. Test Retrieval
Test search with different retrieval modes:

```bash
# Hybrid + reranking (default) - best quality
docker compose exec rag-app python scripts/build_vector_db.py test "What are treatments for asthma?"

# Compare all methods (dense, sparse, hybrid, reranked)
docker compose exec rag-app python scripts/build_vector_db.py compare "dupilumab FEV1"

# Specific mode
docker compose exec rag-app python scripts/build_vector_db.py test "query" --mode dense
```

### 4. Download LLM Model (Optional)
For local LLM inference without API keys:

```bash
# Download Mistral model (~4GB)
docker compose exec rag-app python scripts/download_model.py
```

## Common Commands

| Action | Command |
|--------|---------|
| Start services | `docker compose up -d` |
| Stop services | `docker compose down` |
| View logs | `docker compose logs -f rag-app` |
| Enter container | `docker compose exec rag-app bash` |
| Run tests | `docker compose exec rag-app pytest` |
| Check status | `docker compose ps` |

## When Code Changes

### App Code Changed (Python/Streamlit)
```bash
# Hot reload - changes auto-reflect (volumes mounted)
# If not working, restart the app container:
docker compose restart rag-app
```

### Docker Config Changed (Dockerfile/docker-compose.yml)
```bash
# Rebuild and restart
docker compose up -d --build
```

### Dependencies Changed (requirements.txt)
```bash
# Force rebuild
docker compose build --no-cache rag-app
docker compose up -d
```

## Port Issues

### Check What's Using a Port
```bash
sudo lsof -i :8501    # Check Streamlit port
sudo lsof -i :8000    # Check ChromaDB port
```

### Kill Process Using Port
```bash
sudo kill -9 $(sudo lsof -t -i :8501)
```

### Clean Docker Reset
```bash
docker compose down -v              # Stop and remove volumes
docker system prune -f              # Clean unused resources
docker compose up -d --build        # Fresh start
```

## Evaluation

RAGAS-style LLM-as-judge metrics: **Faithfulness** | **Answer Relevancy** | **Context Relevancy**

```bash
# Quick evaluation (5 queries, local LLM judge)
docker compose exec rag-app python scripts/evaluate.py --quick

# Compare retrieval methods (dense vs hybrid vs reranked)
docker compose exec rag-app python scripts/evaluate.py --compare

# Full options
docker compose exec rag-app python scripts/evaluate.py \
    --method dense|hybrid|reranked \
    --category treatment|diagnosis|medication \
    --use-openai \     # Use OpenAI as judge (more accurate)
    --html             # Generate visual HTML report
```

**Outputs:** `data/evaluation/reports/` (JSON + HTML reports)

## Environment Variables

```env
# API Keys (optional if using local LLM)
OPENAI_API_KEY=         # OpenAI API key
ANTHROPIC_API_KEY=      # Anthropic API key

# Retrieval Settings
USE_HYBRID_RETRIEVAL=true    # Enable hybrid search
USE_RERANKER=true            # Enable cross-encoder reranking
RERANK_CANDIDATES=20         # Docs to retrieve before reranking
```

## Architecture

```
Query → ┬→ Dense (ChromaDB) ─┐
        │                     ├→ RRF Fusion → Reranker → Top K → LLM → Response
        └→ Sparse (BM25) ────┘
```

## Features

- **Hybrid Retrieval**: Dense + BM25 with RRF fusion
- **Reranking**: Cross-encoder for improved precision
- **Local LLM**: MistralLite for offline inference
- **Evaluation**: RAGAS-style automated quality metrics
- **Citations**: Automatic source attribution
- **UI**: Streamlit interface

---

## User Guide

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 20 GB |
| GPU | Not required | CUDA-compatible (faster) |
| Docker | 20.10+ | Latest |

### Using the Chat Interface

1. Open http://localhost:8501
2. Type your pulmonary disease-related question
3. View response with cited sources
4. Click source links to see relevance scores

### Example Queries

| Category | Query |
|----------|-------|
| COPD | "What are first-line treatments for COPD?" |
| Asthma | "How is asthma different from COPD?" |
| Pneumonia | "What antibiotics treat pneumonia?" |
| Tuberculosis | "What is the TB treatment regimen?" |
| Lung Cancer | "What are NSCLC treatment options?" |
| General | "Does pulmonary rehabilitation help?" |

### Adding Custom Documents

```bash
# 1. Add PDFs to data/raw/guidelines/, research/, or patient_education/
cp my_document.pdf data/raw/research/

# 2. Reprocess documents
docker compose exec rag-app python scripts/setup_data.py

# 3. Rebuild indexes
docker compose exec rag-app python scripts/build_vector_db.py build --clear
```

### Limitations

- **Scope**: Pulmonary diseases only (respiratory conditions)
- **Currency**: Based on indexed documents (check `data/processed/manifest.json` for dates)
- **Not medical advice**: For informational purposes only
- **Language**: English only

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "No LLM backend available" | Run `python scripts/download_model.py` or add API key |
| Slow responses | First query loads models; subsequent queries faster |
| Empty results | Check if documents are indexed: `python scripts/build_vector_db.py stats` |
| Port in use | `sudo kill -9 $(sudo lsof -t -i :8501)` |

---

## Project Structure

```
PulmoRAG/
├── src/
│   ├── data/          # Document loading & processing
│   ├── retrieval/     # Dense, sparse, hybrid search
│   ├── generation/    # LLM interface
│   ├── rag/           # Pipeline orchestration
│   └── evaluation/    # RAGAS metrics
├── scripts/           # CLI tools
├── data/
│   ├── raw/           # Source documents (PDF)
│   ├── processed/     # Chunked documents
│   ├── vectordb/      # ChromaDB storage
│   └── bm25_index/    # BM25 index
└── demo.py            # Streamlit app
```

## License

MIT
