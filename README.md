# Product Support Chatbot (RAG)

A retrieval-augmented generation (RAG) chatbot that answers customer queries about products using real review data. Built with LangChain, FastAPI, AstraDB, and Groq.

## Architecture

```
User Query → FastAPI → Conversation History (in-memory)
                            ↓
                     LangChain LCEL Chain
                            ├── Retriever (AstraDB vector search + relevance filtering)
                            ├── Prompt Template (grounded, multi-turn aware)
                            └── LLM (Llama 3.1 8B via Groq)
                            ↓
                     Response → Evaluation (offline scoring)
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM Orchestration | LangChain (LCEL) |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | DataStax AstraDB |
| LLM | Llama 3.1 8B (via Groq) |
| Frontend | HTML/CSS/jQuery |
| Data | Flipkart product reviews (450 entries) |

## Key Features

- **Multi-turn conversation** — Sliding window history so users can ask follow-up questions naturally
- **Retrieval confidence scoring** — Filters low-relevance documents before passing to LLM, reducing hallucination
- **Prompt grounding** — LLM is strictly instructed to use only retrieved context
- **Offline evaluation** — Automated test suite that scores retrieval relevance and answer quality across test cases
- **Batched ingestion** — Rate-limit aware data pipeline with exponential backoff for free-tier APIs

## Project Structure

```
├── main.py                          # FastAPI app, chain setup, conversation memory
├── data_ingestion/
│   └── ingestion_pipeline.py        # CSV → Documents → AstraDB (batched with retry)
├── Retriever/
│   └── retrieval.py                 # Vector store retriever + confidence scoring
├── evaluation/
│   └── evaluate.py                  # Offline RAG quality evaluation
├── utils/
│   ├── config_loader.py             # YAML config reader
│   └── model_loader.py              # Embedding + LLM loader (singleton)
├── prompt_library/
│   └── prompt.py                    # Grounded prompt templates
├── config/
│   └── config.yaml                  # Model names, DB settings, top_k
├── templates/
│   └── chat.html                    # Chat UI
├── static/
│   └── style.css                    # Styles
└── data/
    └── data.csv                     # Product reviews dataset
```

## Setup

```bash
git clone https://github.com/nikhil550000/customer_support_system
cd customer_support_system

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your AstraDB, Google, Groq keys

# Ingest data
python data_ingestion/ingestion_pipeline.py

# Run
uvicorn main:app --reload --port 8000

# Evaluate (optional)
python evaluation/evaluate.py
```

## Evaluation

```bash
python evaluation/evaluate.py
```

Outputs `evaluation/eval_results.json` with per-query retrieval relevance scores, answer keyword overlap, and out-of-scope rejection checks.

## Key Design Decisions

- **Groq (Llama 3.1 8B)** — fast inference at zero cost vs. Gemini/OpenAI
- **Relevance threshold filtering** — documents below similarity threshold are discarded before prompting
- **In-memory conversation store** — simple for demo; would use Redis with TTL in production
- **Batched ingestion with backoff** — stays within free-tier rate limits reliably