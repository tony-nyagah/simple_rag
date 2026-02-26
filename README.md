# 🔍 Simple RAG

A minimal **Retrieval-Augmented Generation** system built with Python, Google
Gemini, and FastAPI. Upload a PDF, ask questions about it, and get accurate
answers grounded in the document's content.

Built as a learning project to understand how RAG works from the ground up — no
vector databases, no heavyweight frameworks, just the core ideas in ~300 lines
of Python.

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that grounds LLM answers
in _your_ documents instead of relying on the model's training data alone.

```
┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│  Upload  │────▶│  Chunk   │────▶│  Embed    │────▶│  Store   │
│   PDF    │     │  Text    │     │  Chunks   │     │ Vectors  │
└──────────┘     └──────────┘     └───────────┘     └────┬─────┘
                                                         │
┌──────────┐     ┌──────────┐     ┌───────────┐         │        ┌──────────┐
│   Ask    │────▶│  Embed   │────▶│  Find Top │─────────┘        │          │
│ Question │     │  Query   │     │  Matches  │─────────────────▶│  Gemini  │
└──────────┘     └──────────┘     └───────────┘ relevant chunks  │   LLM    │
                                                                 └────┬─────┘
                                                                      │
                                                                 ┌────▼─────┐
                                                                 │  Answer  │
                                                                 │ grounded │
                                                                 │ in docs  │
                                                                 └──────────┘
```

1. **Extract** — Pull text out of a PDF using PyMuPDF
2. **Chunk** — Split text into overlapping pieces on sentence boundaries
3. **Embed** — Convert each chunk into a vector using `gemini-embedding-001`
4. **Retrieve** — Embed the question and find the most similar chunks using
   cosine similarity
5. **Generate** — Send the question + top-K chunks to `gemini-2.5-flash`, which
   answers from the document content

---

## Project Structure

```
simple_rag/
├── src/simple_rag/
│   ├── rag.py          # Core RAG logic — chunking, embedding, retrieval, generation
│   ├── api.py          # FastAPI REST API
│   ├── main.py         # Interactive CLI
│   └── eval.py         # LLM-as-a-judge evaluation harness
├── pyproject.toml      # Project config & dependencies (Pixi)
└── README.md
```

| Module    | Purpose                                                                                                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rag.py`  | The engine. PDF extraction, sentence-aware chunking, Gemini embedding, cosine-similarity retrieval, and answer generation. Both the CLI and API import from here.           |
| `api.py`  | FastAPI app wrapping the RAG engine into REST endpoints. Upload PDFs, manage documents, and query them over HTTP. Includes auto-generated interactive docs at `/docs`.      |
| `main.py` | Minimal interactive CLI — load a PDF, ask questions in a loop.                                                                                                              |
| `eval.py` | Automated evaluation harness: runs 10 test questions against "Attention is All You Need", uses Gemini as a judge, and prints a scored report with per-difficulty breakdown. |

---

## Setup

### Prerequisites

- Python 3.11+
- [Pixi](https://pixi.sh) package manager
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free tier
  works)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd simple_rag

# Install dependencies
pixi install

# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

---

## Usage

### 1. CLI — Interactive Chat

Chat with a PDF directly in your terminal:

```bash
pixi run python -m simple_rag.main path/to/document.pdf
```

```
Loading PDF: path/to/document.pdf
Ready! Ask questions (type 'quit' to exit)

You: What is the main contribution of this paper?
Assistant: The paper introduces the Transformer, a sequence-to-sequence model
based entirely on attention mechanisms, dispensing with recurrence and
convolutions entirely...

You: quit
```

### 2. API — REST Interface

Start the API server:

```bash
pixi run serve
# → http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

#### Endpoints

| Method   | Path                  | Description                                   |
| -------- | --------------------- | --------------------------------------------- |
| `POST`   | `/documents`          | Upload a PDF — chunks and embeds it in memory |
| `GET`    | `/documents`          | List all loaded documents                     |
| `GET`    | `/documents/{doc_id}` | Get metadata for a specific document          |
| `DELETE` | `/documents/{doc_id}` | Remove a document from memory                 |
| `POST`   | `/query`              | Ask a question against a loaded document      |

#### Example workflow

```bash
# Upload a PDF
curl -X POST http://localhost:8000/documents \
  -F "file=@path/to/paper.pdf"
# → { "doc_id": "a1b2c3d4e5f6", "num_chunks": 52, ... }

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "a1b2c3d4e5f6", "question": "What is the main contribution?"}'
# → { "answer": "...", "context_chunks": [...] }

# List loaded documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/a1b2c3d4e5f6
```

### 3. Evaluation — LLM-as-a-Judge

Run an automated evaluation against the bundled "Attention is All You Need" PDF:

```bash
pixi run eval
```

This runs 10 question/answer pairs at three difficulty levels (easy, medium,
hard) and uses Gemini to judge correctness. Results are printed to the terminal
and saved to `eval_report.json`.

```
======================================================================
  RAG Evaluation — LLM-as-a-Judge
======================================================================

📄 Ingesting: src/simple_rag/arxiv.pdf

─── Question 1/10 [EASY] ───
  Q: Who are the authors of this paper?
  A: The authors are Ashish Vaswani, Noam Shazeer, Niki Parmar...
  → ✔ CORRECT

...

======================================================================
  RESULTS SUMMARY
======================================================================

  Overall Score:  90.0%
  Total:          10
  ✔ Correct:      8
  ◐ Partial:      2
  ✘ Incorrect:    0

      EASY:  100.0%  (5/5 correct, 0 partial)
    MEDIUM:   83.3%  (2/3 correct, 1 partial)
      HARD:   75.0%  (1/2 correct, 1 partial)
```

You can also point the evaluator at any other PDF:

```bash
pixi run python -m simple_rag.eval path/to/your.pdf
```

---

## How it works — Key Design Decisions

### Sentence-aware chunking with overlap

Rather than splitting on fixed character boundaries (which can cut
mid-sentence), the chunker splits on sentence boundaries using a regex and then
assembles sentences into chunks of ~1024 characters. Each new chunk starts with
the last ~200 characters of the previous one to preserve context across chunk
boundaries.

### No vector database

Embeddings are stored in-memory as plain Python lists. Retrieval is a
brute-force cosine similarity scan across all chunks. This is fine for learning
and for documents up to a few hundred pages. For production scale, you'd swap
this out for something like Chroma, Qdrant, or pgvector.

### Rate-limit handling

The free Gemini tier is rate-limited. `rag.py` includes a `_retry_with_backoff`
helper that catches `429` errors and retries with exponential backoff (15s → 30s
→ 60s) before giving up.

### LLM-as-a-judge evaluation

Rather than writing per-question regex matchers, `eval.py` uses Gemini itself as
an impartial judge. The judge model is given the question, ground-truth answer,
and RAG answer, then returns a structured JSON verdict (`CORRECT` /
`PARTIALLY_CORRECT` / `INCORRECT`). This makes it easy to add new test cases
without writing custom scoring logic.

---

## Configuration

Key constants in `rag.py` that you can tweak:

| Constant        | Default                | Effect                                                   |
| --------------- | ---------------------- | -------------------------------------------------------- |
| `EMBED_MODEL`   | `gemini-embedding-001` | Gemini embedding model to use                            |
| `CHAT_MODEL`    | `gemini-2.5-flash`     | Gemini model used for answer generation and eval judging |
| `CHUNK_SIZE`    | `1024`                 | Target chunk size in characters                          |
| `CHUNK_OVERLAP` | `200`                  | Characters of overlap between consecutive chunks         |
| `TOP_K`         | `5`                    | Number of chunks retrieved per query                     |
| `MAX_RETRIES`   | `3`                    | Retry attempts on rate-limit errors                      |

---

## Dependencies

| Package            | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `google-genai`     | Gemini API client (embeddings + generation) |
| `pymupdf`          | PDF text extraction                         |
| `numpy`            | Cosine similarity computation               |
| `fastapi`          | REST API framework                          |
| `uvicorn`          | ASGI server                                 |
| `python-multipart` | File upload support for FastAPI             |
