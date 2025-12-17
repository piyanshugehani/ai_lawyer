# iLegalLearn — AI Lawyer

**iLegalLearn — AI Lawyer** is an AI-driven legal assistant designed for law students and legal professionals. It simplifies Indian legal research and learning by enabling intelligent search, understanding, and drafting using a curated corpus of Indian legal case PDFs (2004–2009).

---

## Table of Contents
- [Overview](#overview)
- [Core Objectives](#core-objectives)
- [Key Features](#key-features)
  - [👩‍🎓 Student Features](#-student-features)
  - [⚖️ Professional Features](#️-professional-features)
  - [🤖 AI Capabilities](#-ai-capabilities)
- [System Architecture (High-level)](#system-architecture-high-level)
- [Tech Stack (repo reality)](#tech-stack-repo-reality)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Environment variables](#environment-variables)
  - [Run locally (backend / UI)](#run-locally-backend--ui)
- [Folder Structure](#folder-structure)
- [Use Cases](#use-cases)
- [Future Scope](#future-scope)
- [Ethical & Legal Disclaimer](#ethical--legal-disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
iLegalLearn is a focused MVP that helps users explore, learn, and draft legal content using retrieval-augmented generation (RAG) and LLMs. The project is intentionally scoped to a historical corpus of Indian case PDFs (2004–2009) to enable safe, testable functionality while avoiding claims about complete coverage.

## Core Objectives
- Reduce complexity and time spent on legal research.
- Help law students learn faster by providing PYQs, simplified case summaries, and mock arguments.
- Assist legal professionals with drafting and precedent discovery workflows.
- Provide section-wise, year-wise, and jurisdiction-based search within the available corpus.

## Key Features

### 👩‍🎓 Student Features
- **Previous Year Questions (PYQs)**: Generate exam-style questions focused on the requested topic, jurisdiction, and timeframe.
- **Case summaries**: Concise IRAC-style summaries rewritten in simple language for study and revision.
- **Section-wise explanations**: Break down statutory sections and explain them in plain English.
- **Mock arguments & reasoning practice**: Produce short model arguments to practice applying law.

### ⚖️ Professional Features
- **Legal drafting**: Templates and first-draft language for notices, replies, and short submissions.
- **Case & judgment search**: Search the corpus and surface relevant cases by:
  - Section (statutory provision)
  - Year
  - Jurisdiction (court level)
  - Practice area / topic
- **Citation discovery**: Help identify likely reporter citations and suggested sources.

### 🤖 AI Capabilities
- **PDF ingestion & preprocessing**: Extract text from PDFs, chunk, and index.
- **Semantic search (LLM-based)**: Use embeddings + vector search to find relevant chunks.
- **Context-aware answers**: Retrieval-augmented generation (RAG) to ground LLM outputs in retrieved context.
- **Legal language simplification**: Convert complex judgments and provisions into learner-friendly language.

## System Architecture (High-level)
1. PDF ingestion
   - Convert PDFs to text and metadata
   - Chunk documents into manageable passages for embeddings
2. Embedding generation
   - Create vector embeddings for each chunk using an embeddings model
3. Vector index / search
   - Store vectors in a vector DB or FAISS shards for nearest-neighbour search
4. Retrieval
   - Given a query, run semantic search to fetch top-k chunks
5. Response generation
   - Construct a prompt that combines user query + retrieved context
   - Generate the final answer with an LLM (RAG)

Diagram (conceptual):

User → UI (Streamlit) → Backend (Python) → RAG → LLM → Response

## Tech Stack (repo reality)
> The following reflects this repository's current implementation and recommended components.

- Frontend / UI: **Streamlit** 
- Backend: **Python** 
- LLM Integration: **Google Generative AI (Gemini)** via `google-generativeai` or `langchain-google-genai`
- Embeddings & Vector Search: **FAISS** shards are included in the repo for offline retrieval; the project also contains utilities to upload to Pinecone if required.
- Data store: Minimal file-based metadata and JSON; optionally **PostgreSQL** for production workloads.
- Dataset: Indian legal case PDFs — ingested into FAISS shards during preprocessing.

## Installation & Setup
Below are the instructions to run the project locally on Windows (PowerShell). Adjust for macOS / Linux accordingly.

### Prerequisites
- Python 3.10+ (3.11 recommended)
- pip
- (Optional) Virtual environment tool: `venv`, `conda`, etc.
- Access to an LLM provider API key (Gemini / Google Generative API) if you want live LLM responses

### Environment setup
1. Create & activate a virtual environment (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure environment variables
- Create an `.env` file in `ai_lawyer/` or set environment variables in your shell.
- Minimum recommended variables:

```text
GEMINI_API_KEY=<your_api_key>
GEMINI_GEN_MODEL=<optional_custom_model_name>
```

> Note: For local testing you can leave `GEMINI_API_KEY` unset and rely on deterministic or mocked flows included in the repo.

### Run the project locally (Streamlit UI)

```powershell
# From repo root
streamlit run ai_lawyer/app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Folder Structure
Below is a high-level map of the repository and responsibilities (based on current layout):

```
ai_lawyer/
  ├─ app.py                 # Streamlit app (main UI + orchestration)
  ├─ understanding.py       # Intent/topic detection, clarifier logic
  ├─ rag.py                 # Dummy RAG helpers, retrieval wrappers
  ├─ student_tools.py       # Student-oriented output helpers (PYQs, simplify, mock args)
  ├─ professional_tools.py  # Professional-oriented helpers (drafts, citation finder)
  ├─ upload_to_pinecone.py  # Utility to upload vectors to Pinecone (optional)
  ├─ faiss_shards/          # FAISS index shards used for local retrieval
  ├─ faiss_shards_hc/       # Court-specific or category-specific shards
  ├─ processed.json         # Metadata for processed corpus
  └─ .env                   # Optional: local environment variables (not committed)

tests/
  └─ smoke.py               # Small smoke test to validate core modules

requirements.txt
README.md
```

## Use Cases
- Law students preparing for semester exams or bar-style PYQs.
- Advocates drafting notices or searching for precedent on key issues.
- Researchers looking for specific sections or historical decisions.
- Novice users who want simplified explanations of legal provisions.

## Future Scope
Planned improvements and experiments:
- Real-time case ingestion pipeline to support newer publications
- Multi-language support (Hindi / English) for wider accessibility
- Voice-based legal queries and mobile-friendly UI
- Court-specific intelligence (bench composition, frequent judges)
- AI-powered case outcome prediction (research-only; include strong ethical guardrails)

## Ethical & Legal Disclaimer
- iLegalLearn is an educational and research tool. It is NOT a replacement for a licensed lawyer.
- Outputs may contain errors, omissions, or hallucinations. Always verify legal conclusions against primary sources and consult a qualified lawyer for legal advice.
- Do not rely on this tool for time-sensitive or critical legal decisions.

## Contributing
This project is open-source-friendly. Contributions are welcome.

Guidelines:
- Fork the repository and open a branch for your work.
- Run tests and linting locally before opening a PR.
- Keep changes focused and document design decisions in the PR description.
- If adding a model or API integration, avoid committing secrets. Use `.env` or `st.secrets` locally.
