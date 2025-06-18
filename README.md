# Langchain Services

## 1. Setup

### 1.1. Donwload data

Require **wget** and **gdown** package

```bash
pip3 install wget gdown
cd data_source/generative_ai && python download.py
```

### 1.2. Run service in local

Python version: `3.11.9`

```bash
pip3 install -r requirements.txt
# Start the server
uvicorn src.app:app --host "0.0.0.0" --port 5000 --reload
```

Wait a minute for handling data and starting server.

### 1.3 Run service in docker

```bash
docker compose up -d
```

Turn off service

```bash
docker compose -f down
```

## 2. Architecture

The service ingests CV PDFs from local files or Google Drive links, extracts
information using an LLM and stores embeddings in a FAISS vector store. The
workflow is:

`ingestion -> extraction -> storage -> search`.

### Modules

- **src/rag/file_loader.py** – download/load and split documents.
- **src/rag/cv_extractor.py** – prompt chain for CV parsing.
- **src/rag/vectorstore.py** – persistent FAISS store with metadata support.
- **src/app.py** – FastAPI server exposing upload and search endpoints.

### API Usage

Upload a CV from a local file:

```bash
curl -X POST -F "file=@resume.pdf" http://localhost:5000/upload_cv
```

Search for candidates:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "python developer"}' \
     http://localhost:5000/search_candidates
```
## Deployment
### Langserve 
After the service is running, you can deploy it using Langserve in the following url:
```
https://localhost:5000/langserve/chat/playground
https://localhost:5000/langserve/generative_ai/playground
```