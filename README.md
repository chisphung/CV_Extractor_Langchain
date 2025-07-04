# Langchain Services

## 1. Setup

### 1.1. Donwload data
My implementation of the Langchain Services is based on the [AIO Project Langchain Services](https://github.com/ThuanNaN/Langchain_Services/)

This repository uses data from the [Ciriculum Vitae (CV) dataset](https://github.com/arefinnomi/curriculum_vitae_data). You can download the full dataset and other necessary files using the following commands:

```bash
data_source/generative_ai/download.sh
```
Note that the dataset contains over 3000 CVs, which could be resulted in incorrect parsing results. Our repository also provides a subset of the dataset with about 50 CVs for testing purposes.

### 1.2. Run service in local

Python version: `3.11.9`

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
# Start the server
uvicorn src.app:app --host "0.0.0.0" --port 5000 --reload
```
This will ask for the Google API key, which you can get from the [Google Cloud Console](https://aistudio.google.com/apikey). After providing the key, the server will start and you can access it at `http://localhost:5000/docs`.
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
## 3. Deployment
### Langserve 
After the service is running, you can deploy it using Langserve in the following url:
```
https://localhost:5000/langserve/chat/playground
https://localhost:5000/langserve/generative_ai/playground
```
### Streamlit 
You can also deploy the service using Streamlit with the following command:

```bash
streamlit run src/streamlit.py
```