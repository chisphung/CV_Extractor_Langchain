from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langserve import add_routes
from typing import List, Any
from src.base.llm_model import get_llm
from src.rag.file_loader import Loader, Exporter
from src.rag.cv_extractor import CVExtractor
from src.rag.vectorstore import CandidateDB
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.chat.main import build_chat_chain
import os

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

llm = get_llm()
extractor = CVExtractor(llm)
candidate_db = CandidateDB()

genai_docs = "./data_source/generative_ai"
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=6)

@app.post("/upload_cv")
async def upload_cv(
    file: List[UploadFile] = File(None),
    drive_link: str = Form(None)
):
    os.makedirs("./data_source/generative_ai", exist_ok=True)
    sources = []

    if file:
        for f in file:
            file_path = f"./data_source/generative_ai/{f.filename}"
            with open(file_path, "wb") as out_file:
                content = await f.read()
                out_file.write(content)
            sources.append(file_path)

    if drive_link:
        sources.append(drive_link)

    # Load all files with multiprocessing-aware loader
    loader = Loader(file_type="pdf")
    docs = loader.load(sources, workers=7)

    extracted_data = extractor.extract(docs)
    candidate_db.add_documents(docs)

    return {"message": "CVs processed", "extracted": extracted_data}

class SearchRequest(BaseModel):
    query: str
    filter: dict | None = None

@app.post("/search_candidates")
async def search_candidates(req: SearchRequest):
    results = candidate_db.search(req.query)
    matches = [
        {"text": doc.page_content, "metadata": doc.metadata} for doc in results
    ]
    return {"matches": matches}

class ExportRequest(BaseModel):
    data: Any
    outdir: str

@app.post("/export_candidates")
async def export_candidates(req: ExportRequest):
    exporter = Exporter(export_dir=req.outdir, file_name="candidates.json")
    file_path = exporter(req.data)
    return {"message": "Candidates exported", "file_path": file_path}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

@app.get("/check")
async def check():
    return {"status": "ok"}

# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           genai_chain, 
           path="/generative_ai")

add_routes(app,
           chat_chain,
           path="/chat")