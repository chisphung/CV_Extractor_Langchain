from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langserve import add_routes
from src.base.llm_model import get_llm
from src.rag.file_loader import load_cv, split_documents
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
async def upload_cv(file: UploadFile = File(...)):
    os.makedirs("./data_source/generative_ai", exist_ok=True)

    file_path = f"./data_source/generative_ai/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    docs = load_cv(file_path)
    chunks = split_documents(docs)
    extracted_data = extractor.extract(chunks)

    candidate_db.build_db(chunks)
    print("Candidate DB updated with new CVs")

    return {"message": "CV processed", "extracted": extracted_data}


class SearchRequest(BaseModel):
    query: str

@app.post("/search_candidates")
async def search_candidates(req: SearchRequest):
    results = candidate_db.search(req.query)
    return {"matches": [doc.page_content for doc in results]}

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