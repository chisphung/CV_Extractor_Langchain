from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class CandidateDB:
    def __init__(self, embedding_model=None):
        self.embedding = embedding_model or HuggingFaceEmbeddings()
        self.db = None

    def build_db(self, docs):
        self.db = FAISS.from_documents(docs, self.embedding)

    def search(self, query, k=3):
        if self.db is None:
            raise ValueError("Database has not been built. Call build_db() with documents first.")
        else:
            return self.db.similarity_search(query, k=k)
