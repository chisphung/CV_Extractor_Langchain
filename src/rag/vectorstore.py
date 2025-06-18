import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class CandidateDB:
    def __init__(self, persist_dir: str = "./vectorstore", embedding_model=None):
        self.embedding = embedding_model or HuggingFaceEmbeddings()
        self.persist_dir = persist_dir
        self.db = None

    def build_db(self, docs):
        self.db = FAISS.from_documents(docs, self.embedding)
        self.db.save_local(self.persist_dir)

    def add_documents(self, docs):
        if self.db is None:
            if os.path.exists(self.persist_dir):
                self.db = FAISS.load_local(self.persist_dir, self.embedding)
            else:
                self.db = FAISS.from_documents(docs, self.embedding)
                self.db.save_local(self.persist_dir)
                return
        self.db.add_documents(docs)
        self.db.save_local(self.persist_dir)

    def search(self, query, k=3):
        if self.db is None:
            if os.path.exists(self.persist_dir):
                self.db = FAISS.load_local(self.persist_dir, self.embedding)
            else:
                raise ValueError("Database has not been built. Call build_db() with documents first.")
        return self.db.similarity_search(query, k=k)

    def get_retriever(self, k=3):
        if self.db is None:
            if os.path.exists(self.persist_dir):
                self.db = FAISS.load_local(self.persist_dir, self.embedding)
            else:
                raise ValueError("Database has not been built.")
        return self.db.as_retriever(search_kwargs={"k": k})
