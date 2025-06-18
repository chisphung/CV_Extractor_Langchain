from typing import Union, List, Literal
import glob
import os
from tqdm import tqdm
import multiprocessing
import gdown
from pathlib import Path
import zipfile
import shutil
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fetch_pdfs(sources: List[str], dest_dir: str = "./data_source/generative_ai/curriculum_vitae_data") -> List[str]:
    """Fetch pdf files from local paths, ZIP files, or Google Drive links."""
    os.makedirs(dest_dir, exist_ok=True)
    pdf_paths = []

    for src in sources:
        if src.startswith("http"):
            # Handle Google Drive links
            if "folders" in src:
                # Google Drive FOLDER
                gdown.download_folder(url=src, output=dest_dir, quiet=False, use_cookies=False)
            else:
                # Google Drive FILE (could be PDF or ZIP)
                filename = os.path.basename(src)
                download_path = os.path.join(dest_dir, filename)
                gdown.download(url=src, output=download_path, quiet=True, fuzzy=True)

                if zipfile.is_zipfile(download_path):
                    with zipfile.ZipFile(download_path, "r") as zip_ref:
                        zip_ref.extractall(dest_dir)
                    os.remove(download_path)
                elif download_path.endswith(".pdf"):
                    pdf_paths.append(download_path)

        elif os.path.isdir(src):
            # Add all PDFs from the folder recursively
            for pdf_file in Path(src).rglob("*.pdf"):
                pdf_paths.append(str(pdf_file.resolve()))

        elif os.path.isfile(src):
            if src.endswith(".pdf"):
                pdf_paths.append(src)
            elif zipfile.is_zipfile(src):
                with zipfile.ZipFile(src, "r") as zip_ref:
                    zip_ref.extractall(dest_dir)

    # Finally, collect all PDFs under dest_dir
    for pdf_file in Path(dest_dir).rglob("*.pdf"):
        pdf_paths.append(str(pdf_file.resolve()))

    return list(set(pdf_paths))



def remove_non_utf8_characters(text):
    return ''.join(char for char in text if ord(char) < 128)

def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def load_cv(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def load_from_sources(sources: List[str]) -> List:
    files = fetch_pdfs(sources)
    docs = []
    for f in files:
        docs.extend(load_cv(f))
    return docs

def load_html(html_file):
    docs = BSHTMLLoader(html_file).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
def get_num_cpu():
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded


class HTMLLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, html_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(html_files)
            with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
                for result in pool.imap_unordered(load_html, html_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded


class TextSplitter:
    def __init__(self, 
                 separators: List[str] = ['\n\n', '\n', ' ', ''],
                 chunk_size: int = 300,
                 chunk_overlap: int = 0
                 ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    def __call__(self, documents):
        return self.splitter.split_documents(documents)



class Loader:
    def __init__(self, 
                 file_type: str = Literal["pdf", "html"],
                 split_kwargs: dict = {
                     "chunk_size": 300,
                     "chunk_overlap": 0}
                 ) -> None:
        assert file_type in ["pdf", "html"], "file_type must be either pdf or html"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        elif file_type == "html":
            self.doc_loader = HTMLLoader()
        else:
            raise ValueError("file_type must be either pdf or html")

        self.doc_spltter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_spltter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            files = glob.glob(f"{dir_path}/*.html")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        return self.load(files, workers=workers)