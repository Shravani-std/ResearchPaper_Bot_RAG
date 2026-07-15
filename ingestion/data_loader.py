from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.document_loaders import TextLoader 
# from langchain_community.document_loaders import  CSVLoader # type: ignore
# from langchain_community.document_loaders import Docx2txtLoader # type: ignore
# from langchain_community.document_loaders.excel import UnstructuredExcelLoader # type: ignore
# from Langchain_community.document_loaders import JSONLoader # type: ignore


def load_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")
    documents = []

    #PDF Files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files. {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF: {pdf_file}, Error: {e}")


    #Text Files
    text_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(text_files)} Text files. {[str(f) for f in text_files]}")
    for text_file in text_files:
        print(f"[DEBUG] Loading Text File: {text_file}")
        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Text docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Text File: {text_file}, Error: {e}")
    return documents


if __name__=="__main__":
    data_path = r"D:\AI\RAG\Projects\data"
    docs = load_documents(data_path)
    print(f"[DEBUG] Total documents loaded: {len(docs)}")
    print("Example document content:")
    for doc in docs[:3]:  # Print content of first 3 documents
        print(doc.page_content)