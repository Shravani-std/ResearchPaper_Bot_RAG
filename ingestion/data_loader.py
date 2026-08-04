from pathlib import Path
from typing import List, Any
# from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.document_loaders import TextLoader 
from langchain_core.documents import Document
import fitz
from ingestion.text_extraction import clean_text
from core.logger import get_logger, log_step

logger = get_logger("data_loader")

def load_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    log_step(f"Loading documents from: {data_path}")
    documents = []

    # -----------------------------------------------------
    # Handle single-file input (e.g. Streamlit temp upload)
    # vs. folder input (e.g. batch/CLI ingestion)
    # -----------------------------------------------------
    if data_path.is_file():
        pdf_files = [data_path] if data_path.suffix.lower() == ".pdf" else []
        text_files = [data_path] if data_path.suffix.lower() == ".txt" else []
    else:
        pdf_files = list(data_path.glob("**/*.pdf"))
        text_files = list(data_path.glob("**/*.txt"))

    #PDF Files
    for pdf_file in pdf_files:

        try:
            pdf = fitz.open(pdf_file)

            for page_num, page in enumerate(pdf):

                text = page.get_text()
                text = clean_text(text)

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_file),
                            "page": page_num + 1
                        }
                    )
                )

            pdf.close()

        except Exception as e:
            logger.warning("Failed to load PDF file %s: %s", pdf_file, e)

    #Text Files
    log_step(f"Found {len(text_files)} text files to load.")
    for text_file in text_files:
        log_step(f"Loading text file: {text_file}")
        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()
            log_step(f"Loaded {len(loaded)} text documents from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            logger.exception("Failed to load text file %s", text_file)

    return documents

# if __name__ == "__main__":
#     data_path = r"D:\AI\RAG\Projects\data"

#     docs = load_documents(data_path)

#     print(f"Total documents: {len(docs)}")


#     for doc in docs[:5]:
#         print("=" * 50)
#         print(doc.metadata)
#         print(doc.page_content[:300])

