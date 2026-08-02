from pathlib import Path
from typing import List, Any
# from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.document_loaders import TextLoader 
from langchain_core.documents import Document
import fitz
from ingestion.text_extraction import clean_text

def load_documents(data_dir: str) -> List[Any]:

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")
    documents = []

    #PDF Files
    pdf_files = list(data_path.glob("**/*.pdf"))

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
            print(e)


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


# if __name__ == "__main__":
#     data_path = r"D:\AI\RAG\Projects\data"

#     docs = load_documents(data_path)

#     print(f"Total documents: {len(docs)}")


#     for doc in docs[:5]:
#         print("=" * 50)
#         print(doc.metadata)
#         print(doc.page_content[:300])

