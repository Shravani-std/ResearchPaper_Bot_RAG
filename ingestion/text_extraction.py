import re
import fitz  # type: ignore


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Steps:
    1. Remove extra spaces.
    2. Remove multiple blank lines.
    3. Remove tabs.
    4. Remove page breaks/form feed characters.
    5. Remove spaces before punctuation.
    """

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove form feed characters (page breaks)
    text = text.replace("\f", " ")

    # Remove multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF
    and clean the extracted text.
    """

    text = ""

    try:
        with fitz.open(pdf_path) as pdf:

            for page in pdf:
                text += page.get_text()

        # Clean extracted text
        text = clean_text(text)

    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path}")
        print(e)

    return text


# if __name__ == "__main__":

#     pdf_path = r"D:\AI\RAG\Projects\data\sample.pdf"

#     extracted_text = extract_text_from_pdf(pdf_path)

#     print(extracted_text)