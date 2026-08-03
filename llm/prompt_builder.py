from typing import List, Dict


class PromptBuilder:

    def __init__(self):

        self.system_prompt = """
You are an expert AI Research Assistant.

Your job is to answer ONLY using the provided context.

Rules:
1. Use only the supplied context.
2. Do not invent information.
3. If the answer is not present, say:
   "I couldn't find enough information in the provided documents."
4. Answer clearly and concisely.
5. Cite the document source and page whenever possible.
6. If multiple documents support the answer, combine them naturally.
"""

    def build(
        self,
        query: str,
        documents: List[Dict],
    ) -> str:

        context = []

        for idx, doc in enumerate(documents, start=1):

            source = doc.get("source", "Unknown")
            page = doc.get("page", "Unknown")
            section = doc.get("section", "Unknown")
            text = doc.get("text", "")

            context.append(
                f"""
==============================
Document {idx}

Source  : {source}
Page    : {page}
Section : {section}

Content:
{text}
"""
            )

        context = "\n".join(context)

        prompt = f"""
{self.system_prompt}

======================================================
CONTEXT
======================================================

{context}

======================================================
QUESTION
======================================================

{query}

======================================================
ANSWER
======================================================
"""

        return prompt


if __name__ == "__main__":

    builder = PromptBuilder()

    docs = [

        {
            "text": "Retrieval-Augmented Generation combines retrieval with generation.",
            "source": "RAG.pdf",
            "page": 5,
            "section": "Introduction",
        },

        {
            "text": "Dense retrieval uses embedding similarity.",
            "source": "Dense Retrieval.pdf",
            "page": 2,
            "section": "Methods",
        },

    ]

    prompt = builder.build(
        query="What is Retrieval Augmented Generation?",
        documents=docs,
    )

    print(prompt)