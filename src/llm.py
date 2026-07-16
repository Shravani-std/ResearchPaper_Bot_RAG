from __future__ import annotations

from typing import Any

from core.config import settings

try:
    from google import genai
except Exception:  # pragma: no cover - environment fallback
    genai = None


class GeminiLLM:
    def __init__(self) -> None:
        self.model = None
        if settings.GEMINI_API_KEY and genai is not None:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                self.model = None

    def build_prompt(self, context: str, question: str) -> str:
            return f"""
        You are an expert AI research assistant specializing in scientific papers.

        Your task is to answer the user's question ONLY using the retrieved context.

        =========================
        INSTRUCTIONS
        =========================

        1. Read all retrieved context carefully.

        2. Answer in your own words.

        3. NEVER copy long paragraphs from the context.

        4. If multiple context chunks contain relevant information,
        combine them into one coherent answer.

        5. Ignore unrelated retrieved context.

        6. Never invent information that is not present.

        7. If the answer is only partially available,
        clearly mention what is known and what is missing.

        8. If the answer cannot be found in the context, reply exactly:

        "I could not find the answer in the retrieved context."

        =========================
        SPECIAL RULES
        =========================

        If the question asks for:

        • Summary
            - Summarize instead of copying.
            - Mention:
                • Problem
                • Proposed Method
                • Main Contributions
                • Conclusion

        • Explanation
            - Explain step by step.
            - Use simple language first.
            - Then explain technical details.

        • Definition
            - Give a clear definition.
            - Explain why it is important.

        • Equation
            - Write the equation.
            - Explain every variable.
            - Explain what the equation means.
            - Explain why it is introduced.
            - Explain how it is used later.

        • Comparison
            - Use a markdown table.

        =========================
        ANSWER FORMAT
        =========================

        ## Answer

        ## Explanation

        ## Key Points

        ## Conclusion

        =========================
        RETRIEVED CONTEXT
        =========================

        {context}

        =========================
        QUESTION
        =========================

        {question}

        Answer:
        """






    def generate_answer(self, context: str, question: str) -> str:
        if not context.strip():
            return "I couldn't find that information."

        if self.model is None:
            return self._fallback_answer(context, question)

        try:
            prompt = self.build_prompt(context, question)
            response = self.model.generate_content(prompt)
            return getattr(response, "text", "") or self._fallback_answer(context, question)
        except Exception:
            return self._fallback_answer(context, question)

    @staticmethod
    def _fallback_answer(context: str, question: str) -> str:
        summary = " ".join(context.split())
        if len(summary) > 800:
            summary = summary[:800] + "..."
        return f"Based on the available context, the answer is: {summary}"