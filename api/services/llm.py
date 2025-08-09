import logging
import os
import re
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMService:
    def __init__(self, llm_provider: str = "openai", model: str | None = None):
        """Initialize LLM service with specified provider

        Args:
            llm_provider: The LLM provider to use ('openai', 'gemini')
            model: Optional model name override to use for the given provider
        """
        self.provider = llm_provider
        self.model = model
        self.llm = self._get_llm(llm_provider)

        self.prompt_template = PromptTemplate(
            template=(
                "You are a helpful assistant. Use ONLY the provided context to answer.\n"
                "If you did not find exactly the answer on the context, summarize the information you found."
                "If the answer is not in the context, say you cannot answer from the context.\n\n"
                "Return a STRICT JSON object with keys: \n"
                "- answer (string) \n- references (string with supporting excerpt text) \n- citations (array of objects with keys: document_id, page, score, snippet).\n"
                "Do not include markdown, code fences, or extra text.\n\n"
                "Context (each item may include document_id and page metadata):\n{context}\n\n"
                "Question: {question}\n"
            ),
            input_variables=["context", "question"],
        )

    def _get_llm(self, provider: str):
        """Get LLM instance based on provider

        Args:
            provider: LLM provider name
        """

        if provider == "openai":
            return ChatOpenAI(
                model=self.model or "gpt-4.1-mini",
                temperature=0,
            )
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model or "gemini-2.0-flash-lite",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _safe_parse_json(self, content: str) -> Dict:
        """Parse JSON from the model content, with minimal cleanup."""
        import json

        text = content.strip()
        # Remove common fence wrappers if provider adds them
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            logging.warning("Falling back to plain answer due to JSON parse error")
            return {"answer": text, "references": ""}

    def generate_answer(self, question: str, relevant_docs: List[str]) -> Dict:
        """Generate answer based on question and relevant documents

        Args:
            question: User's question
            relevant_docs: List of relevant document contents

        Returns:
            Dict containing answer and sources
        """
        if not relevant_docs:
            return {
                "answer": "No relevant context found in the documents. Please try a different question or upload relevant documents.",
                "references": "",
            }

        # If we received structured retrieval results, extract text for context
        if relevant_docs and isinstance(relevant_docs[0], dict):
            # Include minimal metadata with each snippet so the model can cite it
            def format_context(d: Dict) -> str:
                doc_id = d.get("document_id")
                page = d.get("page")
                score = d.get("score")
                snippet = d.get("snippet", "")
                return f"[doc={doc_id} page={page} score={score}]\n{snippet}"

            context = "\n\n".join([format_context(d) for d in relevant_docs])
        else:
            context = "\n\n".join(relevant_docs)

        # Generate prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Get response from LLM
        response = self.llm.invoke(prompt)
        parsed = self._safe_parse_json(response.content)
        logging.info(
            "[LLMService] parsed keys=%s answer_len=%s refs_len=%s citations=%s",
            list(parsed.keys()),
            (
                len(parsed.get("answer", ""))
                if isinstance(parsed.get("answer"), str)
                else None
            ),
            (
                len(parsed.get("references", ""))
                if isinstance(parsed.get("references"), str)
                else None
            ),
            (
                len(parsed.get("citations", []))
                if isinstance(parsed.get("citations"), list)
                else 0
            ),
        )
        # Ensure citations exist; if model omitted them, pass through retrieval items
        if isinstance(relevant_docs[0], dict):
            parsed.setdefault("citations", relevant_docs)

        logging.info("[LLMService] parsed=%s", parsed)
        return parsed

