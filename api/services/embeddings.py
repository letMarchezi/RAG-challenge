import logging
from typing import List, Dict
import re
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
import os

class OpenAIEmbeddingsDirect(LangChainEmbeddings):
    """LangChain-compatible embeddings wrapper using the official OpenAI SDK.

    Provides embed_documents and embed_query methods required by LangChain
    vector stores while delegating to OpenAI's embeddings API.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.client = OpenAI()
        self.model = model

    def _sanitize_text(self, text: str) -> str:
        # Cleans the text to remove null characters and strip whitespace
        if text is None:
            logging.warning("Text is None")
            return " "
        cleaned = str(text).replace("\x00", "").strip()
        if not cleaned:
            cleaned = ""
            logging.warning("Text after cleaning is empty")
        return cleaned

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned_texts = [self._sanitize_text(t) for t in texts]
        all_embeddings: List[List[float]] = []
        batch_size = 64
        # Send the text to the embeddings model in batches of 64
        for start in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # Embeds a single query string
        response = self.client.embeddings.create(
            model=self.model, input=self._sanitize_text(text)
        )
        return response.data[0].embedding


class EmbeddingsService:
    def __init__(self):
        """Service responsible for generating and persisting embeddings using FAISS."""

        # Where to persist the FAISS index
        self.index_path = "./vector_store"
        os.makedirs(self.index_path, exist_ok=True)

        # Embedding model via official OpenAI SDK
        embedding_model_name = "text-embedding-3-small"
        self.embeddings = OpenAIEmbeddingsDirect(model=embedding_model_name)

        # Try loading an existing FAISS index if present
        if any(fname.endswith(".faiss") for fname in os.listdir(self.index_path)):
            # allow_dangerous_deserialization=True is required when running inside Docker/
            # constrained environments where pickle safety checks are strict.
            logging.info(f"[EmbeddingsService] Loading existing FAISS index from {self.index_path}")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vector_store = None

        # Document splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=300,
            length_function=len,
        )
    
    def process_pdf(self, file_content: bytes, filename: str) -> Dict:
        """Process one PDF file and persist its own FAISS index under a file-specific folder.
        
        Args:
            file_content: Bytes content of the PDF file
            
        Returns:
            Dict containing processing statistics
        """
        # Derive a deterministic folder name from the incoming filename
        stem = os.path.splitext(os.path.basename(filename))[0]
        stem = stem.strip().lower()
        stem = re.sub(r"[^a-z0-9._-]+", "_", stem) or "document"
        doc_dir = os.path.join(self.index_path, stem)

        # If this document was already processed, skip
        existing_faiss = os.path.join(doc_dir, "index.faiss")
        existing_meta = os.path.join(doc_dir, "index.pkl")
        if os.path.exists(existing_faiss) and os.path.exists(existing_meta):
            return {
                "message": "Document already indexed; skipping",
                "skipped": True,
                "document_id": stem,
                "index_path": doc_dir,
                "documents_indexed": 0,
                "total_chunks": 0,
            }

        # Read PDF bytes with PyPDF2
        reader = PdfReader(BytesIO(file_content))
        docs: List[Document] = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            docs.append(Document(page_content=text, metadata={"page": idx + 1}))

        # Split and filter empty
        chunks = self.text_splitter.split_documents(docs)
        chunks = [d for d in chunks if d.page_content and d.page_content.strip()]

        # DEBUG: visualize chunks
        logging.info(f"[EmbeddingsService] Total pages loaded: {len(docs)}")
        logging.info(f"[EmbeddingsService] Total non-empty chunks: {len(chunks)}")
        for i, d in enumerate(chunks):
            preview = d.page_content[:500]
            logging.info(f"\n[Chunk {i}] length={len(d.page_content)}\n{preview}\n---")

        # If nothing extracted, return early
        if not chunks:
            return {
                "documents_indexed": len(docs),
                "total_chunks": 0,
                "index_path": self.index_path,
            }

        # Store in a dedicated FAISS index for this document
        local_store = FAISS.from_documents(chunks, self.embeddings)
        os.makedirs(doc_dir, exist_ok=True)
        local_store.save_local(doc_dir)

        return {
            "message": "Document processed successfully",
            "skipped": False,
            "document_id": stem,
            "documents_indexed": len(docs),
            "total_chunks": len(chunks),
            "index_path": doc_dir,
        }
    
    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Search across all stored FAISS indexes for similar chunks
        
        Args:
            query: Search query string
            k: Number of documents to return
            
        Returns:
            List of relevant document contents
        """
        # Iterate over subdirectories in the vector_store and aggregate results
        if not os.path.isdir(self.index_path):
            return []

        aggregated: List = []
        for entry in os.listdir(self.index_path):
            doc_dir = os.path.join(self.index_path, entry)
            if not os.path.isdir(doc_dir):
                continue
            faiss_path = os.path.join(doc_dir, "index.faiss")
            meta_path = os.path.join(doc_dir, "index.pkl")
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                continue
            try:
                store = FAISS.load_local(
                    doc_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                # Get the similarity search with scores
                docs_with_scores = store.similarity_search_with_score(query, k=k)
                aggregated.extend(docs_with_scores)
            except Exception as exc:
                logging.error(f"[EmbeddingsService] Failed loading index at {doc_dir}: {exc}")

        # # Normalize scores to [0,1] and get global top-k
        # def _normalize(score: float) -> float:
        #     try:
        #         s = float(score)
        #     except Exception:
        #         return 0.0
        #     # If score >= 0 assume it's a distance (smaller is better)
        #     if s >= 0:
        #         return 1.0 / (1.0 + s)
        #     # Otherwise assume cosine similarity in [-1, 1]
        #     return max(0.0, min(1.0, (s + 1.0) / 2.0))

        aggregated.sort(key=lambda pair: pair[1])  # ascending
        top_k = aggregated[:k]
        return [doc.page_content for doc, _ in top_k]
