import os
import streamlit as st
from routers import get_available_models, upload_documents, ask_question
import logging
import time
st.set_page_config(page_title="RAG System", layout="wide")

st.title("Document Q&A System:")
st.caption("Upload the desired Pdf files, process to generate embeddings, select the LLM provider and model, and ask questions about the documents.")
st.caption("Developed by Leticia Bossatto Marchezi")
# Initialize session state for storing upload status
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False
if 'current_doc_ids' not in st.session_state:
    st.session_state.current_doc_ids = []

# File upload section
st.header("1. Upload Documents")
uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            payload = [(f.name, f.getvalue()) for f in uploaded_files]
            result = upload_documents(payload)
            if result.get("error"):
                st.error(f"Upload failed: {result['error']}")
                if result.get("status_code"):
                    st.code(f"Status: {result['status_code']}\nBody: {result.get('text','')}")
                logging.error(f"Upload failed: {result['error']}")
            else:
                st.session_state.documents_uploaded = True
                # Capture the document ids (folder stems) returned per file
                st.session_state.current_doc_ids = [
                    r.get("document_id")
                    for r in result.get("results", [])
                    if r.get("document_id")
                ]
                st.success(
                    f"Succesfully processed {result.get('processed',0)} files, skipped {result.get('skipped',0)}. Total chunks processed: {result.get('total_chunks',0)}"
                )
                if st.session_state.current_doc_ids:
                    st.caption(
                        f"Retrieval will search only: {', '.join(st.session_state.current_doc_ids)}"
                    )
                    logging.info(f"Documents uploaded successfully: {st.session_state.current_doc_ids}")

# Question answering section
st.header("2. Ask Questions")

# LLM provider selection
llm_provider = st.selectbox(
    "Select LLM Provider",
    ["openai", "gemini"],
    help="Choose which LLM provider to use for answering questions"
)

model_options = get_available_models()[llm_provider]
model = st.selectbox(
    "Select Model",
    model_options,
    index=0,
    help="Model inferred from environment variables for the chosen provider",
)

question = st.text_input("Enter your question about the above documents")

if question:
    if not st.session_state.documents_uploaded:
        st.warning("Please upload and process a document first!")
    else:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                logging.info(f"Generating answer with the model: {model} for question: {question}")
                start_time = time.perf_counter()
                result = ask_question(
                    question,
                    llm_provider,
                    model,
                    st.session_state.current_doc_ids,
                )
                elapsed_s = time.perf_counter() - start_time
                if result.get("error"):
                    st.error(f"Request failed: {result['error']}")
                    logging.error(f"Request failed: {result['error']}")
                    if result.get("status_code"):
                        st.code(f"Status: {result['status_code']}\nBody: {result.get('text','')}")
                else:
                    logging.info(f"Answer succesfuly generated: {result['answer']}")
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    st.caption(f"Response time: {elapsed_s:.2f} s")
                    if result.get("references"):
                        with st.expander(f"References"):
                            st.write(result["references"])