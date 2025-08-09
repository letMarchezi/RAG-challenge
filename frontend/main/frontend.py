import os
import streamlit as st
from routers import get_available_models, upload_documents, ask_question
import logging
import time
st.set_page_config(page_title="RAG System")

# Minimal CSS: font, title color, button color, and input background
st.markdown(
    """
<style>
/* Global font */
html, body, [data-testid="stAppViewContainer"] {
  font-family: "Helvetica Neue", Helvetica, Arial;
}

/* Title and header colors */
.block-container { padding-top: 1rem; }
h1, .stMarkdown h1 { color: #334155; font-weight: 700; margin-top: 0.25rem; }
h2, .stMarkdown h2, h3, .stMarkdown h3 { color: #2563EB; }

/* Buttons: blue background, white text */
.stButton > button {
  background-color: #2563EB !important;
  color: #FFFFFF !important;
  border: none;
  border-radius: 8px;
  padding: 0.5rem 1rem;
}
.stButton > button:hover { background-color: #1D4ED8 !important; }

/* Inputs: darker background and blue outline on focus */
:root { --field-bg: #E5E9F2; /* darker than before */ }
.stTextInput > div > div,
.stTextArea > div > div,
.stSelectbox > div > div,
[data-testid="stFileUploadDropzone"] {
  background-color: var(--field-bg) !important;
  border: 1px solid #CBD5E1; /* slate-300 */
  border-radius: 8px;
}

/* Blue edges on focus/hover */
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within,
.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus-within,
[data-testid="stFileUploadDropzone"]:hover {
  border-color: #2563EB !important;
  box-shadow: 0 0 0 1px #2563EB33;
}

/* Ensure the file uploader's "Browse files" button is blue */
[data-testid="stFileUploadDropzone"] button {
  background-color: #2563EB !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 6px;
}

/* Keep inner inputs transparent so the container color shows */
.stTextInput input, .stTextArea textarea { background: transparent !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Document Q&A System")
st.caption("Upload the desired Pdf files, process to generate embeddings, select the LLM provider and model, and ask questions about the documents.")

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
                            
st.caption("Developed by Leticia Bossatto Marchezi")