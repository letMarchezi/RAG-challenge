import os
import streamlit as st
from routers import get_available_models, upload_documents, ask_question

st.set_page_config(page_title="RAG System", layout="wide")

st.title("Document Q&A System")

# Initialize session state for storing upload status
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False

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
            else:
                st.session_state.documents_uploaded = True
                st.success(
                    f"Processed {result.get('processed',0)} files, skipped {result.get('skipped',0)}. Total chunks: {result.get('total_chunks',0)}"
                )

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

question = st.text_input("Enter your question about the documents")

if question:
    if not st.session_state.documents_uploaded:
        st.warning("Please upload and process a document first!")
    else:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                result = ask_question(question, llm_provider, model)
                if result.get("error"):
                    st.error(f"Request failed: {result['error']}")
                    if result.get("status_code"):
                        st.code(f"Status: {result['status_code']}\nBody: {result.get('text','')}")
                else:
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    if result.get("references"):
                        with st.expander(f"References"):
                            st.write(result["references"])