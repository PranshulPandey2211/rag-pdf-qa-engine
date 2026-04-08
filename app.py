import streamlit as st
import time
from src.processor import initialize_rag, query_rag

st.set_page_config(page_title="Offline PDF Q&A", layout="wide")
st.title("Local PDF Q&A")

if "collection" not in st.session_state:
    with st.spinner("Initializing embeddings and vector store..."):
        st.session_state.collection, st.session_state.embedder, st.session_state.generator = initialize_rag()

query = st.text_input("Ask a question about the PDFs:")

if query:
    start_time = time.time()
    with st.spinner("Searching and generating answer..."):
        answer, context_meta = query_rag(
            st.session_state.collection,
            st.session_state.embedder,
            st.session_state.generator,
            query
        )
    elapsed = time.time() - start_time

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Context Used")
    st.write(context_meta)

    st.info(f"Query time: {elapsed:.2f} seconds")