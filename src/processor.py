import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def initialize_rag(data_path="data", persist_directory="./chroma_db"):

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "pdf_chunks"
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

        loader = DirectoryLoader(data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"doc_{i}"],
                metadatas=[{"source": chunk.metadata.get("source"), "page": chunk.metadata.get("page", "?")}],
                documents=[chunk.page_content]
            )

    generator = pipeline(
        "text-generation",
        model = "google/flan-t5-small",
        device=-1,
        max_new_tokens=256,
        temperature=0.1
    )

    return collection, embedder, generator


def query_rag(collection, embedder, generator, query, k=2):
    query_embedding = embedder.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    unique_docs = list(dict.fromkeys(results['documents'][0]))
    context = " ".join(unique_docs)
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {query}"

    answer = generator(prompt)[0]["generated_text"]
    return answer, results['metadatas'][0]