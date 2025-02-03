import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

st.title("Chat with Webpage 🌐")
st.caption("This app allows you to chat with a webpage using local llama2 and RAGe")

# Get the webpage URL from the user
webpage_url = st.text_input("Enter Webpage URL", type="default")

if webpage_url:
    # 1. Load the data
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    # 2. Create Ollama embeddings and vector store with Qdrant
    ollama_host = "http://host.docker.internal:11434"
    embeddings = OllamaEmbeddings(model="llama2",base_url = ollama_host)
    qdrant_client = QdrantClient("http://host.docker.internal:6333")  # Ensure Qdrant is running locally
    collection_name = "web_docs"

    # Ensure collection exists before using it
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE)  # Ensure correct embedding size
    )

    # Initialize vectorstore with collection
    vectorstore = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)

    # Add documents to the collection
    vectorstore.add_documents(splits)

    # 3. Call Ollama 2 model
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    # 4. RAG Setup
    retriever = vectorstore.as_retriever()

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)

    st.success(f"Loaded {webpage_url} successfully!")

    # Ask a question about the webpage
    prompt = st.text_input("Ask any question about the webpage")

    # Chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
