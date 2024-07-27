import streamlit as st
import os,bs4
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# load the Groq API key
groq_api_key = "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"

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    print("Embeddings Has been Genrated ....")
    st.session_state.loader = WebBaseLoader("https://www.tradingview.com/symbols/NSE-HDFCBANK/",
                                            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("tv-category-content"))))
    print("Loader has been Loaded ....")
    st.session_state.docs = st.session_state.loader.load()
    print("Loaded successfully ......")

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    print("Text Splitted successfully.....")
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )
    print("Vectorization has done ..")
st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input you prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time :", time.process_time() - start)
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")