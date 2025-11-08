import os
from dotenv import load_dotenv
load_dotenv()

import nltk
nltk.download("punkt")

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------------------------------
# FUNCTION 1: LOAD PDF → SPLIT → STORE IN CHROMA
# -------------------------------------------

def create_vector_db(docs_path, vector_db_path, collection_name):

    print("Loading PDFs...")
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=UnstructuredPDFLoader
    )

    documents = loader.load()
    print("Total PDFs Loaded:", len(documents))

    # Text split
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    emb = HuggingFaceEmbeddings()

    # Create / update vector DB
    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=vector_db_path,
        collection_name=collection_name
    )

    print("✔ Vector DB created/updated successfully!")


# -------------------------------------------
# FUNCTION 2: ASK QUESTION USING LCEL RAG CHAIN
# -------------------------------------------

def query_rag(user_query, db_path, collection_name):

    emb = HuggingFaceEmbeddings()

    # Load vector DB
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=db_path
    )

    retriever = vector_store.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )

    # Prompt for RAG
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer only using the provided context."),
        ("human", "Question: {query}\n\nContext:\n{context}")
    ])

    # LCEL Chain (official LangChain 1.x method)
    chain = (
    {
        "query": RunnablePassthrough(),
        "context": RunnablePassthrough() 
                   | (lambda x: x["query"]) 
                   | retriever
    }
    | prompt
    | llm
    | StrOutputParser()
)


    print("\nGenerating answer...\n")
    answer = chain.invoke({"query": user_query})

    print("✔ Answer:")
    print(answer)
    print("\n------------------------------------------------------------")


# -------------------------------------------
# RUN THE PIPELINE
# -------------------------------------------

if __name__ == "__main__":

    # 1. Create vector store
    create_vector_db(
        docs_path=r"D:\RAG\Docs",
        vector_db_path=r"D:\RAG\DB",
        collection_name="collection_name"
    )

    # 2. Ask questions
    user_question = input("\nEnter your question: ")
    query_rag(
        user_query=user_question,
        db_path=r"D:\RAG\DB",
        collection_name="collection_name"
    )
