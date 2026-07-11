from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(text):
    """
    Create a FAISS vector database from the extracted medical report.
    """

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    return vectorstore
