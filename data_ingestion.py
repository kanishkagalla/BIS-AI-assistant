import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(data_folder="data"):
    docs = []
    for file in os.listdir(data_folder):
        path = os.path.join(data_folder, file)
        try:
            if file.endswith(".pdf"):
                loader = UnstructuredPDFLoader(path)
                docs.extend(loader.load())
            elif file.endswith(".txt"):
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())
                except Exception:
                    # fallback for structured/multilingual text
                    from langchain_community.document_loaders import UnstructuredFileLoader
                    loader = UnstructuredFileLoader(path)
                    docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
    return docs


def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    print("✅ Vector store built and saved at:", VECTOR_DB_PATH)

if __name__ == "__main__":
    documents = load_documents("data")
    if documents:
        print(f"📄 Loaded {len(documents)} documents.")
        build_vector_store(documents)
    else:
        print("⚠️ No valid documents found in 'data/' folder.")
