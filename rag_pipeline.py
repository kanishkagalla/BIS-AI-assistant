from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import VECTOR_DB_PATH, LLM_MODEL
from utils import format_answer
from langchain.chains import RetrievalQA

def get_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=LLM_MODEL)

    # RetrievalQA is still available in langchain==0.1.20
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa

def answer_query(query: str):
    qa = get_rag_pipeline()
    result = qa.invoke(query)
    if isinstance(result, dict) and "result" in result:
        return format_answer(result["result"])
    return format_answer(str(result))
