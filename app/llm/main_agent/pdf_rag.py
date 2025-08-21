from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from app.llm.rag.rag import load_and_index_documents
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS_INDEX_PATH = "app/llm/main_agent/atlas_collection/index.faiss"
db_path = Path(__file__).resolve().parent.parent / "rag" / "atlas"
FAISS_INDEX_PATH = str(db_path)
# KNOWLEDGE_BASE_PATH = "app/data/atlas_collection.pdf"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "atlas_collection.pdf"



price_manuals_vector_store = load_and_index_documents(KNOWLEDGE_BASE_PATH, FAISS_INDEX_PATH)

@tool
def get_information_from_documents(query: str) -> str:
    """
    Always use this tool for searching any knowladge for user's question
    """
    if price_manuals_vector_store is None:
        return "База не инициализирована. Не могу выполнить поиск."
    
    # print(f"RAG Tool: Поиск информации по запросу: '{query}'")
    try:
        # Получаем релевантные документы (например, 3 наиболее подходящих)
        retriever = price_manuals_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5,
                                                                           "fetch_k": 100,
                                                                           "lambda_mult": 1
                                                                           })
        relevant_docs = retriever.invoke(query) 
        
        if not relevant_docs:
            # print("RAG Tool: Релевантные документы не найдены.")
            return "По вашему запросу информация в документах не найдена."

        # Формируем контекст из найденных документов
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        # print(f"RAG Tool: Найденный контекст:\n{context[:500]}...") # Логируем часть контекста
        return f"Вот информация, найденная в документах по вашему запросу:\n{context}"
    except Exception as e:
        # print(f"RAG Tool: Ошибка при поиске: {e}")
        return f"Произошла ошибка при поиске информации в документах: {e}"

# db_path = Path(__file__).resolve().parent.parent / "rag" / "atlas"
# db_name = str(db_path)
# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# new_vector_store = FAISS.load_local(db_name, embeddings_model, allow_dangerous_deserialization=True)
# print(new_vector_store.index_to_docstore_id)