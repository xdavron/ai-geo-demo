import os
from pprint import pprint

from langchain_community.vectorstores import FAISS
import faiss

from langchain_openai import OpenAIEmbeddings
from app.llm.rag.local_server_embeddings import LocalServerEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from pypdf import PdfReader, PdfWriter
import tempfile
from pathlib import Path

embeddings_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe", model_kwargs={"trust_remote_code": True})

# Путь для сохранения/загрузки индекса FAISS (чтобы не пересоздавать каждый раз)

def load_and_index_documents(docs_path: Path, index_path: str):
    """
    Загружает документы, создает эмбеддинги, индексирует их и сохраняет индекс FAISS.
    Если индекс уже существует, загружает его.
    """
    if os.path.exists(index_path):
        print(f"Загрузка существующего индекса FAISS из: {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        print(f"Создание нового индекса FAISS. Загрузка документов из: {docs_path}")
        # Загрузчик для .txt файлов
        # loader = DirectoryLoader(
        #     docs_path,
        #     glob="**/*.txt",
        #     loader_cls=TextLoader,
        #     show_progress=True,
        #     loader_kwargs={"encoding": "utf-8"}
        # )
        # documents = loader.load()

        password = "5555555"  # <-- put your PDF password here

        reader = PdfReader(docs_path)
        if reader.is_encrypted:
            reader.decrypt(password)

        # Create a temporary decrypted PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.write(tmp)
            decrypted_path = tmp.name

        # Load the decrypted PDF
        loader = PyMuPDFLoader(decrypted_path)
        documents = loader.load()

        # pprint(documents[0:20])

        if not documents:
            print(f"Документы не найдены в {docs_path}. Убедитесь, что путь и файлы корректны.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents[28:])
        #
        print(f"Создание векторного хранилища из {len(split_documents)} чанков...")

        ##### New Version
        single_vector = embeddings_model.embed_query("test data")
        print(len(single_vector))
        index = faiss.IndexFlatL2(len(single_vector))
        print(index.ntotal, index.d)

        vector_store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        ids = vector_store.add_documents(documents=split_documents)

        db_name = 'atlas'
        vector_store.save_local(db_name)

        # new_vector_store = FAISS.load_local(db_name, embeddings_model, allow_dangerous_deserialization=True)
        # print(new_vector_store.index_to_docstore_id)

        # Testing
        question = "что знаешь о Ходжасаят "
        # docs = vector_store.search(query=question, search_type="similarity")
        # for doc in docs:
        #     print(doc.page_content)
        #     print("\n\n")

        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5,
                                                                           "fetch_k": 100,
                                                                           "lambda_mult": 1
                                                                           })
        docs = retriever.invoke(question)
        for doc in docs:
            print(doc.page_content)
            print("\n\n")



        # vector_store = FAISS.from_documents(split_documents, embeddings_model)
        # vector_store = FAISS.from_texts([t.page_content for t in split_documents], embeddings_model)
        # vector_store.save_local(index_path)
        # print(f"Индекс FAISS сохранен в: {index_path}")
    return vector_store

if __name__ == "__main__":
    FAISS_INDEX_PATH = "aaa/atlas"
    # KNOWLEDGE_BASE_PATH = "app/data/atlas_collection.pdf"
    KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "atlas_collection.pdf"



    price_manuals_vector_store = load_and_index_documents(KNOWLEDGE_BASE_PATH, FAISS_INDEX_PATH)