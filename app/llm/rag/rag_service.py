from pathlib import Path
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader, PdfWriter
import tempfile

class RAGService:
    """
    Build a simple RAG pipeline with LangGraph.
    """

    def __init__(self) -> None:
        # Load PDFs into vector store on startup
        self.store = self._build_vector_store()

    def _load_documents(self, folder: Path):
        """Read all PDF files and return text."""
        password = "5555555" # <-- put your PDF password here

        reader = PdfReader(str(folder))
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
        loader = PyPDFLoader(decrypted_path)
        docs = loader.load()

        # Clean up the temporary file
        os.remove(decrypted_path)

        return docs

    def _build_vector_store(self):
        """Create FAISS index from PDFs."""
        pdf_folder = Path(__file__).resolve().parent.parent.parent / "data" / "atlas_collection.pdf"

        docs = self._load_documents(pdf_folder)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        # embeddings = LocalServerEmbeddings(base_url="http://localhost:1234/v1")

        vector_store = FAISS.from_texts([t.page_content for t in texts], embeddings)
        return vector_store

    def _retrieve(self, query: str, k=5):
        """Retrieve top‑k documents from the vector store."""
        docs = self.store.similarity_search(query, k=k)
        print(docs)
        return docs

    def generate_answer(
        self,
        user_message: str,
    ) -> str:
        """
        1. Retrieve relevant context.
        2. Send prompt + context to LLM.
        """
        context = self._retrieve(user_message)
        return context

rag = RAGService()
print(rag.generate_answer("Hello World!"))