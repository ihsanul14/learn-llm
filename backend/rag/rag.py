import os
import uuid
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGService:
    def __init__(self, db_path="./chroma_db", knowledge_dir="./knowledge"):
        self.knowledge_dir = knowledge_dir
        self.ollama_ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text", 
            url="http://localhost:11434/api/embeddings"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="universal_knowledge", 
            embedding_function=self.ollama_ef
        )

    def query(self, text: str, n_results: int = 3):
        """Retrieve relevant context from the vector store."""
        try:
            results = self.collection.query(query_texts=[text], n_results=n_results)
            return "\n\n".join(results["documents"][0]) if results["documents"] else "No context found."
        except Exception as e:
            return f"Query Error: {str(e)}"

    def sync(self):
        """Standardizes the ingestion logic."""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            
        loader = DirectoryLoader(self.knowledge_dir, glob="**/*.*")
        docs = loader.load()
        if not docs: return "Knowledge directory is empty."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        ids = [f"id_{uuid.uuid4().hex[:6]}" for _ in range(len(chunks))]
        self.collection.add(
            documents=[c.page_content for c in chunks],
            metadatas=[{"source": c.metadata["source"]} for c in chunks],
            ids=ids
        )
        return f"Successfully indexed {len(chunks)} new chunks."