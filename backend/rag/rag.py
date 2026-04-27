import os
import uuid
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGService:
    def __init__(self, db_path="./chroma_db", knowledge_dir="./rag/knowledge"):
        self.knowledge_dir = os.path.abspath(knowledge_dir)
        self.ollama_ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text", 
            url="http://localhost:11434/api/embeddings"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="universal_knowledge", 
            embedding_function=self.ollama_ef,
            metadata={"hnsw:space": "cosine"} 
        )

    def query(self, text: str, n_results: int = 5):
        try:
            results = self.collection.query(query_texts=[text], n_results=n_results)
            if not results["documents"] or not results["documents"][0]:
                return ""
            return "\n\n".join(results["documents"][0])
        except Exception as e:
            return f"Query Error: {str(e)}"

    def sync(self):
        """Hardened ingestion logic with mapping to prevent missing files."""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            
        count_before = self.collection.count()
        if count_before > 0:
            self.client.delete_collection("universal_knowledge")
            self.collection = self.client.create_collection(
                name="universal_knowledge", 
                embedding_function=self.ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )

        loaders = {
            ".txt": DirectoryLoader(self.knowledge_dir, glob="**/*.txt", loader_cls=TextLoader),
            ".pdf": DirectoryLoader(self.knowledge_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".docx": DirectoryLoader(self.knowledge_dir, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
        }

        all_docs = []
        for ext, loader in loaders.items():
            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {ext} files: {e}")

        if not all_docs:
            return "Sync failed: No readable files found in directory."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(all_docs)

        documents = [c.page_content for c in chunks]
        metadatas = [{"source": c.metadata.get("source", "unknown")} for c in chunks]
        ids = [f"id_{i}_{uuid.uuid4().hex[:4]}" for i in range(len(chunks))]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        return f"Sync Complete. Indexed {len(all_docs)} files into {len(chunks)} chunks."