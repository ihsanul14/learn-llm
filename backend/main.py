import os
import json
import asyncio
import ollama
import chromadb

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings",
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="universal_knowledge", 
    embedding_function=ollama_ef
)

def ingest_anything(directory_path: str):
    global collection
    print(f"📂 Scanning directory: {directory_path}...")
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    loader = DirectoryLoader(directory_path, glob="**/*.*", show_progress=True)
    docs = loader.load()
    
    if not docs:
        print("Empty directory. Waiting for files...")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    documents = [c.page_content for c in chunks]
    metadatas = [{"source": c.metadata["source"]} for c in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    try:
        chroma_client.delete_collection("universal_knowledge")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="universal_knowledge", 
        embedding_function=ollama_ef
    )

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"🚀 Indexed {len(documents)} chunks.")

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    model = data.get("model", "llama3.2")
    messages = data.get("messages", [])
    user_query = messages[-1]["content"]

    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=3
        )

        relevant_context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

        if relevant_context:
            rag_system_prompt = {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the following context retrieved from "
                    "local files to answer the user's question. If the answer is not in the "
                    "context, tell the user you don't know based on the files.\n\n"
                    f"CONTEXT FROM FILES:\n{relevant_context}"
                )
            }
            messages.insert(0, rag_system_prompt)
            print(f"✅ Context injected from: {results['metadatas'][0]}")

    except Exception as e:
        print(f"❌ RAG Search Error: {e}")

    async def event_generator():
        try:
            response = ollama.chat(model=model, messages=messages, stream=True)
            for chunk in response:
                yield json.dumps(chunk.model_dump()) + "\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    ingest_anything("./knowledge")
    uvicorn.run(app, host="0.0.0.0", port=30001)