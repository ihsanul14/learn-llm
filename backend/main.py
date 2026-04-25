import os
import json
import asyncio
import uuid
import chromadb
from docx import Document
from ollama import AsyncClient
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.tools import DuckDuckGoSearchRun
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- ADDED IMPORTS FOR INGESTION ---
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- INITIALIZATION ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SHARED_DOWNLOADS = "./shared_downloads"
KNOWLEDGE_DIR = "./knowledge"
os.makedirs(SHARED_DOWNLOADS, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

ollama_client = AsyncClient(host="http://localhost:11434")
search_tool = DuckDuckGoSearchRun()

# --- CHROMADB PERSISTENCE ---
DB_PATH = os.path.join(os.getcwd(), "chroma_db")
ollama_ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text", 
    url="http://localhost:11434/api/embeddings"
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Global collection variable
collection = None

# --- THE RESTORED INGESTION ENGINE ---
def ingest_anything(directory_path: str):
    global collection
    print(f"📂 Scanning directory: {directory_path}...")
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Loads everything: pdf, txt, docx, etc.
    loader = DirectoryLoader(directory_path, glob="**/*.*", show_progress=True)
    docs = loader.load()
    
    if not docs:
        print("Empty directory. Knowledge base is idle.")
        # Ensure collection exists even if empty
        collection = chroma_client.get_or_create_collection(
            name="universal_knowledge", 
            embedding_function=ollama_ef
        )
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    documents = [c.page_content for c in chunks]
    metadatas = [{"source": c.metadata["source"]} for c in chunks]
    ids = [f"id_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]

    # Wipe old data to ensure a fresh sync on restart
    try:
        chroma_client.delete_collection("universal_knowledge")
        print("🗑️ Cleared old index for fresh sync.")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="universal_knowledge", 
        embedding_function=ollama_ef
    )

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"🚀 Indexed {len(documents)} chunks from local files.")

# RUN INGESTION ON STARTUP
ingest_anything(KNOWLEDGE_DIR)

# --- TOOLS & API (Remains the same but uses the global 'collection') ---

def execute_create_file(filename: str, content: str) -> str:
    file_path = os.path.join(SHARED_DOWNLOADS, filename)
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".docx":
            doc = Document()
            doc.add_paragraph(content)
            doc.save(file_path)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return f"SUCCESS: File created. Link: http://localhost:30001/api/download/{filename}"
    except Exception as e:
        return f"ERROR: {str(e)}"

@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    file_path = os.path.join(SHARED_DOWNLOADS, file_id)
    return FileResponse(file_path, filename=file_id) if os.path.exists(file_path) else {"error": "File not found"}

async def event_generator(messages: list):
    user_query = messages[-1]["content"]
    
    yield json.dumps({"status": "Searching Knowledge Base..."}) + "\n"
    
    try:
        # Search the global collection
        results = collection.query(query_texts=[user_query], n_results=3)
        rag_context = "\n\n".join(results["documents"][0]) if results["documents"] else "No local info."
    except:
        rag_context = "RAG error."

    system_prompt = (
        f"## LOCAL KNOWLEDGE CONTEXT:\n{rag_context}\n\n"
        "## OPERATIONAL SEQUENCE:\n"
        "1. RAG-FIRST: Always analyze the LOCAL KNOWLEDGE CONTEXT first. If the answer is there, provide it immediately. DO NOT use tools if the context is sufficient.\n"
        "2. SEARCH: If the context is missing, irrelevant, or the user asks for 'live' info, use the 'web_search' tool. Do not ask for permission; just search.\n"
        "3. FILE-ACTION: If the user asks to save, download, or create a document (like .docx or .txt), you MUST use 'create_file_for_download'. You have direct server access; DO NOT ask for cloud permissions or 'Proceed?'. Just call the tool.\n\n"
        "## OUTPUT FORMAT:\n"
        "- Provide the direct answer first.\n"
        "- If you created a file, provide the link exactly as: http://localhost:30001/api/download/FILENAME\n"
        "- DO NOT mention proxy servers, cloud storage, or external permissions."
    )

    tools = [
        {"type": "function", "function": {"name": "web_search", "description": "Search web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "create_file_for_download", "description": "Create files.", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}}}
    ]

    try:
        yield json.dumps({"status": "Thinking..."}) + "\n"
        # Using llama3.1 as per your code
        response = await ollama_client.chat(model="llama3.1", messages=[{"role": "system", "content": system_prompt}] + messages, tools=tools)

        if response.get("message", {}).get("tool_calls"):
            messages.append(response["message"])
            for tool in response["message"]["tool_calls"]:
                name = tool["function"]["name"]
                args = tool["function"]["arguments"]
                yield json.dumps({"status": f"Running {name}..."}) + "\n"
                if name == "web_search":
                    print("Executing web search for query")
                    result = await asyncio.to_thread(search_tool.run, args["query"])
                elif name == "create_file_for_download":
                    print("Executing file creation for query")
                    result = execute_create_file(args["filename"], args["content"])
                messages.append({"role": "tool", "content": result, "name": name})

        yield json.dumps({"status": "Finalizing..."}) + "\n"
        final_stream = await ollama_client.chat(model="llama3.1", messages=[{"role": "system", "content": system_prompt}] + messages, stream=True)
        async for chunk in final_stream:
            if chunk.get("message", {}).get("content"):
                yield json.dumps(chunk.model_dump()) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    return StreamingResponse(event_generator(data.get("messages", [])), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30001)