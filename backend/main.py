import json
import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ollama import AsyncClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ollama_client = AsyncClient(host="http://localhost:11434")

server_params = StdioServerParameters(
    command="/home/user/Python/software-engineering/learn-llm/backend/venv/bin/python3",
    args=["-m", "mcp_server.mcp_server"],
    env=None
)

async def event_generator(messages: list):
    user_query = messages[-1]["content"]
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available_tools = await session.list_tools()
            
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in available_tools.tools
            ]

            yield json.dumps({"status": "Searching Knowledge Base..."}) + "\n"
            rag_result = await session.call_tool("search_knowledge_base", {"query": user_query})
            rag_context = rag_result.content[0].text if rag_result.content else "No local info."

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

            yield json.dumps({"status": "Analyzing Request..."}) + "\n"
            response = await ollama_client.chat(
                model="llama3.1",
                messages=[{"role": "system", "content": system_prompt}] + messages,
                tools=ollama_tools
            )

            if response.get("message", {}).get("tool_calls"):
                messages.append(response["message"])
                
                for tool_call in response["message"]["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = tool_call["function"]["arguments"]
                
                    if name == "search_knowledge_base":
                        continue

                    yield json.dumps({"status": f"Executing {name}..."}) + "\n"
                    mcp_result = await session.call_tool(name, args)
                    result_text = mcp_result.content[0].text
                    
                    messages.append({"role": "tool", "content": result_text, "name": name})

            yield json.dumps({"status": "Finalizing..."}) + "\n"
            final_stream = await ollama_client.chat(
                model="llama3.1",
                messages=[{"role": "system", "content": system_prompt}] + messages,
                stream=True
            )

            async for chunk in final_stream:
                if chunk.get("message", {}).get("content"):
                    yield json.dumps(chunk.model_dump()) + "\n"

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    return StreamingResponse(event_generator(data.get("messages", [])), media_type="application/x-ndjson")

@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    file_path = os.path.join("./shared_downloads", file_id)
    from fastapi.responses import FileResponse
    return FileResponse(file_path) if os.path.exists(file_path) else {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30001)