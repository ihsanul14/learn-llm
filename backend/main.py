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

PYTHON_PATH = os.getenv("MCP_PYTHON_PATH", "python3")

server_params = StdioServerParameters(
    command=PYTHON_PATH,
    args=["-m", "mcp_server.mcp_server"],
    env=None
)

async def event_generator(messages: list):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            available_tools = await session.list_tools()
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                } for t in available_tools.tools
            ]

            system_prompt = (
                "You are a helpful assistant. "
                "Use tools ONLY if the user asks for technical information, files, or web searches. "
                "For casual greetings (hi, hello, etc.), reply normally with a brief greeting. "
                "Output ONLY the final result. NO internal reasoning."
            )

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
                
                    yield json.dumps({"status": f"Invoking {name}..."}) + "\n"
                    mcp_result = await session.call_tool(name, args)
                    messages.append({"role": "tool", "content": mcp_result.content[0].text, "name": name})

                final_stream = await ollama_client.chat(
                    model="llama3.1",
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    stream=True
                )
                async for chunk in final_stream:
                    if chunk.get("message", {}).get("content"):
                        yield json.dumps(chunk.model_dump()) + "\n"
            else:
                yield json.dumps(response.model_dump()) + "\n"

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