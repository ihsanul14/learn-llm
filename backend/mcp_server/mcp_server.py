import os
import asyncio
import logging
import sys
import mcp.types as types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from langchain_community.tools import DuckDuckGoSearchRun
import uvicorn

# Setup Logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp-sse-server")

server = Server("agent-power-tools")
SHARED_STORAGE = "./shared_downloads"
os.makedirs(SHARED_STORAGE, exist_ok=True)

AVAILABLE_TOOLS = [
    types.Tool(
        name="web_search",
        description="Search the internet.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    types.Tool(
        name="create_file_for_download",
        description="Creates a file on the server.",
        inputSchema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    ),
]

@server.list_tools()
async def handle_list_tools():
    return AVAILABLE_TOOLS

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None):
    if name == "web_search":
        query = arguments.get("query")
        search = DuckDuckGoSearchRun()
        result = await asyncio.to_thread(search.run, query)
        print("web_serach: ", result)
        return [types.TextContent(type="text", text=result)]
    elif name == "create_file_for_download":
        filename = arguments.get("filename")
        content = arguments.get("content")
        file_path = os.path.join(SHARED_STORAGE, filename)
        print("file: ", file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return [types.TextContent(type="text", text=f"FILE_CREATED_SUCCESSFULLY: {filename}")]
    raise ValueError(f"Unknown tool: {name}")

sse = SseServerTransport("/messages")

async def app(scope, receive, send):
    """
    Direct ASGI Application. 
    Bypasses Starlette routing to avoid 'NoneType' errors.
    """
    if scope["type"] == "http":
        path = scope["path"]
        if path == "/sse":
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options()
                )
        elif path == "/messages":
            await sse.handle_post_message(scope, receive, send)
        else:
            # Standard 404 for anything else
            await send({
                'type': 'http.response.start',
                'status': 404,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'Not Found',
            })

if __name__ == "__main__":
    logger.info("🚀 Standalone MCP SSE Server live at http://localhost:30002")
    uvicorn.run(app, host="0.0.0.0", port=30002)