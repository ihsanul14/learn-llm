import os
import logging
from mcp.server.fastmcp import FastMCP
from docx import Document
from langchain_community.tools import DuckDuckGoSearchRun
from rag.rag import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

mcp = FastMCP("ReliableEngineer-Tools")
rag = RAGService()
search_tool = DuckDuckGoSearchRun()

SHARED_DOWNLOADS = "./shared_downloads"
os.makedirs(SHARED_DOWNLOADS, exist_ok=True)

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    Search local files (PDF, DOCX, TXT) for specific technical information.
    Use this for any specific terms, proper nouns, or internal codebase questions.
    """
    logger.info(f"Querying RAG for: {query}")
    try:
        result = rag.query(query)
        if not result or len(result.strip()) < 10:
            logger.warning(f"RAG returned no content for: {query}")
            return "RESULT: No information found in local knowledge base. Proceed to web_search if applicable."
            
        return result
    except Exception as e:
        logger.error(f"RAG Error: {str(e)}")
        return f"ERROR: Retrieval failed with message: {str(e)}"

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for real-time information ONLY if local knowledge is insufficient."""
    logger.info(f"Executing Web Search: {query}")
    return search_tool.run(query)

@mcp.tool()
def create_file_for_download(filename: str, content: str) -> str:
    """Create a document on the server. Returns a download link."""
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
        
        logger.info(f"File created successfully: {filename}")
        return f"SUCCESS: File created. Link: http://localhost:30001/api/download/{filename}"
    except Exception as e:
        logger.error(f"File Creation Error: {str(e)}")
        return f"ERROR: {str(e)}"

@mcp.resource("knowledge://status")
def get_kb_status() -> str:
    """Returns the current state of the knowledge base."""
    count = rag.collection.count()
    return f"Knowledge Base contains {count} vectors."

@mcp.tool()
def refresh_knowledge_base() -> str:
    """Triggers a rescan of the local knowledge folder to find new files."""
    logger.info("Syncing RAG service...")
    return rag.sync()

if __name__ == "__main__":
    mcp.run(transport="stdio")