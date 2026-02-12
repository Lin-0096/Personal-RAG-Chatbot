# File: /support-copilot-rag/support-copilot-rag/src/support_copilot/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Anthropic-compatible (used by Minimax gateway)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")

    # Legacy / unused in the current minimal build (kept for compatibility)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./.chroma")
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "./knowledge")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    # For Minimax M2.1 (Anthropic-compatible), set: LLM_MODEL=m2.1
    LLM_MODEL = os.getenv("LLM_MODEL", "m2.1")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 150))