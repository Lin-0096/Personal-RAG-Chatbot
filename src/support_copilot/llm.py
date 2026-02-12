from __future__ import annotations

from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import Config


ContextAnswerer = Callable[[str, str], str]


SYSTEM_PROMPT = """You are a precise question-answering assistant.

Rules:
- ONLY use the provided "Context" to answer questions.
- If the context is insufficient, clearly state "Cannot confirm from the provided materials" and specify what additional information is needed.
- Do NOT make up information or extend beyond the context.
- Answer in ENGLISH only.
- **IMPORTANT:** If the context contains documentation links (URLs starting with https://docs.verda.com/), ALWAYS include them in your answer. These links provide detailed information users can reference.
- Structure your response as follows:
  1. If needed, briefly analyze the question and context (keep this brief)
  2. Then provide a clear "ANSWER:" section that can be directly copied
  3. If documentation links are available in the context, include them prominently in the answer

Format example:
[Brief analysis if needed]

ANSWER:
[Your clear, concise answer here]
[Use bullet points or numbered steps for procedures]
[Include relevant documentation links from context prominently]

**For more details, see:**
[List all relevant https://docs.verda.com/ URLs found in the context]
""".strip()


def build_minimax_chat_answerer_from_env() -> Optional[ContextAnswerer]:
    """Build an LLM answerer using an Anthropic-compatible endpoint.

    This is designed to work with Minimax's Anthropic gateway by setting:
    - ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
    - ANTHROPIC_API_KEY=...
    - LLM_MODEL=m2.1

    Returns None if required env vars are not present.
    """

    if not Config.ANTHROPIC_API_KEY:
        return None

    # Comprehensive compatibility shim for LangChain 1.x
    try:  # pragma: no cover
        import langchain  # type: ignore

        for attr, default in [
            ("verbose", False),
            ("debug", False),
            ("llm_cache", None),
            ("cache", None),
        ]:
            if not hasattr(langchain, attr):
                setattr(langchain, attr, default)
    except Exception:
        pass

    # ChatAnthropic reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL from env.
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for Anthropic-compatible LLM. Install: pip install langchain-anthropic"
        ) from e

    chat = ChatAnthropic(
        model=Config.LLM_MODEL,
        temperature=0,
        max_tokens=Config.MAX_TOKENS,
    )

    def answer(question: str, context: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Question:\n"
                    f"{question}\n\n"
                    "Context:\n"
                    f"{context}\n"
                )
            ),
        ]
        resp = chat.invoke(messages)
        # Handle both string and list content formats
        content = resp.content
        if isinstance(content, list):
            # Minimax may return list of content blocks
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts).strip()
        return (content or "").strip()

    return answer
