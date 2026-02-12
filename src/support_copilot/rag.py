from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

try:
    # Newer LangChain splits Chroma into a separate package.
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

from .embeddings import create_embeddings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float


def _chunk_text(text: str, *, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    cleaned = (text or "").replace("\r\n", "\n")
    if len(cleaned) <= chunk_size:
        return [cleaned]

    if overlap < 0 or overlap >= chunk_size:
        overlap = 0

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end]
        chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def load_text_documents(directory: str) -> List[Document]:
    base = Path(directory)
    if not base.exists():
        return []

    documents: List[Document] = []
    for file_path in sorted(base.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": str(file_path), "chunk": idx},
                )
            )
    return documents


class RAG:
    """Minimal, readable, offline-first RAG.

    - Vector store: Chroma (local)
    - Embeddings: LocalHashEmbeddings (deterministic, offline)
    - Answering: extractive (returns best chunk + short snippet)

    You can later swap in a real LLM (OpenAI/local) to generate nicer answers.
    """

    def __init__(
        self,
        *,
        persist_directory: str,
        collection_name: str = "knowledge_base",
        k: int = 4,
        llm_answerer: Optional[Callable[[str, str], str]] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.k = k
        self._llm_answerer = llm_answerer
        self._embeddings = create_embeddings()
        self._vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self._embeddings,
        )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        blocks: List[str] = []
        for i, c in enumerate(chunks[: self.k]):
            src = Path(c.source).name if c.source else ""
            header = f"[{i + 1}] {src}" if src else f"[{i + 1}]"
            blocks.append(f"{header}\n{c.text.strip()}")
        return "\n\n---\n\n".join(blocks)

    def ingest(self, documents: Iterable[Document]) -> int:
        docs = list(documents)
        if not docs:
            return 0
        self._vs.add_documents(docs)
        # Older chroma/langchain versions require explicit persist.
        if hasattr(self._vs, "persist"):
            self._vs.persist()
        return len(docs)

    def ingest_directory(self, knowledge_dir: str) -> int:
        return self.ingest(load_text_documents(knowledge_dir))

    def retrieve(self, query: str, *, k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = k or self.k
        results = self._vs.similarity_search_with_score(query, k=top_k)
        chunks: List[RetrievedChunk] = []
        for doc, score in results:
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=str(doc.metadata.get("source", "")),
                    score=float(score),
                )
            )
        return chunks

    def answer(self, question: str) -> Tuple[str, List[str]]:
        chunks = self.retrieve(question)
        if not chunks:
            return (
                "我在知识库里没有检索到足够信息来回答。你可以提供更多上下文或补充相关文档吗？",
                [],
            )

        # Prefer LLM answer when configured.
        if self._llm_answerer is not None:
            try:
                context = self._format_context(chunks)
                llm_answer = (self._llm_answerer(question, context) or "").strip()
                if llm_answer:
                    answer_text = llm_answer
                else:
                    answer_text = "无法从提供的资料中确认。请补充更多信息或文档。"
            except Exception as e:
                # Log error and fallback to offline extractive mode
                import sys
                print(f"\n⚠️  LLM 调用失败: {e}", file=sys.stderr)
                print("   自动切换到离线抽取式回答\n", file=sys.stderr)
                answer_text = ""
        else:
            answer_text = ""

        if not answer_text:
            best = chunks[0]
            snippet = best.text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            answer_text = (
                "我根据知识库中最相关的内容给出回答：\n\n"
                f"{snippet}\n\n"
                "如果你希望我用更自然的语言总结/改写，我可以在下一步接入 LLM 来生成更好的答复。"
            )
        sources: List[str] = []
        seen = set()
        for c in chunks:
            if not c.source:
                continue
            if c.source in seen:
                continue
            seen.add(c.source)
            sources.append(c.source)
        return answer_text, sources