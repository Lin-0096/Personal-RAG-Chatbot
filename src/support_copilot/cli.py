from __future__ import annotations

import argparse
from pathlib import Path
import sys

def _ensure_src_on_path() -> None:
    # Allows running without installing the package (src-layout).
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    src_dir = project_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    from support_copilot.config import Config
    from support_copilot.llm import build_minimax_chat_answerer_from_env
    from support_copilot.rag import RAG

    parser = argparse.ArgumentParser(description="Support Copilot minimal RAG")
    parser.add_argument("--ask", type=str, default=None, help="Ask one question and exit")
    parser.add_argument("--rebuild-index", action="store_true", help="Clear and rebuild Chroma index")
    args = parser.parse_args(argv)

    knowledge_dir = Path(Config.KNOWLEDGE_DIR)
    persist_dir = Path(Config.CHROMA_DB_PATH)
    persist_dir.mkdir(parents=True, exist_ok=True)

    llm_answerer = build_minimax_chat_answerer_from_env()
    
    if args.rebuild_index:
        print("🔄 清空并重建索引...")
        import shutil
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
    
    rag = RAG(persist_directory=str(persist_dir), llm_answerer=llm_answerer)
    if knowledge_dir.exists():
        count = rag.ingest_directory(str(knowledge_dir))
        if args.rebuild_index:
            print(f"✅ 已索引 {count} 个文档块")

    # Show LLM mode status
    mode_str = "🤖 LLM 生成式回答已启用 (Minimax m2.1)" if llm_answerer else "📄 离线抽取式回答 (无 LLM)"
    if not args.ask:
        print(f"Support Copilot (minimal RAG). {mode_str}")
        print("输入问题，输入 'exit' 退出。")

    if args.ask:
        answer, sources = rag.answer(args.ask)
        print(answer)
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"- {s}")
        return

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer, sources = rag.answer(question)
        print("\nAnswer:\n")
        print(answer)
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"- {s}")
        print()


if __name__ == "__main__":
    main()