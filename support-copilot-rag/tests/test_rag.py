import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from support_copilot.rag import RAG


class TestRAG(unittest.TestCase):
    def test_ingest_retrieve_and_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_dir = Path(tmp) / "kb"
            kb_dir.mkdir(parents=True, exist_ok=True)

            (kb_dir / "france.txt").write_text(
                "France: The capital of France is Paris.",
                encoding="utf-8",
            )
            (kb_dir / "germany.txt").write_text(
                "Germany: The capital of Germany is Berlin.",
                encoding="utf-8",
            )

            db_dir = Path(tmp) / "chroma"
            rag = RAG(persist_directory=str(db_dir))
            ingested = rag.ingest_directory(str(kb_dir))
            self.assertEqual(ingested, 2)

            chunks = rag.retrieve("What is the capital of France?")
            self.assertTrue(chunks)
            self.assertIn("Paris", chunks[0].text)

            answer, sources = rag.answer("What is the capital of France?")
            self.assertIn("Paris", answer)
            self.assertTrue(any("france.txt" in s for s in sources))


if __name__ == "__main__":
    unittest.main()