from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.embedding_sdk import MockHashEmbeddingProvider, embed_text, resolve_provider


class EmbeddingSdkTest(unittest.TestCase):
    def test_mock_is_deterministic_and_768d(self) -> None:
        provider = MockHashEmbeddingProvider()
        left = provider.embed("복도에서 쓰러진 사람")
        right = provider.embed("복도에서 쓰러진 사람")
        self.assertEqual(len(left), 768)
        self.assertEqual(left, right)

    def test_resolve_default_mock(self) -> None:
        provider = resolve_provider("mock")
        result = embed_text("fall hallway", provider=provider)
        self.assertEqual(result.model, "mock-hash-768")
        self.assertEqual(result.dimension, 768)

    def test_process_embed_cli_stdout_json(self) -> None:
        script = ROOT / "scripts" / "process_embed.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--text", "쓰러짐 복도"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout.strip())
        self.assertEqual(payload["dimension"], 768)
        self.assertEqual(len(payload["embedding"]), 768)


if __name__ == "__main__":
    unittest.main()
