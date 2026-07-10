#!/usr/bin/env python3
"""CLI: mock/fake-timer backend benchmark (no GPU required for mock mode).

Real TensorRT timing must be run on a GPU PC; this CLI never claims TRT speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.backend_instrumentation import run_mock_backend_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backend start-log + metrics benchmark (mock by default).")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "backend_benchmark")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fake-inference-ms", type=float, default=8.0)
    parser.add_argument("--requested-backend", default="tensorrt")
    args = parser.parse_args(argv)

    result = run_mock_backend_benchmark(
        output_root=args.output_root,
        iterations=args.iterations,
        fake_inference_ms=args.fake_inference_ms,
        requested_backend=args.requested_backend,
    )
    print(result["metrics"]["disclaimer"], file=sys.stderr)
    print(
        json.dumps(
            {
                "runId": result["runId"],
                "outputDir": result["outputDir"],
                "actual_backend": result["metrics"]["start"]["actual_backend"],
                "disclaimer": result["metrics"]["disclaimer"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
