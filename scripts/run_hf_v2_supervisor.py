from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


DATA_PATH = Path("data/site-data.json")
TOTAL_PROMPTS = 10


def completed() -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return sum(
        1
        for pair in data.get("prompt_pairs", [])
        if (pair.get("model_generations", {}).get("hf-v2") or {}).get("status") == "ok"
    )


def main() -> int:
    for attempt in range(1, 30):
        before = completed()
        print(f"[supervisor] attempt={attempt} completed_before={before}/{TOTAL_PROMPTS}", flush=True)
        if before >= TOTAL_PROMPTS:
            return 0

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/generate_hf_v2_prompt_pairs.py",
                "--batch-size",
                "1",
                "--max-new-tokens",
                "8192",
            ]
        )
        after = completed()
        print(
            f"[supervisor] attempt={attempt} returncode={proc.returncode} "
            f"completed_after={after}/{TOTAL_PROMPTS}",
            flush=True,
        )
        if after >= TOTAL_PROMPTS:
            return 0
        if after <= before and proc.returncode != 0:
            return proc.returncode
        time.sleep(5)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
