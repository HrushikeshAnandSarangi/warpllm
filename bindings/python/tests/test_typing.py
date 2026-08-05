"""Runs mypy over the package and `typing_probe.py`.

The generated TypedDicts do nothing at runtime, so no other test in this suite
can tell a correct annotation from a wrong one -- `test_chat.py` passes just as
happily against `dict[str, Any]`. This is the only thing that reads them, and
it is why regenerating from a changed Rust type is checked rather than trusted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_generated_types_say_what_they_should():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--warn-unused-ignores",
            "--no-error-summary",
            str(ROOT / "python" / "warpllm"),
            str(Path(__file__).parent / "typing_probe.py"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
