import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_timeline():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js 18+ is required for the frontend timeline tests.")
    version = subprocess.run(
        [node, "-p", "process.versions.node"], capture_output=True, text=True, check=True, timeout=5
    ).stdout.strip()
    if int(version.split(".", 1)[0]) < 18:
        pytest.skip("Node.js 18+ is required for the frontend timeline tests.")
    test_file = Path(__file__).with_name("labeler_frontend.test.cjs")
    result = subprocess.run(
        [node, "--test", str(test_file)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
