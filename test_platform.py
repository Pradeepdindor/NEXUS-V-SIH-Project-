"""
Smart India Hackathon 2026 - Problem Statement 26124 / 24124
Root Test Suite Runner Shim (test_platform.py -> scripts/test_platform.py)
"""

import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.test_platform import run_all_tests

if __name__ == "__main__":
    run_all_tests()
