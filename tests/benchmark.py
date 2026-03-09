"""Compatibility wrapper for the moved block benchmark entry."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)

from run_block_benchmark import main


if __name__ == "__main__":
    main()
