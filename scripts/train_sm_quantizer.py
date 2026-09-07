"""Thin entry point; implementation and configuration live in sm_quantizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dicodec.modules.sm_quantizer.training import main


if __name__ == "__main__":
    main()
