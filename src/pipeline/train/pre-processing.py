#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper.

The implementation has moved to `src.pipeline.preprocess`.
This wrapper is kept temporarily while the TRACE pipeline is being refactored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.preprocess import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
