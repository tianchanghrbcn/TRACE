#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper.

The implementation has moved to `src.pipeline.clustering_runner`.
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

from src.pipeline.clustering_runner import ClusterMethod, run_clustering

__all__ = ["ClusterMethod", "run_clustering"]
