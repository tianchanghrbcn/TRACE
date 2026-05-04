"""Paper artifact rebuild utilities.

This package provides reviewer-facing builders for paper figures and tables.
The maintained entry points are:

- scripts/50_build_all_paper_figures.py
- scripts/51_build_all_paper_tables.py

Individual figure/table builders live under src/paper_artifact/figures and
src/paper_artifact/tables. Each builder exposes an ARTIFACT dictionary and a
build(ctx) function.
"""

__all__ = ["io", "registry"]
