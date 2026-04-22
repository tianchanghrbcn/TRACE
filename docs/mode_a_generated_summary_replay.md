# Mode A: Generated Summary Workbook Replay

This document describes the first generated layer of Mode A paper-exact replay.

The previous Mode A archive replay selects and archives paper-exact outputs from legacy repositories. This step starts rebuilding generated summary workbooks from archived analysis CSV files.

## Run

First ensure the paper-exact archive exists:

    python scripts/trace.py mode-a --audit --clean

Then run:

    python scripts/50_audit_paper_table_scripts.py
    python scripts/51_build_paper_summary_workbooks.py
    python scripts/52_validate_paper_summary_workbooks.py

## Outputs

Generated outputs are placed under:

    analysis/paper_generated/

Important files:

- `analysis/paper_generated/table_script_audit.csv`
- `analysis/paper_generated/table_script_audit.json`
- `analysis/paper_generated/table_script_audit.md`
- `analysis/paper_generated/summary_workbooks/beers_summary.xlsx`
- `analysis/paper_generated/summary_workbooks/flights_summary.xlsx`
- `analysis/paper_generated/summary_workbooks/hospital_summary.xlsx`
- `analysis/paper_generated/summary_workbooks/rayyan_summary.xlsx`
- `analysis/paper_generated/summary_workbooks/paper_summary_index.xlsx`
- `analysis/paper_generated/generated_summary_manifest.json`
- `analysis/paper_generated/generated_summary_validation_report.json`
- `analysis/paper_generated/generated_summary_validation_report.md`

## Scope

This step validates that summary workbooks can be regenerated from archived analysis CSV files.

It does not yet claim byte-identical reproduction of every paper table. The next step will migrate the selected table scripts, including `Tab.8_cal.py`, `Tab.9_cal.py`, `Tab.11-12_cal.py`, and the `6.1.*_cal_*.py` scripts.

## Unified entry

Run archive replay and generated summary workbook replay together:

    python scripts/trace.py mode-a --audit --clean --generated-summaries
