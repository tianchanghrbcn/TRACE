"""Template for adding a paper figure builder.

Copy this file to a non-underscore module name, e.g. fig03_edr_gain.py.
Modules whose names start with '_' are ignored by the registry.
"""

from pathlib import Path
from src.paper_artifact.io import ArtifactResult, BuildContext

ARTIFACT = {
    "id": "figXX_short_name",
    "paper_id": "Figure X",
    "label": "Short figure label",
    "description": "One sentence explaining what this builder generates.",
}


def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = ctx.output_dir / "figXX_short_name"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read source files under ctx.input_root or ctx.project_root.
    # Generate PDF/PNG outputs under out_dir.
    # Return all outputs and important inputs.
    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="skipped",
        outputs=[],
        inputs=[],
        message="Template only; replace with actual figure code.",
    )
