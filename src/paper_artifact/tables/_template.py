"""Template for adding a paper table builder.

Copy this file to a non-underscore module name, e.g. table03_main_results.py.
Modules whose names start with '_' are ignored by the registry.
"""

from src.paper_artifact.io import ArtifactResult, BuildContext

ARTIFACT = {
    "id": "tableXX_short_name",
    "paper_id": "Table X",
    "label": "Short table label",
    "description": "One sentence explaining what this builder generates.",
}


def build(ctx: BuildContext) -> ArtifactResult:
    out_dir = ctx.output_dir / "tableXX_short_name"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate .tex and .csv outputs under out_dir.
    return ArtifactResult(
        artifact_id=ARTIFACT["id"],
        status="skipped",
        outputs=[],
        inputs=[],
        message="Template only; replace with actual table code.",
    )
