from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_deploy_workflow_serializes_runs() -> None:
    workflow = (ROOT / ".github" / "workflows" / "production.yml").read_text(encoding="utf-8")

    assert "concurrency:" in workflow
    assert "group: production-deploy" in workflow
    assert "cancel-in-progress: false" in workflow
