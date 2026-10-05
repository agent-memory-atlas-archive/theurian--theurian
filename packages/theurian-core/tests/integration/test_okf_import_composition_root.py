"""What ``okf_commands._service`` wires, driven by ``theurian okf import`` (GHSA-v2qg-23fc-7fqp).

``test_okf_import.py`` builds its own ``OkfImportService``, so deleting a lookup from the
CLI's composition root left every one of those tests green. These drive the real command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from label_inheritance_support import (
    ITEM_ID,
    cli,
    create_only_project,
    labelled_project,
    proposals_tree,
)

from theurian.cli import migration_pipeline, okf_commands
from theurian.domain import migration as domain_migration

pytestmark = pytest.mark.integration

_OPTIONS: Final = (
    "--owner",
    "platform-team",
    "--author",
    "dana@example.com",
    "--agent-id",
    "claude-code",
    "--task-id",
    "task-okf",
    "--model",
    "claude-opus-5",
    "--reasoning",
    "Importing a bundle a teammate shared.",
)


def _bundle(root: Path, item_id: str) -> Path:
    """One concept naming ``item_id`` and asking for ``internal`` / ``backend`` / ``reviewed``."""
    bundle = root.parent / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "concept.md").write_text(
        f"---\ntype: architecture\ntitle: Imported concept\nstatus: stable\n"
        f"theurian_export_version: 1\ntheurian_item_id: {item_id}\n"
        f"theurian_namespace: backend\ntheurian_trust_level: reviewed\n"
        f"theurian_sensitivity: internal\ntheurian_content_type: text/markdown\n---\n\nBody.\n",
        encoding="utf-8",
    )
    return bundle


@pytest.fixture
def replays(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []
    real = migration_pipeline.current_item_in

    def spy(loaded: Any, item_id: Any, **kwargs: Any) -> Any:
        asked.append(item_id.value)
        return real(loaded, item_id, **kwargs)

    monkeypatch.setattr(okf_commands, "current_item_in", spy)
    return asked


@pytest.fixture
def revision_lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []
    real = domain_migration.current_revision_in

    def spy(migrations: Any, item_id: Any) -> Any:
        asked.append(item_id.value)
        return real(migrations, item_id)

    monkeypatch.setattr(okf_commands, "current_revision_in", spy)
    return asked


def test_a_concept_for_a_create_only_item_is_staged_with_the_items_labels_not_the_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replays: list[str]
) -> None:
    """Without the replay lookup the import drafts a create-only id as new and re-labels it.

    The bundle asks for ``internal`` / ``backend``; the item holds ``confidential`` /
    ``security``. Trust is ``inferred`` because the import fixes it there by design.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    bundle = _bundle(project.root, project.item_id)

    code, payload = cli("okf", "import", str(bundle), *_OPTIONS)

    assert code == 0, payload
    assert payload["conceptsAdmitted"] == 1, payload
    assert replays == [project.item_id]
    [proposal_id] = payload["proposalIds"]
    [migration] = (project.root / ".theurian/proposals" / proposal_id).glob("*.yaml")
    operations = yaml.safe_load(migration.read_text(encoding="utf-8"))["operations"]
    metadata = next(op for op in operations if op["op"] == "upsertRevision")["metadata"]
    assert (metadata["sensitivity"], metadata["trustLevel"], metadata["namespace"]) == (
        "confidential",
        "inferred",
        "security",
    )


def test_a_concept_for_an_item_with_a_revision_is_refused_and_replays_only_to_check_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replays: list[str],
    revision_lookups: list[str],
) -> None:
    """The import never names ``expectedRevision``; the cheap check refuses it.

    The one replay is the refusal path asking whether the item is retired, whose
    remedy differs; a draft that succeeds makes none.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    bundle = _bundle(project.root, ITEM_ID)
    before = proposals_tree(project.root)

    code, payload = cli("okf", "import", str(bundle), *_OPTIONS)

    assert code == 0, payload
    assert payload["conceptsAdmitted"] == 0, payload
    assert [(r["kind"], r["key"]) for r in payload["refusals"]] == [("draft", ITEM_ID)]
    assert proposals_tree(project.root) == before
    assert revision_lookups == [ITEM_ID]
    assert replays == [ITEM_ID]
