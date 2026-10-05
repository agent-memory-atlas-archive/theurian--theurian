"""Two guards GHSA-v2qg-23fc-7fqp added to ``ProposalService.draft``, each driven on its own.

``ProposalRequest.sensitivity_is_default`` is what lets a candidate for an item
that exists but has no revision inherit that item's sensitivity instead of
naming its own ``internal`` and being refused. The ``expectedRevision`` check
that runs before ``current_item`` is what keeps a missing or stale token from
paying for a replay of the landed set. Neither had a test that fails when the
guard goes: the first is a branch only a candidate for a create-only item reaches, and the
second changes cost and not the answer.

The MCP half of the flag -- ``knowledge.proposeChange`` never sets it, so a caller
naming ``internal`` on a confidential visible item is still refused -- is held by
``test_update_label_inheritance.py``'s
``test_an_mcp_update_naming_a_lower_sensitivity_is_refused_without_naming_a_label``
and ``test_create_only_item_labels.py``'s
``test_an_mcp_draft_naming_a_lower_sensitivity_for_a_create_only_item_is_refused``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
import review_candidate_fixtures as corpus
from label_inheritance_support import (
    ITEM_ID,
    ROOT_REVISION_ID,
    cli_ok,
    cli_propose,
    item_row,
    labelled_project,
    land,
    proposals_tree,
    serving_grant,
    staged_metadata,
)
from review_candidate_project import ServedProject, served_project

from theurian.cli import migration_pipeline, propose_commands
from theurian.daemon.runner import build_server

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration

CANDIDATE_ITEM_ID: Final = "reliability.retry-lock-order"
_CREATE_ONLY_MIGRATION_ID: Final = "01K1DDDDDD01234567890ABCDE"
_STALE_REVISION: Final = "01K1BBBREV01234567890ABCDE"


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ServedProject]:
    yield from served_project(
        tmp_path, monkeypatch, records=corpus.evidence_records(), withheld=frozenset()
    )


def _candidate(served: ServedProject, tmp_path: Path, item_id: str) -> dict[str, Any]:
    """The tool's payload, called under a ceiling that lets the caller see a confidential item.

    Under the default ceiling the item is withheld and answers as an absent id, which
    would draft ``internal`` for the wrong reason.
    """
    grant = serving_grant(tmp_path / "datadir", "confidential")
    arguments: dict[str, Any] = {
        "projectId": "demo",
        "repository": corpus.REPOSITORY,
        "recordKey": corpus.THREAD_SATISFYING,
        "fixCommit": served.verifying,
        "itemId": item_id,
        "title": "Acquire locks after reads in retry-eligible paths",
        "body": "Acquire locks after reads, never before, in retry-eligible paths.\n",
        "kind": "convention",
        "category": "reliability-rule",
        "owner": "platform-team",
        "author": "dana@example.com",
        "description": "Generalise the deadlock thread into a locking rule",
        "evidence": {
            "agentId": "claude-code",
            "taskId": "task-7",
            "model": "claude-opus-5",
            "reasoning": "The thread settled the lock ordering.",
        },
        "sourceAnchors": [
            {
                "provider": "github",
                "sourceUri": (
                    f"https://github.com/{corpus.REPOSITORY}/pull/"
                    f"{corpus.PULL_REQUEST_CI_PASSED}#discussion_r1"
                ),
                "repository": corpus.REPOSITORY,
                "filePath": corpus.FILE_PATH,
            }
        ],
    }
    with mcp_session(build_server(served.registry, grant), tmp_path / "wire") as call:
        answer = call("review.generateKnowledgeCandidate", arguments)

    result: dict[str, Any] = answer["result"]
    assert result["isError"] is False, result
    payload: dict[str, Any] = result["structuredContent"]
    return payload


def _land_create_only_item(root: Path, item_id: str, sensitivity: str) -> None:
    """A root migration with a ``createItem`` and no revision, applied."""
    (root / f".theurian/migrations/{_CREATE_ONLY_MIGRATION_ID}-create-only.yaml").write_text(
        f"""apiVersion: theurian.dev/v1
id: {_CREATE_ONLY_MIGRATION_ID}
createdAt: 2026-08-02T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {item_id}
    kind: convention
    namespace: reliability
    owner: platform-team
    sensitivity: {sensitivity}
"""
    )
    cli_ok("migrate", "apply")


def test_a_candidate_for_a_confidential_item_with_no_revision_inherits_its_sensitivity(
    served: ServedProject, tmp_path: Path
) -> None:
    """The candidate's ``internal`` is a default, not a statement, once the item exists.

    Without ``sensitivity_is_default`` the drafter would read ``internal`` as the
    caller naming a level below ``confidential`` and refuse the candidate, or, were
    the floor absent, re-label the item on its first revision (GHSA-v2qg-23fc-7fqp).
    """
    _land_create_only_item(served.root, CANDIDATE_ITEM_ID, "confidential")
    before = item_row(served.root, CANDIDATE_ITEM_ID)
    assert (before["sensitivity"], before["current_revision_id"]) == ("confidential", None), (
        "the item must exist, confidential, with no revision"
    )

    drafted = _candidate(served, tmp_path, CANDIDATE_ITEM_ID)

    assert staged_metadata(served.root, drafted)["sensitivity"] == "confidential"

    land(drafted["proposalId"])

    row = item_row(served.root, CANDIDATE_ITEM_ID)
    assert row["current_revision_id"] == drafted["revisionId"], "the revision must have landed"
    assert row["sensitivity"] == "confidential"


def test_a_candidate_for_a_brand_new_id_still_drafts_internal(
    served: ServedProject, tmp_path: Path
) -> None:
    """Nothing exists to inherit, so the flag changes nothing for a first revision.

    The project also holds a confidential create-only item under another id, so a
    drafter that cleared the default for every id fails here as well.
    """
    _land_create_only_item(served.root, "reliability.some-other-item", "confidential")

    drafted = _candidate(served, tmp_path, CANDIDATE_ITEM_ID)

    assert staged_metadata(served.root, drafted)["sensitivity"] == "internal"


# -- the revision check precedes the replay ---------------------------------------


@pytest.fixture
def replays(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The item ids ``current_item_in`` -- the replay -- was asked about by ``theurian propose``."""
    asked: list[str] = []
    real = migration_pipeline.current_item_in

    def spy(loaded: Any, item_id: Any, **kwargs: Any) -> Any:
        asked.append(item_id.value)
        return real(loaded, item_id, **kwargs)

    monkeypatch.setattr(propose_commands, "current_item_in", spy)  # the name `_service` calls
    return asked


def test_an_update_without_an_expected_revision_is_refused_and_replays_only_for_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replays: list[str]
) -> None:
    """The refusal needs only the item's current revision; the one replay follows it.

    The replay builds a throwaway store from the whole landed set (0.286 s at 1,000
    items). It runs after the cheap check has refused, to tell a retired item, whose
    remedy is readmission, from one that wants the token; a draft that succeeds
    makes no replay beyond its own.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, ITEM_ID)

    assert code != 0, payload
    assert "--expected-revision" in str(payload), payload
    assert proposals_tree(project.root) == before
    assert replays == [ITEM_ID]


def test_an_update_with_a_stale_expected_revision_is_refused_and_replays_only_for_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replays: list[str]
) -> None:
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, ITEM_ID, "--expected-revision", _STALE_REVISION)

    assert code != 0, payload
    assert ROOT_REVISION_ID in str(payload), "the refusal must name the revision in place"
    assert proposals_tree(project.root) == before
    assert replays == [ITEM_ID]


def test_an_update_with_the_current_expected_revision_reaches_the_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replays: list[str]
) -> None:
    """The control for the two above: a spy that never fires would pass them for any reason."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")

    code, payload = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)

    assert code == 0, payload
    assert replays == [ITEM_ID]
