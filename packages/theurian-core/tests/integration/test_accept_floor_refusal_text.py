"""What ``propose accept`` says when the sensitivity floor refuses, and why.

The refusal text is read by an author who has to act on it, so a sentence that
blames the wrong party sends them to fix something that is correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest
from git_harness import commit_migrations
from label_inheritance_support import (
    EVIDENCE,
    ITEM_ID,
    SORTS_AFTER_A_DRAFT,
    SORTS_BEFORE_A_DRAFT,
    UPDATED_BODY,
    cli,
    cli_propose,
    labelled_project,
    landing_zone,
    reclassification,
    staged_metadata,
)

from theurian.cli.migration_pipeline import rehearse_migration_set
from theurian.daemon.runner import build_server
from theurian.domain.errors import MigrationError

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration


_NEW_ITEM = "architecture.brand-new"

#: Where ``theurian propose accept`` looks its replay up at call time.
_PROPOSE_REHEARSE: Final = "theurian.cli.propose_commands.rehearse_migration_set"


def test_accept_does_not_blame_the_landed_set_for_a_proposal_that_makes_it_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The landed set fails alone and passes with the proposal, so "with or without" is false.

    The proposal creates the item that a landed ``changeSensitivity`` names and
    sorts before it. Without the proposal that migration has no item to
    reclassify; with it the union replays. The floor cannot establish the item's
    current labels from the landed set, so it must refuse (fail closed), and the
    reason it gives is that one, not that the set is broken regardless.
    """
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    code, drafted = cli_propose(project, _NEW_ITEM, "--sensitivity", "internal")
    assert code == 0, drafted
    assert drafted["expectedRevision"] is None, "this must be a first revision"
    assert drafted["migrationId"] < SORTS_AFTER_A_DRAFT, "the proposal must sort first"
    (project.root / f".theurian/migrations/{SORTS_AFTER_A_DRAFT}-reclassify.yaml").write_text(
        reclassification(SORTS_AFTER_A_DRAFT, _NEW_ITEM, "confidential")
    )
    commit_migrations(project.root)
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    text = json.dumps(payload)
    assert "with or without this proposal" not in text, text
    assert "does not replay on its own" in text, text
    assert landing_zone(project.root) == before, "a refused accept moved files"


def test_accept_of_a_draft_made_against_a_lagging_served_state_points_at_migrate_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP draft inherits the label the served state holds, which can be stale.

    A reclassification to ``confidential`` is committed under ``migrations/`` and
    ``migrate apply`` has not run, so the tool still serves ``internal`` and drafts
    it. ``accept`` replays the files and refuses that draft, and re-drafting over
    MCP would inherit the same stale label, so the remedy must name applying the
    migration. It is not the only cure: ``theurian propose --expected-revision``
    reads the landed files and stages the right label at once, which
    ``test_the_lowering_remedy_names_theurian_propose_as_the_route_to_draft_again``
    pins.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    (project.root / f".theurian/migrations/{SORTS_BEFORE_A_DRAFT}-reclassify.yaml").write_text(
        reclassification(SORTS_BEFORE_A_DRAFT, ITEM_ID, "confidential")
    )
    commit_migrations(project.root)
    arguments: dict[str, Any] = {
        "projectId": "demo",
        "itemId": ITEM_ID,
        "title": "Authentication policy",
        "kind": "architecture",
        "owner": "platform-team",
        "author": "platform-team@example.com",
        "description": "Tighten the token lifetime.",
        "body": UPDATED_BODY,
        "contentType": "text/markdown",
        "evidence": EVIDENCE,
        "sourceAnchors": [{"provider": "git", "sourceUri": "git://demo/auth-policy.md"}],
        "expectedRevision": project.revision_id,
    }
    with mcp_session(build_server(project.registry), tmp_path / "wire") as call:
        answer = call("knowledge.proposeChange", arguments)
    result = answer["result"]
    assert result["isError"] is False, result
    drafted = result["structuredContent"]
    assert staged_metadata(project.root, drafted)["sensitivity"] == "internal", "not stale"
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the draft must replay last"
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert "theurian migrate apply" in str(payload.get("remedy", "")), payload
    assert landing_zone(project.root) == before, "a refused accept moved files"


def test_the_lowering_remedy_names_theurian_propose_as_the_route_to_draft_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Draft it again" gives a CLI reader no command: ``theurian propose`` is the route.

    The draft reads the landed files, so it stages the item's current label even
    while the served state lags.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(
        project, ITEM_ID, "--expected-revision", project.revision_id, "--sensitivity", "internal"
    )
    assert code == 0, drafted
    (project.root / f".theurian/migrations/{SORTS_BEFORE_A_DRAFT}-reclassify.yaml").write_text(
        reclassification(SORTS_BEFORE_A_DRAFT, ITEM_ID, "confidential")
    )
    commit_migrations(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert "lower the sensitivity" in str(payload.get("error", "")), payload
    assert "theurian propose" in str(payload.get("remedy", "")), payload


def test_a_landed_set_that_fails_alone_is_not_sent_to_migrate_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``migrate validate``'s verdict does not rest on a replay: it reports valid on this set.

    The union replays and the landed set alone does not, which only the held
    replay can say. The refusal carries the engine's own words and sends the
    reader to read what they name in ``.theurian/migrations/``.
    """
    project = labelled_project(tmp_path, monkeypatch)
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", project.revision_id)
    assert code == 0, drafted
    real = rehearse_migration_set

    def failing_when_alone(candidate: Any, *, clock: Any) -> Any:
        if not candidate.incoming:
            raise MigrationError("Revision conflict on the landed set alone")
        return real(candidate, clock=clock)

    monkeypatch.setattr(_PROPOSE_REHEARSE, failing_when_alone)
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert "Revision conflict on the landed set alone" in str(payload.get("error", "")), payload
    remedy = str(payload.get("remedy", ""))
    assert "migrate validate" not in remedy, payload
    assert ".theurian/migrations/" in remedy, payload
    assert "`theurian migrate apply` runs the same replay" in remedy, payload
    assert landing_zone(project.root) == before, "a refused accept moved files"
