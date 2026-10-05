"""``propose accept`` refuses a proposal whose own upsert the union replay would report.

GHSA-v2qg-23fc-7fqp: a proposal's migration id is minted at draft, so a
``restoreItem`` hand-authored after the draft, for an item the proposal was
drafted against while it was retired, sorts after the proposal's migration.
Accepted then, the upsert's ``approved`` replays over ``deprecated`` before the
restore does, and ``migrate validate`` reports ``deprecated -> approved`` for the
proposal's migration for as long as it exists. This file holds the incoming
migration's own row; ``test_accept_never_introduces_a_report_row.py`` holds the
wider invariant, that an accept introduces no row at all, a landed migration's
included.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from label_inheritance_support import (
    EVIDENCE,
    ITEM_ID,
    ROOT_REVISION_ID,
    SORTS_AFTER_A_DRAFT,
    SORTS_BEFORE_A_DRAFT,
    UPDATED_BODY,
    LabelledProject,
    cli,
    cli_ok,
    cli_propose,
    create_only_project,
    item_row,
    labelled_project,
    land_deprecation,
    landing_zone,
)

from theurian.application import item_labels
from theurian.daemon.runner import build_server
from theurian.infrastructure.determinism import UlidGenerator

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration


def _no_row_for(migration_id: str) -> None:
    """``migrate validate`` carries no permissive-move row for ``migration_id``."""
    validated = cli_ok("migrate", "validate")
    assert isinstance(validated.get("permissiveMoves"), list), validated
    rows = [row for row in validated["permissiveMoves"] if row["migrationId"] == migration_id]
    assert rows == [], rows


def _mcp_draft(project: LabelledProject, tmp_path: Path) -> dict[str, Any]:
    """``knowledge.proposeChange`` with no ``expectedRevision``; a retired item looks absent."""
    arguments: dict[str, Any] = {
        "projectId": "demo",
        "itemId": project.item_id,
        "title": "Authentication policy",
        "kind": "architecture",
        "owner": "platform-team",
        "author": "platform-team@example.com",
        "description": "Tighten the token lifetime.",
        "body": UPDATED_BODY,
        "contentType": "text/markdown",
        "evidence": EVIDENCE,
        "sourceAnchors": [{"provider": "git", "sourceUri": "git://demo/auth-policy.md"}],
    }
    with mcp_session(build_server(project.registry), tmp_path / "wire") as call:
        answer = call("knowledge.proposeChange", arguments)
    assert answer["result"]["isError"] is False, answer
    drafted: dict[str, Any] = answer["result"]["structuredContent"]
    return drafted


def _restore(
    project: LabelledProject,
    *,
    depends_on: list[str] | None = None,
    migration_id: str | None = None,
) -> str:
    """A hand-authored ``restoreItem``, minted now, so it sorts after any draft made before it.

    ``depends_on`` declares ``dependsOn``, which the layered replay sort puts after
    every migration that declares none, whatever the ids. ``migration_id`` hand-picks
    the id, as one that sorts after every draft to come.
    """
    time.sleep(0.05)
    migration_id = migration_id or UlidGenerator().new_ulid().value
    document: dict[str, Any] = {
        "apiVersion": "theurian.dev/v1",
        "id": migration_id,
        "createdAt": "2026-10-02T00:00:00+09:00",
        "author": "engineer@example.com",
        "operations": [{"op": "restoreItem", "itemId": project.item_id}],
    }
    if depends_on is not None:
        document["dependsOn"] = depends_on
    (project.root / f".theurian/migrations/{migration_id}-restore.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False)
    )
    cli_ok("migrate", "apply")
    time.sleep(0.05)
    return migration_id


def _drafted_then_refused_then_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    restore_depends_on: list[str] | None = None,
    restore_id: str | None = None,
) -> tuple[LabelledProject, dict[str, Any], dict[str, Any], str]:
    """A draft for a retired create-only item, the first accept's refusal, and the restore.

    The restore's id sorts after the draft's: the order the first refusal's remedy
    sends a reader into.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    land_deprecation(project.root, SORTS_BEFORE_A_DRAFT, project.item_id)
    drafted = _mcp_draft(project, tmp_path)

    code, refused = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, refused
    restore_id = _restore(project, depends_on=restore_depends_on, migration_id=restore_id)
    assert restore_id > drafted["migrationId"], "the restore must sort after the draft"
    return project, drafted, refused, restore_id


# -- the first refusal's remedy -----------------------------------------------------


def test_the_readmission_remedy_says_to_draft_again_after_the_restore_lands() -> None:
    """Re-accepting the refused proposal is a dead end: it replays before the restore."""
    remedy = item_labels.ACCEPT_READMISSION_REMEDY

    assert "restoreItem" in remedy
    assert "theurian propose" in remedy
    assert "again" in remedy


def test_the_first_accept_of_a_proposal_for_a_retired_item_prints_the_draft_again_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The printed refusal, not just the constant, carries the route that works."""
    _, _, refused, _ = _drafted_then_refused_then_restored(tmp_path, monkeypatch)

    remedy = str(refused.get("remedy", ""))

    assert "restoreItem" in remedy, refused
    assert "theurian propose" in remedy and "again" in remedy, refused


# -- the second accept ---------------------------------------------------------------


def test_accepting_a_proposal_that_replays_before_a_landed_restore_is_refused_and_moves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accepted upsert would read ``deprecated -> approved`` in every later report."""
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(tmp_path, monkeypatch)
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    for named in (
        drafted["migrationId"],
        project.item_id,
        "status",
        "deprecated",
        "approved",
        restore_id,
    ):
        assert named in error, (named, payload)
    assert "replays before" in error, payload
    remedy = str(payload.get("remedy", ""))
    assert "theurian propose" in remedy, payload
    assert f"replays after {restore_id}" in remedy, payload
    assert landing_zone(project.root) == before, "a refused accept moved files"


def test_the_own_row_remedy_keeps_the_content_path_for_an_upsert_and_never_says_accept_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refused proposal is an ``upsertRevision``, which only ``theurian propose`` drafts.

    The landed restore here is a root migration, and the remedy still names
    ``dependsOn`` for it: a later id is not what places a draft after it.
    """
    _, drafted, _, restore_id = _drafted_then_refused_then_restored(tmp_path, monkeypatch)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    remedy = str(payload.get("remedy", ""))
    assert "`theurian propose`" in remedy, payload
    assert f"dependsOn: [{restore_id}]" in remedy, payload
    assert "later migration id" not in remedy, payload
    assert not re.search(r"accept[^.]*again", remedy, re.IGNORECASE), payload


def test_the_own_row_remedy_names_dependson_when_the_landed_restore_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh draft carries no ``dependsOn``, so it still replays before such a restore.

    The layered sort replays every migration without ``dependsOn`` before every
    migration with one, whatever the ids, so "a later id replays after it" is false
    here and a remedy saying only that loops the reader back into the refusal.
    """
    _, drafted, _, restore_id = _drafted_then_refused_then_restored(
        tmp_path, monkeypatch, restore_depends_on=[SORTS_BEFORE_A_DRAFT]
    )

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert restore_id in str(payload.get("error", "")), payload
    remedy = str(payload.get("remedy", ""))
    assert re.search(rf"dependsOn\W+{restore_id}", remedy), payload
    assert not re.search(r"accept[^.]*again", remedy, re.IGNORECASE), payload


def test_the_own_row_remedy_authors_every_kind_of_a_proposal_carrying_a_non_content_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming ``theurian propose`` for a proposal that also changes an owner would send
    the reader to drop the owner change: the remedy reads the refused document's kinds.
    """
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(tmp_path, monkeypatch)
    staged = project.root / drafted["proposalDirectory"] / drafted["migrationFile"]
    document = yaml.safe_load(staged.read_text(encoding="utf-8"))
    document["operations"].append(
        {"op": "changeOwner", "itemId": project.item_id, "owner": "security-team"}
    )
    staged.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert drafted["migrationId"] in str(payload.get("error", "")), payload
    assert restore_id in str(payload.get("error", "")), payload
    remedy = str(payload.get("remedy", ""))
    assert (
        "Author the changeOwner and createItem and upsertRevision operations as a migration"
        in remedy
    ), payload
    assert "theurian propose" not in remedy, payload


def _land_unrelated(project: LabelledProject, operation: dict[str, Any]) -> str:
    """Land a migration minted now, after the restore, that is not about the row."""
    time.sleep(0.05)
    migration_id = UlidGenerator().new_ulid().value
    (project.root / f".theurian/migrations/{migration_id}-unrelated.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "theurian.dev/v1",
                "id": migration_id,
                "createdAt": "2026-10-02T00:00:00+09:00",
                "author": "engineer@example.com",
                "operations": [operation],
            },
            sort_keys=False,
        )
    )
    cli_ok("migrate", "apply")
    return migration_id


_ULID = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")


@pytest.mark.parametrize(
    "unrelated",
    [
        {
            "op": "createItem",
            "itemId": "architecture.unrelated",
            "kind": "architecture",
            "namespace": "security",
            "owner": "platform-team",
        },
        {
            "op": "changeSensitivity",
            "itemId": "<same item>",
            "sensitivity": "confidential",
            "reason": "reclassified after review",
        },
    ],
    ids=["another-item", "another-field-of-the-same-item"],
)
def test_the_refusal_names_only_the_landed_migrations_that_wrote_the_reported_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unrelated: dict[str, Any]
) -> None:
    """The reader is told which landed migration to wait for; an unrelated one misdirects them.

    The ids in the error are exactly the proposal's own and the restore's. A later
    migration that wrote another item, or another field of this one, is not what the
    proposal replays before in the sense that matters, and listing it sends the
    reader to review a migration with nothing to do with the refusal.
    """
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(tmp_path, monkeypatch)
    operation = {
        key: project.item_id if value == "<same item>" else value
        for key, value in unrelated.items()
    }
    unrelated_id = _land_unrelated(project, operation)
    assert unrelated_id > restore_id > drafted["migrationId"]

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert set(_ULID.findall(error)) == {drafted["migrationId"], restore_id}, payload


def test_accepting_a_proposal_while_another_migrations_row_exists_leaves_only_that_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row on another item is not this proposal's, so it neither refuses nor joins the accept.

    A refusal matching any row of the report, rather than the one for the proposal's
    own migration, would stop every accept once a race row exists anywhere.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, raced = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, raced
    cli_ok("propose", "accept", raced["proposalId"])
    cli_ok("migrate", "apply")
    land_deprecation(project.root, SORTS_BEFORE_A_DRAFT, ITEM_ID)
    other = "architecture.other-policy"
    _land_unrelated(
        project,
        {
            "op": "createItem",
            "itemId": other,
            "kind": "architecture",
            "namespace": "backend",
            "owner": "platform-team",
        },
    )
    rows_before = cli_ok("migrate", "validate")["permissiveMoves"]
    assert [(row["itemId"], row["migrationId"]) for row in rows_before] == [
        (ITEM_ID, raced["migrationId"])
    ], rows_before
    code, drafted = cli_propose(project, other)
    assert code == 0, drafted

    accepted_code, accepted = cli("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    assert accepted_code == 0, accepted
    rows = cli_ok("migrate", "validate")["permissiveMoves"]
    assert rows == rows_before, rows
    _no_row_for(drafted["migrationId"])


def test_no_row_names_the_migration_of_a_proposal_accepted_across_a_landed_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the accept did, the report holds no row for the proposal's migration."""
    _, drafted, _, _ = _drafted_then_refused_then_restored(tmp_path, monkeypatch)

    cli("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    _no_row_for(drafted["migrationId"])


def test_a_fresh_draft_after_the_restore_is_accepted_and_reports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft with no ``dependsOn`` whose id sorts after the restore is accepted.

    No row is reported. The remedy always names ``dependsOn``; this holds only the case where the id
    order alone already places the draft after the restore. The ``dependsOn``
    case is the next test.
    """
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(tmp_path, monkeypatch)
    code, refused = cli("propose", "accept", drafted["proposalId"])
    assert code == 1, refused

    code, fresh = cli_propose(project, project.item_id)
    assert code == 0, fresh
    assert fresh["migrationId"] > restore_id, "the new draft must sort after the restore"
    cli_ok("propose", "accept", fresh["proposalId"])
    cli_ok("migrate", "apply")

    assert cli_ok("migrate", "validate")["permissiveMoves"] == []
    _no_row_for(fresh["migrationId"])


def test_a_redraft_after_a_dependent_restore_is_accepted_only_with_dependson_edited_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``dependsOn`` clause of the remedy is the step that makes the redraft land.

    The layered sort replays a migration without ``dependsOn`` before every one
    with it, so the fresh draft is refused until ``dependsOn: [<restore>]`` is in
    its staged file, as the remedy says and as ``theurian propose`` has no option for.
    """
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(
        tmp_path, monkeypatch, restore_depends_on=[SORTS_BEFORE_A_DRAFT]
    )
    code, refused = cli("propose", "accept", drafted["proposalId"])
    assert code == 1, refused
    code, fresh = cli_propose(project, project.item_id)
    assert code == 0, fresh
    assert fresh["migrationId"] > restore_id, "the new draft must sort after the restore"

    code, bare = cli("propose", "accept", fresh["proposalId"])

    assert code == 1, bare
    assert restore_id in str(bare.get("error", "")), bare

    staged = Path(project.root / fresh["proposalDirectory"] / fresh["migrationFile"])
    document = yaml.safe_load(staged.read_text(encoding="utf-8"))
    document["dependsOn"] = [restore_id]
    staged.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    cli_ok("propose", "accept", fresh["proposalId"])
    cli_ok("migrate", "apply")

    assert cli_ok("migrate", "validate")["permissiveMoves"] == []


def _edit_dependson_into(project: LabelledProject, fresh: dict[str, Any], ids: list[str]) -> None:
    staged = Path(project.root / fresh["proposalDirectory"] / fresh["migrationFile"])
    document = yaml.safe_load(staged.read_text(encoding="utf-8"))
    document["dependsOn"] = ids
    staged.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_a_redraft_after_a_restore_whose_id_sorts_after_every_draft_lands_with_dependson(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy cannot rest on id order: a landed id past the drafting clock outlasts redrafts.

    A hand-chosen id, an id edited before accept and a collaborator's fast clock all
    produce it. Followed literally the remedy looped, exit 1 every time; ``dependsOn``
    is what places the fresh draft after the restore.
    """
    project, drafted, _, restore_id = _drafted_then_refused_then_restored(
        tmp_path, monkeypatch, restore_id=SORTS_AFTER_A_DRAFT
    )
    code, payload = cli("propose", "accept", drafted["proposalId"])
    assert code == 1, payload
    remedy = str(payload.get("remedy", ""))
    assert f"dependsOn: [{restore_id}]" in remedy, payload
    code, fresh = cli_propose(project, project.item_id)
    assert code == 0, fresh
    assert fresh["migrationId"] < restore_id, "the restore must sort after the new draft"

    _edit_dependson_into(project, fresh, [restore_id])
    accepted_code, accepted = cli("propose", "accept", fresh["proposalId"])
    cli_ok("migrate", "apply")

    assert accepted_code == 0, accepted
    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == ("approved", fresh["revisionId"])
    assert cli_ok("migrate", "validate")["permissiveMoves"] == []


def test_a_redraft_after_two_dependent_restores_names_both_and_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error and the remedy name the same landed migrations: every one the redraft must follow.

    ``replays after`` is the list ``dependsOn`` repeats; naming only the first, the
    last or the refused proposal's own id would each leave a writer for the redraft
    to replay before.
    """
    project, drafted, _, first = _drafted_then_refused_then_restored(tmp_path, monkeypatch)
    second = _restore(project, depends_on=[first])

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert first in error and second in error, payload
    remedy = str(payload.get("remedy", ""))
    assert f"replays after {first}, {second}" in remedy, payload
    assert f"dependsOn: [{first}, {second}]" in remedy, payload
    code, fresh = cli_propose(project, project.item_id)
    assert code == 0, fresh

    _edit_dependson_into(project, fresh, [first, second])
    accepted_code, accepted = cli("propose", "accept", fresh["proposalId"])
    cli_ok("migrate", "apply")

    assert accepted_code == 0, accepted
    assert cli_ok("migrate", "validate")["permissiveMoves"] == []


# -- the invariant on the plain race fixture ---------------------------------------


def test_an_accepted_update_leaves_no_report_row_before_any_withdrawal_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the invariant: a plain accepted update is not itself reported."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted

    cli_ok("propose", "accept", drafted["proposalId"])

    _no_row_for(drafted["migrationId"])
