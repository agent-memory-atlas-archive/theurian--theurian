"""``propose accept`` refuses a proposal that readmits a retired item (GHSA-v2qg-23fc-7fqp).

``upsertRevision`` adopts ``metadata.status`` and the drafter always writes
``status: approved``, so a content update lands any deprecated, superseded or
rejected item back in the served set, with no reviewer asked to readmit it.

The floor compares effective post-state, never document keys: for an item the
landed set holds, the status after replaying the landed set alone must not be
non-surfaceable while the status after replaying landed plus proposal is
surfaceable. Readmission is a hand-authored ``restoreItem`` migration. Moving
between surfaceable statuses (``proposed`` or ``draft`` to ``approved``) stays
allowed: the merge of the proposal is the approval.

A retired item cannot be drafted for over the CLI once the draft-side refusal
exists, so the proposals here are built the way a pre-upgrade build left them: a
draft written while the item was still approved, or one copied from a donor
project holding the same item approved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import pytest
from label_inheritance_support import (
    CREATE_ONLY_ITEM_ID,
    EVIDENCE,
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
    land,
    land_deprecation,
    landing_zone,
    proposals_tree,
)

from theurian.cli.migration_pipeline import rehearse_migration_set
from theurian.daemon.runner import build_server
from theurian.domain.errors import MigrationError

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration

#: Where ``theurian propose accept`` looks its replay up at call time.
_PROPOSE_REHEARSE: Final = "theurian.cli.propose_commands.rehearse_migration_set"
_RETIRED: Final = ["deprecated", "superseded", "rejected"]
_STATUS_LINE: Final = re.compile(r"^(?P<head>    status: )\w+$", re.MULTILINE)


def _deprecate(project: LabelledProject, migration_id: str) -> None:
    land_deprecation(project.root, migration_id, project.item_id)


def _migration_file(project: LabelledProject, drafted: dict[str, Any]) -> Path:
    return Path(project.root / drafted["proposalDirectory"] / drafted["migrationFile"])


def _restate(project: LabelledProject, drafted: dict[str, Any], status: str) -> None:
    """Rewrite the staged ``upsertRevision``'s ``status``, as a hand edit or an older build would.

    Text, not a YAML round trip, and anchored to the four-space ``metadata``
    indentation, so no other value moves.
    """
    path = _migration_file(project, drafted)
    text = path.read_text(encoding="utf-8")
    assert len(_STATUS_LINE.findall(text)) == 1, "exactly one metadata status line to restate"
    path.write_text(_STATUS_LINE.sub(rf"\g<head>{status}", text), encoding="utf-8")


def _append_restore(project: LabelledProject, drafted: dict[str, Any], item_id: str) -> None:
    path = _migration_file(project, drafted)
    path.write_text(
        path.read_text(encoding="utf-8") + f"- op: restoreItem\n  itemId: {item_id}\n",
        encoding="utf-8",
    )


def _draft_update(project: LabelledProject) -> dict[str, Any]:
    code, drafted = cli_propose(project, project.item_id, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted
    assert drafted["expectedRevision"] == ROOT_REVISION_ID, "this must be an update"
    return drafted


def _donor_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """An update drafted in a separate project that holds the item ``approved``.

    The files are returned to be planted in a project where the item is retired:
    what an earlier build wrote while the item was still approved.
    """
    base = tmp_path / "donor"
    base.mkdir()
    donor = labelled_project(base, monkeypatch)
    drafted = _draft_update(donor)
    files = {
        path: data
        for path, data in proposals_tree(donor.root).items()
        if not path.endswith(".gitkeep")
    }
    assert files, "the donor staged nothing"
    return drafted, files


def _plant(project: LabelledProject, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        target = project.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _retired_project_with_a_donor_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> tuple[LabelledProject, dict[str, Any]]:
    drafted, files = _donor_proposal(tmp_path, monkeypatch)
    base = tmp_path / "target"
    base.mkdir()
    if status == "deprecated":
        project = labelled_project(base, monkeypatch)
        _deprecate(project, SORTS_BEFORE_A_DRAFT)
    else:
        project = labelled_project(base, monkeypatch, status=status)
    assert item_row(project.root, project.item_id)["status"] == status, "the item must be retired"
    _plant(project, files)
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the proposal must replay last"
    return project, drafted


def _refused_and_nothing_moved(project: LabelledProject, drafted: dict[str, Any]) -> dict[str, Any]:
    """Accept ``drafted``, expecting the status floor's refusal, and nothing moved.

    The remedy must carry ``restoreItem``: a refusal for any other reason would
    otherwise satisfy this over a proposal the floor never saw.
    """
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload  # the exit `propose accept --help` documents for a floor refusal
    assert "restoreItem" in str(payload.get("remedy", "")), payload
    assert landing_zone(project.root) == before, "a refused accept moved files"
    return payload


# -- A retired item is not readmitted by an update --------------------------------


@pytest.mark.parametrize("status", _RETIRED)
def test_accept_refuses_a_proposal_that_would_readmit_a_retired_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """A copied pre-upgrade proposal says ``status: approved``; the item must stay retired."""
    project, drafted = _retired_project_with_a_donor_proposal(tmp_path, monkeypatch, status)

    _refused_and_nothing_moved(project, drafted)
    cli_ok("migrate", "apply")

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == (status, ROOT_REVISION_ID)


@pytest.mark.parametrize("readmitted_as", ["draft", "proposed"])
def test_accept_refuses_a_readmission_to_a_surfaceable_status_that_is_not_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readmitted_as: str
) -> None:
    """``draft`` and ``proposed`` are served to a caller asking for unapproved items."""
    project, drafted = _retired_project_with_a_donor_proposal(tmp_path, monkeypatch, "deprecated")
    _restate(project, drafted, readmitted_as)

    _refused_and_nothing_moved(project, drafted)


def test_accept_refuses_the_mcp_draft_for_a_deprecated_create_only_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP lookup answers ``None`` for a retired item, so it drafts like an absent id.

    That answer is what keeps a withheld item indistinguishable from a missing
    one, so the draft must succeed and the refusal belongs to ``accept``.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    _deprecate(project, SORTS_BEFORE_A_DRAFT)
    assert item_row(project.root, project.item_id)["status"] == "deprecated"
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
    drafted = answer["result"]["structuredContent"]
    assert project.item_id == CREATE_ONLY_ITEM_ID

    _refused_and_nothing_moved(project, drafted)
    cli_ok("migrate", "apply")

    assert item_row(project.root, project.item_id)["status"] == "deprecated"


# -- The comparison is on effective post-state ------------------------------------


def _race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deprecation_id: str
) -> tuple[LabelledProject, dict[str, Any]]:
    """An update drafted while the item is approved, then a hand-authored deprecation lands."""
    project = labelled_project(tmp_path, monkeypatch)
    drafted = _draft_update(project)
    _deprecate(project, deprecation_id)
    assert item_row(project.root, project.item_id)["status"] == "deprecated"
    return project, drafted


def test_accept_refuses_a_draft_that_replays_after_a_deprecation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deprecation sorts first, so the draft's ``approved`` would replay over it."""
    project, drafted = _race(tmp_path, monkeypatch, SORTS_BEFORE_A_DRAFT)
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the deprecation must sort first"

    _refused_and_nothing_moved(project, drafted)


def test_accept_lands_a_draft_that_replays_before_a_deprecation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft sorts first and the deprecation replays last, so the item ends deprecated.

    Only a floor on the replayed status tells this from the test above: both
    proposals carry the same ``status: approved``, so a floor reading document
    keys refuses this one too.
    """
    project, drafted = _race(tmp_path, monkeypatch, SORTS_AFTER_A_DRAFT)
    assert drafted["migrationId"] < SORTS_AFTER_A_DRAFT, "the draft must sort first"

    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    row = item_row(project.root, project.item_id)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["status"] == "deprecated"


def test_accept_refuses_a_proposal_whose_own_restore_item_readmits_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upsert keeps ``status: deprecated``; the appended ``restoreItem`` is what readmits.

    A floor reading the upsert's ``status`` sees no readmission here.
    """
    project, drafted = _race(tmp_path, monkeypatch, SORTS_BEFORE_A_DRAFT)
    _restate(project, drafted, "deprecated")
    _append_restore(project, drafted, project.item_id)

    _refused_and_nothing_moved(project, drafted)


# -- What the refusal says --------------------------------------------------------


def test_the_refusal_names_the_item_and_both_statuses_and_the_way_to_readmit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's view is unscoped, so the message may name what the replay compared.

    Without the item and both statuses an author cannot tell which of several
    items in the set is the one being readmitted.
    """
    project, drafted = _race(tmp_path, monkeypatch, SORTS_BEFORE_A_DRAFT)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    error = str(payload.get("error", ""))
    assert project.item_id in error, payload
    assert "deprecated" in error, payload
    assert "approved" in error, payload
    remedy = str(payload.get("remedy", ""))
    assert "restoreItem" in remedy, payload
    assert ".theurian/migrations/" in remedy, payload
    assert "Nothing has moved" in remedy, payload


def _lower_the_staged_sensitivity(project: LabelledProject, drafted: dict[str, Any]) -> None:
    path = _migration_file(project, drafted)
    path.write_text(
        re.sub(r"^(    sensitivity: )\w+$", r"\1internal", path.read_text(), flags=re.MULTILINE),
        encoding="utf-8",
    )


def test_a_proposal_that_lowers_a_sensitivity_and_readmits_an_item_is_refused_for_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both defects at once are named, each with its own cure.

    The sensitivity floor alone refuses this proposal, so a refusal by itself
    proves nothing about the status floor: a floor that dropped either cause, or
    either remedy, would still exit non-zero and send the author to fix half.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    drafted = _draft_update(project)
    _lower_the_staged_sensitivity(project, drafted)
    _deprecate(project, SORTS_BEFORE_A_DRAFT)
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert "lower the sensitivity" in error and "readmit" in error, payload
    assert error.count(project.item_id) == 2, payload
    remedy = str(payload.get("remedy", ""))
    assert "changeSensitivity" in remedy and "restoreItem" in remedy, payload
    assert remedy.index("changeSensitivity") < remedy.index("restoreItem"), payload
    assert landing_zone(project.root) == before, "a refused accept moved files"


_OTHER_ITEM_ID: Final = "architecture.other-policy"
_OTHER_ITEM_MIGRATION_ID: Final = "01K1AAAAAB01234567890ABCDE"


def _land_a_second_item_and_deprecate_it(project: LabelledProject) -> None:
    (project.root / f".theurian/migrations/{_OTHER_ITEM_MIGRATION_ID}-other.yaml").write_text(
        f"""apiVersion: theurian.dev/v1
id: {_OTHER_ITEM_MIGRATION_ID}
createdAt: 2026-08-02T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {_OTHER_ITEM_ID}
    kind: architecture
    namespace: backend
    owner: platform-team
    sensitivity: internal
"""
    )
    cli_ok("migrate", "apply")
    land_deprecation(project.root, SORTS_BEFORE_A_DRAFT, _OTHER_ITEM_ID)
    assert item_row(project.root, _OTHER_ITEM_ID)["status"] == "deprecated"


def test_a_proposal_that_lowers_one_item_and_readmits_another_names_both_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floors report per item, so a set holding two different faults names each item once.

    The proposal's own migration lowers the confidential item and carries a
    ``restoreItem`` for a different, deprecated one.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _land_a_second_item_and_deprecate_it(project)
    drafted = _draft_update(project)
    _lower_the_staged_sensitivity(project, drafted)
    _append_restore(project, drafted, _OTHER_ITEM_ID)
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert re.search(rf"lower the sensitivity of '{project.item_id} from confidential", error), (
        payload
    )
    assert re.search(rf"readmit '{_OTHER_ITEM_ID} from deprecated", error), payload
    assert project.item_id not in error.split("readmit")[1], payload
    assert _OTHER_ITEM_ID not in error.split("readmit", maxsplit=1)[0], payload
    remedy = str(payload.get("remedy", ""))
    assert "changeSensitivity" in remedy and "restoreItem" in remedy, payload
    assert landing_zone(project.root) == before, "a refused accept moved files"


# -- Allowed: moving between surfaceable statuses, or staying retired -------------


def test_accept_lands_the_first_revision_of_a_create_only_draft_item_as_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``createItem`` leaves status ``draft``; its first revision is the approval path."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    assert item_row(project.root, project.item_id)["status"] == "draft"
    code, drafted = cli_propose(project, project.item_id)
    assert code == 0, drafted

    land(drafted["proposalId"])

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == ("approved", drafted["revisionId"])


@pytest.mark.parametrize("status", ["draft", "proposed"])
def test_accept_lands_an_update_of_a_surfaceable_unapproved_item_as_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The recorded decision: the merge is the approval, so ``draft`` and ``proposed`` move up."""
    project = labelled_project(tmp_path, monkeypatch, status=status)
    assert item_row(project.root, project.item_id)["status"] == status
    drafted = _draft_update(project)

    land(drafted["proposalId"])

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == ("approved", drafted["revisionId"])


def test_accept_lands_a_hand_written_proposal_that_deprecates_an_approved_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor is on readmission only: retiring an item through a proposal is not refused."""
    project = labelled_project(tmp_path, monkeypatch)
    drafted = _draft_update(project)
    _restate(project, drafted, "deprecated")

    land(drafted["proposalId"])

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == ("deprecated", drafted["revisionId"])


@pytest.mark.parametrize("status", _RETIRED)
def test_accept_lands_a_proposal_that_keeps_a_retired_item_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Retired to retired is not readmission, so an edit of a retired item's body still lands."""
    project, drafted = _retired_project_with_a_donor_proposal(tmp_path, monkeypatch, status)
    _restate(project, drafted, status)

    land(drafted["proposalId"])

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == (status, drafted["revisionId"])


def test_the_donor_proposal_is_acceptable_where_the_item_is_not_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the refusals above: the planted proposal is sound, so only status refuses it."""
    drafted, files = _donor_proposal(tmp_path, monkeypatch)
    base = tmp_path / "target"
    base.mkdir()
    project = labelled_project(base, monkeypatch)
    _plant(project, files)

    land(drafted["proposalId"])

    row = item_row(project.root, project.item_id)
    assert (row["status"], row["current_revision_id"]) == ("approved", drafted["revisionId"])


# -- The premise: a document cannot omit status to dodge the floor -----------------


def test_a_proposal_omitting_status_is_refused_by_the_schema_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``status`` is required by ``revisionMetadata``, so the floor never meets a missing one.

    A document without it would load with some default, and a floor reading
    document keys would not see a readmission.
    """
    project = labelled_project(tmp_path, monkeypatch)
    drafted = _draft_update(project)
    path = _migration_file(project, drafted)
    path.write_text(_STATUS_LINE.sub("", path.read_text(encoding="utf-8")), encoding="utf-8")
    assert "status:" not in path.read_text(encoding="utf-8")
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert path.name in str(payload.get("error", "")), payload
    assert "is not a valid migration" in str(payload.get("error", "")), payload
    assert landing_zone(project.root) == before, "a refused accept moved files"


# -- No third replay ----------------------------------------------------------------


def test_accept_rehearses_exactly_twice_the_union_then_the_landed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both floors read the one held replay; a third full replay is cost the design forbids.

    Measured on the review of GHSA-v2qg-23fc-7fqp, the second replay took ``accept``
    at 1,000 items from 4.2 s to 6.6 s. No other test counts the replays.
    """
    project = labelled_project(tmp_path, monkeypatch)
    drafted = _draft_update(project)
    incoming_seen: list[bool] = []
    real = rehearse_migration_set

    def spy(candidate: Any, *, clock: Any) -> Any:
        incoming_seen.append(bool(candidate.incoming))
        return real(candidate, clock=clock)

    monkeypatch.setattr(_PROPOSE_REHEARSE, spy)

    cli_ok("propose", "accept", drafted["proposalId"])

    assert incoming_seen == [True, False]


def test_the_spy_sees_a_failing_landed_replay_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the spy is wired to the replay accept uses, since a raise there is a refusal."""
    project = labelled_project(tmp_path, monkeypatch)
    drafted = _draft_update(project)
    real = rehearse_migration_set

    def failing_when_alone(candidate: Any, *, clock: Any) -> Any:
        if not candidate.incoming:
            raise MigrationError("landed set cannot replay alone")
        return real(candidate, clock=clock)

    monkeypatch.setattr(_PROPOSE_REHEARSE, failing_when_alone)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert "landed set cannot replay alone" in str(payload.get("error", "")), payload
