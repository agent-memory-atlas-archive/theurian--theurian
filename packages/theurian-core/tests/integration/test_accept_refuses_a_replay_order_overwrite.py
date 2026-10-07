"""``propose accept`` refuses a proposal that a landed migration replaying after it
would leave looser.

The guard refuses, with exit 1 and nothing moved, when the union replay ends a
gate field the proposal P wrote looser than P left it, whatever the id of the landed
migration that loosened it. The predicate is 0.5.1's floors': the class for
sensitivity, ``may_surface(..., include_unapproved=True)`` for status. The remedy is
``_redraft_remedy``'s and names ``dependsOn: [<the overwriting landed migration>]``.

P3: for the migration set ``accept`` replays, every floor-read label an accepted proposal
writes (sensitivity class; status surfaceability via ``may_surface(...,
include_unapproved=True)``) is, by those predicates, no looser after that replay than the
proposal left it. P3 is scoped to accepted proposals.

Not tested: a proposal that loosens, then a landed migration that tightens. A proposal
cannot loosen past 0.5.1's floors, so accept never reaches it. The tests drive the CLI
only: the guard's helper has no contract to pin yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from label_inheritance_support import (
    ROOT_MIGRATION_ID,
    SORTS_AFTER_A_DRAFT,
    cli,
    cli_ok,
    cli_propose,
    deprecation,
    item_row,
    labelled_project,
    land_reclassification,
    landing_zone,
    reclassification,
    staged_metadata,
)
from replay_order_support import (
    ABOVE_A_DRAFT,
    ACCEPTED,
    D1,
    D2,
    FACES,
    OTHER_ITEM,
    create_other,
    declare_dependency,
    depending_on,
    draft,
    get_is_withheld,
    history,
    restoration,
    write,
)

pytestmark = pytest.mark.integration

LARGER_IDS = (ABOVE_A_DRAFT, SORTS_AFTER_A_DRAFT)


@pytest.mark.parametrize("face", FACES)
def test_accept_refuses_and_names_the_overwriting_migration_and_the_route(
    face: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error is the end-state refusal's own, and the remedy never says "accept again"."""
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, face)
    drafted = draft(p, face)
    before = landing_zone(p.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert all(named in str(payload["error"]) for named in (D2, p.item_id, face)), payload
    assert "loosen what it sets" in str(payload["error"]), payload
    assert f"dependsOn: [{D2}]" in str(payload["remedy"]), payload
    assert not re.search(r"accept[^.]*again", str(payload["remedy"]), re.IGNORECASE), payload
    assert landing_zone(p.root) == before, "a refused accept moved files"


@pytest.mark.parametrize("face", FACES)
def test_a_fresh_draft_declaring_the_named_dependson_is_accepted_and_its_label_survives(
    face: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal comes first: the follow-through alone passes without the refusal."""
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, face)
    code, refused = cli("propose", "accept", draft(p, face)["proposalId"])
    assert code == 1, refused

    fresh = draft(p, face)
    declare_dependency(p, fresh, D2)
    cli_ok("propose", "accept", fresh["proposalId"])
    cli_ok("migrate", "apply")

    assert item_row(p.root, p.item_id)[face] == ACCEPTED[face]
    assert get_is_withheld(p)


@pytest.mark.parametrize(
    "ids", [(D1, D2), LARGER_IDS], ids=["smaller-id-restatement", "larger-id-restatement"]
)
@pytest.mark.parametrize("face", FACES)
def test_a_restatement_after_the_proposal_hides_no_later_loosening(
    face: str, ids: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first waits on the root, so it replays after the proposal and restates its label.

    Whatever its id, the second's loosening leaves the field looser than the proposal did.
    """
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, face, chained=True, ids=ids)
    drafted = draft(p, face)
    before = landing_zone(p.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert ids[1] in str(payload["error"]), payload
    assert "loosen what it sets" in str(payload["error"]), payload
    assert landing_zone(p.root) == before, "a refused accept moved files"


@pytest.mark.parametrize("ids", [(D1, D2), LARGER_IDS], ids=["smaller", "larger"])
@pytest.mark.parametrize("face", FACES)
def test_a_loosening_after_the_proposal_that_a_landed_migration_raises_back_is_accepted(
    face: str, ids: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first lowers the field below the proposal's, the second raises it back, with both
    ids smaller or both larger: the union ends at the proposal's own label.

    The status face pins that a status raise clears the loosening as a class raise does.
    """
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    drafted = draft(p, face)
    first_id, second_id = ids
    if face == "sensitivity":
        lower = reclassification(first_id, p.item_id, "internal")
        raise_back = reclassification(second_id, p.item_id, "confidential")
    else:
        lower, raise_back = restoration(first_id, p.item_id), deprecation(second_id, p.item_id)
    write(p.root, first_id, "lower", depending_on(lower, ROOT_MIGRATION_ID))
    write(p.root, second_id, "raise-back", depending_on(raise_back, first_id))
    cli_ok("migrate", "apply")

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 0, payload


def test_a_raise_of_another_item_does_not_forget_this_items_loosening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loosening is keyed by item as well as field: a later migration raising a
    different item's class must not read as this item being raised back."""
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    drafted = draft(p, "sensitivity")
    created = "01K1AAAAAB01234567890ABCDE"
    write(p.root, created, "other", create_other(created, OTHER_ITEM))
    lower = reclassification(D1, p.item_id, "internal")
    write(p.root, D1, "lower", depending_on(lower, ROOT_MIGRATION_ID))
    raise_other = reclassification(D2, OTHER_ITEM, "confidential")
    write(p.root, D2, "raise-other", depending_on(raise_other, D1))
    cli_ok("migrate", "apply")
    before = landing_zone(p.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert "loosen what it sets" in str(payload["error"]), payload
    assert landing_zone(p.root) == before, "a refused accept moved files"


def test_the_report_check_speaks_first_when_both_checks_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A withdrawal drafted before a landed update is undone by it: the report check's
    refusal, and its words, are the ones a caller sees; the end-state check would also refuse."""
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    drafted = draft(p, "status")
    code, update = cli_propose(p, p.item_id, "--expected-revision", p.revision_id)
    assert code == 0, update
    cli_ok("propose", "accept", update["proposalId"])
    cli_ok("migrate", "apply")
    before = landing_zone(p.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert "undoing what this proposal sets" in str(payload["error"]), payload
    assert "loosen what it sets" not in str(payload["error"]), payload
    assert landing_zone(p.root) == before, "a refused accept moved files"


def test_a_content_update_minted_before_a_landed_larger_id_declassification_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate, safe-side behaviour change: the update asserts the label it inherited.

    A drafted update writes its labels, so one minted before a landed
    declassification with a larger id restates the old class just ahead of it. The
    remedy's redraft inherits the new label and is accepted.
    """
    p = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    code, stale = cli_propose(p, p.item_id, "--expected-revision", p.revision_id)
    assert code == 0, stale
    land_reclassification(p.root, SORTS_AFTER_A_DRAFT, p.item_id, "internal")
    before = landing_zone(p.root)

    code, payload = cli("propose", "accept", stale["proposalId"])

    assert code == 1, payload
    assert "loosen what it sets" in str(payload["error"]), "no report row: the check's own"
    assert landing_zone(p.root) == before, "a refused accept moved files"
    assert f"dependsOn: [{SORTS_AFTER_A_DRAFT}]" in str(payload["remedy"]), payload
    code, fresh = cli_propose(p, p.item_id, "--expected-revision", p.revision_id)
    assert code == 0, fresh
    assert staged_metadata(p.root, fresh)["sensitivity"] == "internal"
    declare_dependency(p, fresh, SORTS_AFTER_A_DRAFT)
    cli_ok("propose", "accept", fresh["proposalId"])


@pytest.mark.parametrize(
    ("face", "relabel"),
    [("status", "internal"), ("sensitivity", "confidential")],
    ids=["a-different-field", "the-same-class"],
)
def test_a_landed_dependson_migration_that_leaves_the_proposals_predicate_alone_does_not_refuse(
    face: str, relabel: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The predicate is the class of the field the proposal wrote, not any later write."""
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, "sensitivity", relabel=relabel)

    code, payload = cli("propose", "accept", draft(p, face)["proposalId"])

    assert code == 0, payload
