"""A migration that declares ``dependsOn`` replays after every one that does not.

So it overwrites a later accepted proposal's label. A fix is correct either way:
``propose accept`` refuses and moves nothing, or the accepted label survives. The
faces fail without the accept-time refusal; each control drops only ``dependsOn``. P2:
an edge between the migrations of another item, none of which this item's migrations
depend on, changes nothing served for this one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from label_inheritance_support import (
    LabelledProject,
    cli,
    cli_ok,
    cli_propose,
    item_row,
    labelled_project,
    landing_zone,
    reclassification,
)
from replay_order_support import (
    ACCEPTED,
    FACES,
    OTHER_ITEM,
    create_other,
    depending_on,
    draft,
    get_is_withheld,
    history,
    write,
)

pytestmark = pytest.mark.integration


def _accept_and_apply(p: LabelledProject, drafted: dict[str, str]) -> tuple[int, bool]:
    before = landing_zone(p.root)
    code, _ = cli("propose", "accept", drafted["proposalId"])
    if code == 0:
        cli_ok("migrate", "apply")
    return code, landing_zone(p.root) != before


@pytest.mark.parametrize("face", FACES)
def test_an_accepted_label_is_not_overwritten_by_a_migration_that_declares_dependson(
    face: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accepted label is the one served, never the earlier-id ``dependsOn`` writer's.

    Either outcome is correct: ``propose accept`` refuses and moves nothing, or the
    accepted label survives the replay.
    """
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, face)

    code, moved = _accept_and_apply(p, draft(p, face))

    if code != 0:
        assert not moved, "a refused acceptance moved a migration into the set"
        return
    assert item_row(p.root, p.item_id)[face] == ACCEPTED[face]
    assert get_is_withheld(p), "knowledge.get served an item the accepted proposal withheld"


@pytest.mark.parametrize("face", FACES)
def test_without_dependson_the_accepted_label_is_the_one_served(
    face: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, face, dependent=False)

    code, moved = _accept_and_apply(p, draft(p, face))

    assert (code, moved) == (0, True), "the control proposal must be accepted"
    assert item_row(p.root, p.item_id)[face] == ACCEPTED[face]
    assert get_is_withheld(p)


def _two_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, edge: bool) -> tuple[str, str]:
    """OTHER's raise waits on its lowering when ``edge``; OTHER's label and the item's signature."""
    tmp_path.mkdir()
    p = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    history(p, "sensitivity", dependent=False)
    first, second, third = (f"01K1EEEEE{n}01234567890ABCDE" for n in "123")
    write(p.root, first, "other", create_other(first, OTHER_ITEM))
    text = reclassification(second, OTHER_ITEM, "confidential")
    write(p.root, second, "other-raise", depending_on(text, third) if edge else text)
    write(p.root, third, "other-lower", reclassification(third, OTHER_ITEM, "internal"))
    code, drafted = cli_propose(p, p.item_id, "--expected-revision", p.revision_id)
    assert code == 0, drafted
    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")
    labels = {k: v for k, v in item_row(p.root, p.item_id).items() if k != "current_revision_id"}
    return str(item_row(p.root, OTHER_ITEM)["sensitivity"]), f"{labels} {get_is_withheld(p)}"


def test_an_edge_between_migrations_of_an_unrelated_item_changes_nothing_served_for_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2. The positive control is ``OTHER_ITEM``: the edge reorders its migrations, so it
    ends with a different label; without it, equal outcomes could mean the edge did nothing.
    """
    plain_other, plain_signature = _two_items(tmp_path / "x", monkeypatch, edge=False)
    edged_other, edged_signature = _two_items(tmp_path / "y", monkeypatch, edge=True)

    assert plain_other != edged_other, "the edge reordered nothing: the property is vacuous"
    assert plain_signature == edged_signature
    assert plain_signature.endswith("False"), "the item must still be served"
