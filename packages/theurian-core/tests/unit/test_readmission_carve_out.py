"""``readmitted_items`` reports a retired item coming back, and only that."""

from __future__ import annotations

import pytest

from theurian.application import item_labels
from theurian.application.item_labels import ItemLabels, ReplayedItems, readmitted_items
from theurian.domain.enums import KnowledgeStatus, Sensitivity, TrustLevel, may_surface
from theurian.domain.identifiers import ItemId

pytestmark = pytest.mark.unit


def _held(status: KnowledgeStatus) -> ItemLabels:
    return ItemLabels(Sensitivity.INTERNAL, TrustLevel.REVIEWED, "backend", status)


def _replay(**statuses: KnowledgeStatus) -> ReplayedItems:
    return {
        ItemId(f"architecture.{name.replace('_', '-')}"): _held(status)
        for name, status in statuses.items()
    }


def test_a_rejected_item_is_retired_for_the_floor_while_draft_and_proposed_are_not() -> None:
    """A future flag surfacing ``rejected`` must not silently widen what a proposal may readmit."""
    before = _replay(
        was_rejected=KnowledgeStatus.REJECTED,
        was_draft=KnowledgeStatus.DRAFT,
        was_proposed=KnowledgeStatus.PROPOSED,
    )
    after = _replay(
        was_rejected=KnowledgeStatus.APPROVED,
        was_draft=KnowledgeStatus.APPROVED,
        was_proposed=KnowledgeStatus.APPROVED,
    )

    assert may_surface(KnowledgeStatus.REJECTED, include_unapproved=True) is False
    assert readmitted_items(before, after) == (
        (ItemId("architecture.was-rejected"), KnowledgeStatus.REJECTED, KnowledgeStatus.APPROVED),
    )


def test_readmitted_items_are_ordered_by_item_id_whatever_the_replays_dict_order() -> None:
    """The refusal text lists items in this order, so a dict-order dependence would make it vary."""
    later, earlier = ItemId("architecture.b-later"), ItemId("architecture.a-earlier")
    before: ReplayedItems = {
        later: _held(KnowledgeStatus.DEPRECATED),
        earlier: _held(KnowledgeStatus.DEPRECATED),
    }
    after: ReplayedItems = {
        later: _held(KnowledgeStatus.APPROVED),
        earlier: _held(KnowledgeStatus.APPROVED),
    }
    assert list(before) == [later, earlier], (
        "the fixture's dict order must be the reverse of sorted"
    )

    assert [item for item, _, _ in readmitted_items(before, after)] == [earlier, later]


def test_the_readmit_remedy_sends_the_reader_to_restore_item_and_to_validate_never_to_propose() -> (
    None
):
    """``propose`` and ``propose accept`` are the steps a retired item is refused at."""
    remedy = item_labels.READMIT_REMEDY

    assert "`restoreItem`" in remedy
    assert ".theurian/migrations/" in remedy
    assert "`theurian migrate validate`" in remedy
    assert "theurian propose" not in remedy
    assert "propose accept" not in remedy
