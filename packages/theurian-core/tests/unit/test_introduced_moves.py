"""``introduced_moves`` keeps only the report rows an accept would add (GHSA-v2qg-23fc-7fqp).

The accept-side invariant is that an accept never introduces a report row. The
baseline is the landed-alone replay's report, not an empty one: a history that
already holds a row must still accept anything that leaves that row as it is. A row
is the same row when its key ``(migration_id, item_id, field, undoes)`` is held,
whatever ``before``, ``after`` or ``kind`` the union reads for it. An accept never
introduces or re-attributes a row: the same row naming another migration it undoes
is new.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import replace

import pytest

from theurian.application.permissive_moves import PermissiveMove
from theurian.domain.enums import KnowledgeStatus, Sensitivity
from theurian.domain.identifiers import ItemId, MigrationId

pytestmark = pytest.mark.unit

_LANDED = MigrationId("01K1CCCCCC01234567890ABCDE")
_INCOMING = MigrationId("01K1DDDDDD01234567890ABCDE")
_ROOT = MigrationId("01K1BBBBBB01234567890ABCDE")
_X = ItemId("architecture.x")
_Y = ItemId("architecture.y")


def _introduced(
    held: Sequence[PermissiveMove], union: Sequence[PermissiveMove]
) -> tuple[PermissiveMove, ...]:
    module = importlib.import_module("theurian.application.permissive_moves")
    result: tuple[PermissiveMove, ...] = module.introduced_moves(held, union)
    return result


def _status_move(
    migration_id: MigrationId,
    item_id: ItemId = _X,
    *,
    before: KnowledgeStatus = KnowledgeStatus.DEPRECATED,
    after: KnowledgeStatus = KnowledgeStatus.APPROVED,
    kind: str | None = "undoes",
) -> PermissiveMove:
    return PermissiveMove(
        migration_id=migration_id,
        item_id=item_id,
        field="status",
        before=before,
        after=after,
        undoes=_ROOT if kind is not None else None,
        kind=kind,  # type: ignore[arg-type]
    )


def _sensitivity_move(migration_id: MigrationId, item_id: ItemId = _X) -> PermissiveMove:
    return PermissiveMove(
        migration_id=migration_id,
        item_id=item_id,
        field="sensitivity",
        before=Sensitivity.CONFIDENTIAL,
        after=Sensitivity.INTERNAL,
        undoes=_ROOT,
        kind="undoes",
    )


def test_the_incoming_migrations_own_row_is_introduced() -> None:
    own = _status_move(_INCOMING)

    assert _introduced([], [own]) == (own,)


def test_a_landed_migrations_row_that_only_the_union_holds_is_introduced() -> None:
    """The withdrawal-through-accept face: accepting moves a landed row into the report."""
    landed_row = _status_move(_LANDED)

    assert _introduced([], [landed_row]) == (landed_row,)


def test_a_row_present_in_both_reports_is_not_introduced() -> None:
    row = _status_move(_LANDED, _Y)

    assert _introduced([row], [row]) == ()


@pytest.mark.parametrize(
    "union_row",
    [
        _status_move(_LANDED, _Y, before=KnowledgeStatus.REJECTED, after=KnowledgeStatus.PROPOSED),
        _status_move(_LANDED, _Y, kind="lowers"),
    ],
    ids=["other-before-after", "other-kind"],
)
def test_a_row_under_a_held_key_is_not_introduced_whatever_its_values(
    union_row: PermissiveMove,
) -> None:
    """The key, not the whole row, is the identity: a pre-existing row must not refuse."""
    held = _status_move(_LANDED, _Y)
    assert union_row != held, "the union's row must differ from the held one for this to bite"

    assert _introduced([held], [union_row]) == ()


@pytest.mark.parametrize(
    "reattributed",
    [
        replace(_status_move(_LANDED, _Y), undoes=_INCOMING),
        _status_move(_LANDED, _Y, kind=None),
        replace(_sensitivity_move(_LANDED, _Y), undoes=_INCOMING),
    ],
    ids=["status-undoes-the-incoming", "no-writer-recorded", "sensitivity-undoes-the-incoming"],
)
def test_a_row_under_a_held_key_whose_undoes_differs_is_introduced(
    reattributed: PermissiveMove,
) -> None:
    """GHSA-v2qg: a raise replayed between a row and what it undid re-attributes the row."""
    held = (
        _status_move(_LANDED, _Y)
        if reattributed.field == "status"
        else _sensitivity_move(_LANDED, _Y)
    )
    assert reattributed.undoes != held.undoes

    assert _introduced([held], [reattributed]) == (reattributed,)


@pytest.mark.parametrize(
    "other",
    [
        _status_move(_INCOMING, _Y),
        _status_move(_LANDED, _X),
        _sensitivity_move(_LANDED, _Y),
    ],
    ids=["other-migration", "other-item", "other-field"],
)
def test_a_row_differing_from_a_held_row_in_any_key_part_is_introduced(
    other: PermissiveMove,
) -> None:
    held = _status_move(_LANDED, _Y)

    assert _introduced([held], [held, other]) == (other,)


def test_an_empty_union_introduces_nothing() -> None:
    assert _introduced([_status_move(_LANDED)], []) == ()
    assert _introduced([], []) == ()


def test_introduced_rows_keep_the_unions_order() -> None:
    first = _status_move(_INCOMING, _Y)
    held = _status_move(_LANDED, _Y)
    second = _sensitivity_move(_INCOMING, _X)
    third = _status_move(_INCOMING, _X)

    assert _introduced([held], [first, held, second, third]) == (first, second, third)
