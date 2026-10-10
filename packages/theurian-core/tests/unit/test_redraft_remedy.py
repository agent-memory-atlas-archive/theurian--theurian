"""The redraft remedy routes by every kind the refused proposal carries (GHSA-v2qg-23fc-7fqp).

A remedy that names only the kinds of one route sends its reader to author a
fraction of the proposal: a restore-then-deprecate proposal told to author
``restoreItem`` alone readmits an item the proposal meant to end withdrawn.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pytest

from theurian.application.proposal_service import (
    _REFUSED_TO_CLI,
    _REFUSED_TO_CONTENT_PATH,
    V1_OPERATION_KINDS,
    _ProposalLocation,
    _redraft_remedy,
)
from theurian.domain.identifiers import ItemId, MigrationId
from theurian.domain.migration import Migration, RestoreItem
from theurian.domain.values import ContentHash

pytestmark = pytest.mark.unit

_CONTENT = sorted(k.value for k in _REFUSED_TO_CONTENT_PATH)
_V1 = sorted(k.value for k in V1_OPERATION_KINDS)
_CLI = sorted(k.value for k in _REFUSED_TO_CLI)

_PROPOSE = "`theurian propose`"
_GENERATE = "`knowledge.generateMigrationDraft`"
_ACCEPT = "`theurian propose accept`"

_ROOT = MigrationId("01K2AAAAAAAAAAAAAAAAAAAAAA")
_DEPENDENT = MigrationId("01K2BBBBBBBBBBBBBBBBBBBBBB")
_LOCATION = _ProposalLocation(
    directory=Path("proposals"), relative=".theurian/proposals/p1", local=False
)


def _migration(migration_id: MigrationId, *depends_on: MigrationId) -> Migration:
    """The cheapest valid migration: one restore, a zero checksum, no file behind it."""
    return Migration(
        migration_id=migration_id,
        created_at=datetime(2026, 10, 2, tzinfo=UTC),
        author="engineer@example.com",
        operations=(RestoreItem(item_id=ItemId("architecture.a")),),
        checksum=ContentHash("0" * 64),
        depends_on=depends_on,
    )


def _remedy(kinds: set[str], *, after: list[Migration] | None = None) -> str:
    return _redraft_remedy(frozenset(kinds), "THE-LANDED-ID", after or [], _LOCATION)


def _route_of(kinds: set[str]) -> str:
    if kinds <= set(_CONTENT):
        return "propose"
    if kinds <= set(_V1):
        return "generate"
    return "author"


def _assert_route(remedy: str, route: str) -> None:
    assert remedy.startswith("Nothing has moved. "), remedy
    assert (_PROPOSE in remedy) is (route == "propose"), remedy
    assert (_GENERATE in remedy) is (route == "generate"), remedy
    assert (_ACCEPT in remedy) is (route == "author"), remedy
    assert "migrate apply" not in remedy, remedy
    if route == "author":
        assert "this proposal's migration file in .theurian/proposals/p1/" in remedy, remedy
        assert "Author the " not in remedy, remedy
        assert remedy.endswith(f", then run {_ACCEPT} again."), remedy
    else:
        assert remedy.endswith(" Then delete .theurian/proposals/p1/."), remedy
        assert not re.search(r"accept[^.]*again", remedy, re.IGNORECASE), remedy


def test_the_kind_sets_are_disjoint_and_non_empty() -> None:
    """A route test over an empty or overlapping set asserts nothing about routing."""
    assert _CONTENT and _V1 and _CLI
    assert len(set(_CONTENT) | set(_V1) | set(_CLI)) == len(_CONTENT) + len(_V1) + len(_CLI)


def test_a_content_only_proposal_is_sent_to_theurian_propose() -> None:
    kinds = set(_CONTENT)

    _assert_route(_remedy(kinds), "propose")


@pytest.mark.parametrize("kind", _CONTENT)
def test_each_content_kind_alone_is_sent_to_theurian_propose(kind: str) -> None:
    _assert_route(_remedy({kind}), "propose")


@pytest.mark.parametrize("kind", _V1)
def test_each_v1_kind_alone_is_sent_to_generate_migration_draft(kind: str) -> None:
    _assert_route(_remedy({kind}), "generate")


@pytest.mark.parametrize("kind", _CLI)
def test_each_cli_kind_alone_is_edited_into_its_own_file_and_accepted_again(kind: str) -> None:
    remedy = _remedy({kind})

    _assert_route(remedy, "author")
    assert "/, so it replays after THE-LANDED-ID only through its `dependsOn`, then" in remedy


def test_a_restore_then_deprecate_proposal_is_edited_in_its_own_file() -> None:
    """Face 4: authoring ``restoreItem`` alone readmits where the proposal withdrew."""
    kinds = {"restoreItem", "deprecateItem"}

    _assert_route(_remedy(kinds), "author")


def test_an_upsert_with_a_v1_kind_is_edited_in_its_own_file() -> None:
    """``generateMigrationDraft`` refuses content; ``theurian propose`` cannot draft the v1 kind."""
    kinds = {"upsertRevision", "deprecateItem"}

    _assert_route(_remedy(kinds), "author")


_PAIRS = [set(pair) for pair in combinations(sorted(_CONTENT + _V1 + _CLI), 2)]


@pytest.mark.parametrize("kinds", _PAIRS, ids=lambda kinds: "+".join(sorted(kinds)))
def test_every_pair_of_kinds_is_routed_to_a_tool_that_can_draft_all_of_it(
    kinds: set[str],
) -> None:
    """The population is every unordered pair of the live kind constants, none picked."""
    _assert_route(_remedy(kinds), _route_of(kinds))


_ALL_CASES = [
    *({k} for k in _CONTENT),
    *({k} for k in _V1),
    *({k} for k in _CLI),
    {"restoreItem", "deprecateItem"},
    {"upsertRevision", "deprecateItem"},
]
_FORMS = {
    "propose": r"edit `dependsOn: \[{ids}\]` into the drafted migration file",
    "generate": r"with `dependsOn: \[{ids}\]` in its document",
    "author": r"Edit `dependsOn: \[{ids}\]` into this proposal's migration file",
}


_AFTERS = {
    "root-only": [_migration(_ROOT)],
    "dependent-only": [_migration(_DEPENDENT, _ROOT)],
    "root-and-dependent": [_migration(_ROOT), _migration(_DEPENDENT, _ROOT)],
}


@pytest.mark.parametrize("after_id", sorted(_AFTERS))
@pytest.mark.parametrize("kinds", _ALL_CASES, ids=lambda kinds: "+".join(sorted(kinds)))
def test_dependson_names_every_landed_migration_whether_or_not_it_declares_one(
    kinds: set[str], after_id: str
) -> None:
    """A landed id can sort after the drafting clock, so no id order puts a fresh draft after it.

    A hand-chosen id, an id edited before accept and a collaborator's fast clock all
    do; only ``dependsOn`` places a migration after another, so the remedy always
    names it, and never claims the later id does the work.
    """
    after = _AFTERS[after_id]
    ids = ", ".join(m.migration_id.value for m in after)

    remedy = _remedy(kinds, after=after)

    assert re.search(_FORMS[_route_of(kinds)].format(ids=re.escape(ids)), remedy), remedy
    assert "later migration id" not in remedy, remedy
    _assert_route(remedy, _route_of(kinds))


def test_every_landed_migration_is_listed_in_the_clause_in_the_order_given() -> None:
    second = MigrationId("01K2CCCCCCCCCCCCCCCCCCCCCC")

    remedy = _remedy(
        {"restoreItem"},
        after=[_migration(_DEPENDENT, _ROOT), _migration(second, _ROOT), _migration(_ROOT)],
    )

    expected = f"dependsOn: [{_DEPENDENT.value}, {second.value}, {_ROOT.value}]"
    assert expected in remedy, remedy


_AUTHORED = sorted(
    {frozenset(k) for k in (*_ALL_CASES, *_PAIRS) if _route_of(k) == "author"},
    key=sorted,
)


@pytest.mark.parametrize("kinds", _AUTHORED, ids=lambda kinds: "+".join(sorted(kinds)))
def test_a_hand_authored_remedy_routes_through_accept_never_to_migrate_apply(
    kinds: frozenset[str],
) -> None:
    """``migrate apply`` re-checks nothing; ``propose accept`` re-judges (GHSA-fjqq)."""
    remedy = _remedy(set(kinds), after=[_migration(_ROOT)])

    assert "`theurian propose accept`" in remedy and "migrate apply" not in remedy, remedy
    assert f"dependsOn: [{_ROOT.value}]" in remedy, remedy
