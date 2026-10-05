"""``current_revision_in`` and ``item_named_in`` disagree on a create-only item, on purpose.

An item with a ``createItem`` and no ``upsertRevision`` exists and holds labels
(GHSA-v2qg-23fc-7fqp), but has no revision. A drafter that reads "no revision" as
"no item" re-labels it through its first revision, so the two answers must stay
apart: no current revision, and still named.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from theurian.domain.enums import KnowledgeKind, KnowledgeStatus, Sensitivity, TrustLevel
from theurian.domain.identifiers import ItemId, MigrationId, RevisionId
from theurian.domain.migration import (
    CreateItem,
    DeprecateItem,
    Migration,
    MigrationSet,
    RevisionMetadataSpec,
    UpsertRevision,
    current_revision_in,
    item_named_in,
)
from theurian.domain.values import MARKDOWN, ContentHash

pytestmark = pytest.mark.unit

CREATED = ItemId("architecture.placeholder")
OTHER = ItemId("architecture.other")
MIGRATION_ID = "01K1AAAAAA01234567890ABCDE"


def _set(*operations: CreateItem | DeprecateItem | UpsertRevision) -> MigrationSet:
    return MigrationSet(
        (
            Migration(
                migration_id=MigrationId(MIGRATION_ID),
                created_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
                author="engineer@example.com",
                operations=tuple(operations),
                checksum=ContentHash.of_text(MIGRATION_ID),
            ),
        )
    )


def _create(item_id: ItemId) -> CreateItem:
    return CreateItem(
        item_id=item_id,
        kind_=KnowledgeKind.ARCHITECTURE,
        namespace="security",
        owner="platform-team",
    )


def test_a_create_only_item_has_no_current_revision_and_is_still_named() -> None:
    migrations = _set(_create(CREATED))

    assert current_revision_in(migrations, CREATED) is None
    assert item_named_in(migrations, CREATED) is True


def test_an_id_no_migration_creates_is_not_named() -> None:
    """Control: a set holding other ids, and an operation that only acts on an existing item."""
    migrations = _set(_create(OTHER), DeprecateItem(item_id=CREATED))

    assert current_revision_in(migrations, CREATED) is None
    assert item_named_in(migrations, CREATED) is False


def test_an_item_created_by_a_bare_first_revision_is_named_and_has_a_current_revision() -> None:
    """No ``createItem``: the ``upsertRevision`` alone creates the item, so it must be named.

    A drafter that asked only about ``createItem`` would treat this id as brand
    new and re-label it through its next revision.
    """
    revision = RevisionId("01K1AAAREV01234567890ABCDE")
    migrations = _set(
        UpsertRevision(
            item_id=CREATED,
            revision_id=revision,
            content_file_path="../knowledge/a.md",
            metadata=RevisionMetadataSpec(
                title="Doc",
                content_type=MARKDOWN,
                kind=KnowledgeKind.ARCHITECTURE,
                namespace="security",
                status=KnowledgeStatus.APPROVED,
                owner="platform-team",
                trust_level=TrustLevel.REVIEWED,
                sensitivity=Sensitivity.INTERNAL,
            ),
        )
    )

    assert current_revision_in(migrations, CREATED) == revision
    assert item_named_in(migrations, CREATED) is True
    assert item_named_in(migrations, OTHER) is False
