"""``PermissiveMove`` refuses a half-known writer (GHSA-v2qg-23fc-7fqp).

``undoes`` and ``kind`` come from one fact, who last wrote the field. A move
carrying one without the other would render a row that names a migration it
cannot classify, or a kind with nothing to point at.
"""

from __future__ import annotations

import pytest

from theurian.application.permissive_moves import PermissiveMove
from theurian.domain.enums import KnowledgeStatus
from theurian.domain.identifiers import ItemId, MigrationId

pytestmark = pytest.mark.unit

_MIGRATION = MigrationId("01K1CCCCCC01234567890ABCDE")
_EARLIER = MigrationId("01K1BBBBBB01234567890ABCDE")
_ITEM = ItemId("auth-policy")


def _move(*, undoes: MigrationId | None, kind: str | None) -> PermissiveMove:
    return PermissiveMove(
        migration_id=_MIGRATION,
        item_id=_ITEM,
        field="status",
        before=KnowledgeStatus.DEPRECATED,
        after=KnowledgeStatus.APPROVED,
        undoes=undoes,
        kind=kind,  # type: ignore[arg-type]
    )


def test_a_move_with_both_undoes_and_kind_or_neither_is_accepted() -> None:
    assert _move(undoes=_EARLIER, kind="undoes").kind == "undoes"
    assert _move(undoes=None, kind=None).undoes is None


@pytest.mark.parametrize(("undoes", "kind"), [(_EARLIER, None), (None, "lowers")])
def test_a_move_naming_only_one_of_undoes_and_kind_is_refused(
    undoes: MigrationId | None, kind: str | None
) -> None:
    with pytest.raises(ValueError, match="known together"):
        _move(undoes=undoes, kind=kind)
