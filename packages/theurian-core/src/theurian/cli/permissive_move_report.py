"""The permissive-move report as ``migrate validate`` and ``migrate apply`` print it.

Operator-side by decision (GHSA-v2qg-23fc-7fqp): a row names the labels an item
held before the migration of an upsert that loosened it.
``test_no_served_path_mentions_the_report`` holds ``permissiveMoves`` out of every
file under ``mcp/`` and ``schemas/mcp/``.
"""

from __future__ import annotations

from collections.abc import Sequence

from theurian.application.permissive_moves import PermissiveMove


def permissive_move_rows(
    moves: Sequence[PermissiveMove], *, as_json: bool
) -> list[dict[str, str | None]] | list[str]:
    """One object per row for ``--json``; otherwise one line per row."""
    if as_json:
        return [_as_object(move) for move in moves]
    return [_as_line(move) for move in moves]


def _as_object(move: PermissiveMove) -> dict[str, str | None]:
    return {
        "migrationId": str(move.migration_id),
        "itemId": str(move.item_id),
        "field": move.field,
        "before": move.before.value,
        "after": move.after.value,
        "undoes": None if move.undoes is None else str(move.undoes),
        "kind": move.kind,
    }


def _as_line(move: PermissiveMove) -> str:
    if move.undoes is None:
        written = "last written by an earlier apply"
    elif move.kind == "undoes":
        written = f"undoes {move.undoes}"
    else:
        written = f"lowers what {move.undoes} set"
    return (
        f"{move.migration_id} moves {move.item_id} {move.field} "
        f"{move.before.value} -> {move.after.value}: {written}"
    )
