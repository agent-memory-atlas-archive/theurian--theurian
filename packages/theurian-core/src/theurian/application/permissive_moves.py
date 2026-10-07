"""Replayed ``upsertRevision`` operations that leave an item more permissive (GHSA-v2qg-23fc-7fqp).

The accept floors compare at accept time, so a withdrawal merged after an accept
but sorting before it replays first and is then undone by the accepted upsert's
``status`` / ``sensitivity``. Refusing that would stop histories that already
apply, so the engine reports it instead and no exit code moves. A field is
reported when an upsert itself loosened it and the upsert's migration, at its
end, left it looser than the migration found it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from theurian.application.item_labels import is_lowering
from theurian.domain.enums import KnowledgeStatus, Sensitivity, may_surface
from theurian.domain.identifiers import ItemId, MigrationId
from theurian.domain.knowledge import KnowledgeItem
from theurian.domain.migration import ChangeSensitivity, CreateItem, DeprecateItem, RestoreItem

LabelField = Literal["status", "sensitivity"]
MoveKind = Literal["undoes", "lowers"]

_LABELS: Final[tuple[LabelField, ...]] = ("status", "sensitivity")

#: The operations besides ``upsertRevision`` that write a label.
LabelWrite = CreateItem | DeprecateItem | RestoreItem | ChangeSensitivity


@dataclass(frozen=True, slots=True)
class PermissiveMove:
    """One label an ``upsertRevision`` loosened, on an item its migration left looser.

    ``before`` and ``after`` are the label at the migration's start and at its end.
    ``undoes`` is the migration that last changed whether the item may surface
    (status) or its class (sensitivity) before the upsert's migration. ``kind`` is
    decided by that change's effect, not its operation: ``undoes`` when it tightened
    the field -- status from surfaceable to not, sensitivity raised -- on an item that
    existed when its migration began, otherwise ``lowers``. A write between two
    retired statuses is not a change.
    Both are ``None`` when this replay recorded no writer for the field, because an
    earlier run set it. With no writer recorded the change rule does not hold: a
    retired-to-retired write is still recorded, as no withdrawal, so a readmission
    can read ``lowers`` naming it.
    """

    migration_id: MigrationId
    item_id: ItemId
    field: LabelField
    before: KnowledgeStatus | Sensitivity
    after: KnowledgeStatus | Sensitivity
    undoes: MigrationId | None
    kind: MoveKind | None

    def __post_init__(self) -> None:
        label_type = KnowledgeStatus if self.field == "status" else Sensitivity
        if not (isinstance(self.before, label_type) and isinstance(self.after, label_type)):
            raise TypeError(f"a {self.field} move needs {label_type.__name__} values")
        if (self.undoes is None) != (self.kind is None):
            raise ValueError("undoes and kind are known together or not at all")


@dataclass(frozen=True, slots=True)
class Overwrite:
    """A field a migration wrote, on the item as the migration found it and as its end left it.

    ``found`` is ``left`` on an item the migration created. Not a report row: ``accept``
    reads these.
    """

    migration_id: MigrationId
    item_id: ItemId
    field: LabelField
    found: KnowledgeItem
    left: KnowledgeItem

    @property
    def before(self) -> KnowledgeStatus | Sensitivity:
        return _value(self.found, self.field)

    @property
    def after(self) -> KnowledgeStatus | Sensitivity:
        return _value(self.left, self.field)


def loosened_after(
    overwrites: Sequence[Overwrite], migration_id: MigrationId | None
) -> tuple[Overwrite, ...]:
    """Per field ``migration_id`` wrote that ends looser than it left it, by the accept floors'
    predicates, the write that last took it below that level."""
    level: dict[tuple[ItemId, LabelField], KnowledgeItem] = {}
    below: dict[tuple[ItemId, LabelField], Overwrite | None] = {}
    for o in overwrites:
        key = (o.item_id, o.field)
        if o.migration_id == migration_id:
            level[key], below[key] = o.left, None
        elif key in level:
            now, was = (_loosens(level[key], item, o.field) for item in (o.left, o.found))
            if now != was:
                below[key] = o if now else None
    return tuple(o for o in below.values() if o is not None)


def introduced_moves(
    held: Sequence[PermissiveMove], union: Sequence[PermissiveMove]
) -> tuple[PermissiveMove, ...]:
    """The union rows whose ``(migration_id, item_id, field, undoes)`` no held row has.

    The held rows are the landed-alone replay's report. A row whose key matches one
    is the history's, not the accept's, even if its ``before``, ``after`` or ``kind``
    differ. ``undoes`` is in the key because the incoming migration can change
    which migration a held row undoes: by becoming the field's last writer
    itself, or by turning a landed write into a change, as a restore replaying
    before a landed deprecation of an already-deprecated item does.
    """
    known = {(m.migration_id, m.item_id, m.field, m.undoes) for m in held}
    return tuple(m for m in union if (m.migration_id, m.item_id, m.field, m.undoes) not in known)


@dataclass(frozen=True, slots=True)
class _Writer:
    migration_id: MigrationId
    withdrawal: bool


@dataclass(frozen=True, slots=True)
class _Held:
    """An item as a migration found it, and who had written its labels."""

    migration_id: MigrationId
    item: KnowledgeItem
    writers: Mapping[LabelField, _Writer]


def _loosens(before: KnowledgeItem, after: KnowledgeItem, label: LabelField) -> bool:
    """By the accept floors' predicates, for an upsert's own move and for its migration's.

    Read the other way round it is whether a label write tightened the field.
    """
    if label == "status":
        return not may_surface(before.status, include_unapproved=True) and may_surface(
            after.status, include_unapproved=True
        )
    return is_lowering(before.sensitivity, after.sensitivity)


def _value(item: KnowledgeItem, label: LabelField) -> KnowledgeStatus | Sensitivity:
    return item.status if label == "status" else item.sensitivity


def _labels_of(operation: LabelWrite) -> tuple[LabelField, ...]:
    match operation:
        case CreateItem():
            return _LABELS
        case DeprecateItem() | RestoreItem():
            return ("status",)
        case ChangeSensitivity():
            return ("sensitivity",)


def _row(held: _Held, label: LabelField, end: KnowledgeItem) -> PermissiveMove:
    writer = held.writers.get(label)
    return PermissiveMove(
        migration_id=held.migration_id,
        item_id=end.item_id,
        field=label,
        before=_value(held.item, label),
        after=_value(end, label),
        undoes=None if writer is None else writer.migration_id,
        kind=None if writer is None else ("undoes" if writer.withdrawal else "lowers"),
    )


class LabelWriters:
    """Per item and label, the migration that last changed it; and the moves found so far.

    Fed in replay order by ``MigrationEngine.apply``, so it knows only the writes
    of the run that feeds it. A migration's rows are decided at its end: when the
    next migration's first label write arrives, or, for the last, when
    :attr:`moves` is read.
    """

    def __init__(self) -> None:
        self._last: dict[tuple[ItemId, LabelField], _Writer] = {}
        self._moves: list[PermissiveMove] = []
        self._migration: MigrationId | None = None
        # Per item the current migration has written a label of: as the migration found
        # it (`None` when it found no such item), and as its latest label write left it.
        self._held: dict[ItemId, _Held | None] = {}
        self._latest: dict[ItemId, KnowledgeItem] = {}
        # The fields an upsert in the current migration loosened, in replay order.
        self._loosened: dict[tuple[ItemId, LabelField], _Held] = {}
        self._written: dict[tuple[ItemId, LabelField], MigrationId] = {}
        self._overwrites: list[Overwrite] = []

    @property
    def moves(self) -> tuple[PermissiveMove, ...]:
        """Every migration's rows; the last migration fed is decided as its writes stand."""
        return (*self._moves, *self._settled())

    @property
    def overwrites(self) -> tuple[Overwrite, ...]:
        """Every migration's, decided as :attr:`moves` is."""
        return (*self._overwrites, *self._overwritten())

    def wrote(
        self,
        migration_id: MigrationId,
        operation: LabelWrite,
        *,
        prior: KnowledgeItem | None,
        landed: KnowledgeItem,
    ) -> None:
        """Record an operation that wrote a label; a ``createItem`` only when it created.

        ``prior`` and ``landed`` are the item as the operation read it and as it wrote it.
        """
        held = self._track(migration_id, prior, landed)
        for label in _labels_of(operation):
            self._record(migration_id, held, prior, landed, label)

    def upserted(
        self, migration_id: MigrationId, prior: KnowledgeItem | None, landed: KnowledgeItem
    ) -> None:
        """Note each label the upsert loosened; then the upsert is the writer of each it changed.

        A label is noted when the upsert loosened it against ``prior``, the item just
        before the operation, and reported only if the migration, at its end, leaves
        it looser than it found it. The first keeps a content update that restates
        what a ``restoreItem`` or ``changeSensitivity`` earlier in its migration set
        from reading as the loosening. The second is the floors' own granularity:
        they compare the landed state before a proposal's migration with the state
        after it, and never see a new item. So an item the migration creates is not
        compared, and a migration that withdraws and re-asserts an item, or lowers a
        label and restores it, nets out -- one migration is one reviewed diff.
        """
        held = self._track(migration_id, prior, landed)
        if held is not None and prior is not None:
            for label in _LABELS:
                if _loosens(prior, landed, label):
                    self._loosened.setdefault((landed.item_id, label), held)
        for label in _LABELS:
            self._record(migration_id, held, prior, landed, label)

    def _track(
        self, migration_id: MigrationId, prior: KnowledgeItem | None, landed: KnowledgeItem
    ) -> _Held | None:
        """The item as ``migration_id`` found it: ``prior`` at the migration's first label write.

        A new id settles the migration before it.
        """
        if migration_id != self._migration:
            # Ids are unique within a set and fed in replay order: a new id is a new migration.
            self._moves.extend(self._settled())
            self._overwrites.extend(self._overwritten())
            self._migration = migration_id
            self._held = {}
            self._latest = {}
            self._loosened = {}
            self._written = {}
        item_id = landed.item_id
        if item_id not in self._held:
            self._held[item_id] = (
                None
                if prior is None
                else _Held(
                    migration_id,
                    prior,
                    {
                        label: self._last[item_id, label]
                        for label in _LABELS
                        if (item_id, label) in self._last
                    },
                )
            )
        self._latest[item_id] = landed
        return self._held[item_id]

    def _overwritten(self) -> list[Overwrite]:
        """The current migration's: each field it wrote."""
        writes: list[Overwrite] = []
        for (item_id, label), migration_id in self._written.items():
            held, left = self._held[item_id], self._latest[item_id]
            found = left if held is None else held.item
            writes.append(Overwrite(migration_id, item_id, label, found, left))
        return writes

    def _settled(self) -> list[PermissiveMove]:
        """The current migration's rows: each noted field its end leaves looser than its start."""
        return [
            _row(held, label, self._latest[item_id])
            for (item_id, label), held in self._loosened.items()
            if _loosens(held.item, self._latest[item_id], label)
        ]

    def _record(
        self,
        migration_id: MigrationId,
        held: _Held | None,
        prior: KnowledgeItem | None,
        landed: KnowledgeItem,
        label: LabelField,
    ) -> None:
        """Make the write ``label``'s writer, unless it left the field as it found it.

        Either way it is one of the fields the migration wrote.

        A write on an item its migration created (``held`` is ``None``) is the item's
        initial labelling, as a ``createItem`` is, so it is no withdrawal: a root
        migration's upsert stating ``confidential`` over its ``createItem``'s default
        ``internal`` withdrew nothing a reader had been served.

        "As it found it" is the gate's own granularity. A status write that changes the
        value and not ``may_surface`` leaves a recorded writer in place, so draft ->
        proposed after a retirement does not displace the withdrawal. Sensitivity is
        totally ordered by ``DISCLOSURE_ORDER``: a changed value is a changed class.
        """
        self._written[landed.item_id, label] = migration_id
        tightened = prior is not None and _loosens(landed, prior, label)
        if (
            prior is not None
            and not tightened
            and not _loosens(prior, landed, label)
            and (
                _value(prior, label) == _value(landed, label)
                or (landed.item_id, label) in self._last
            )
        ):
            return
        withdrawal = held is not None and tightened
        self._last[landed.item_id, label] = _Writer(migration_id, withdrawal)
