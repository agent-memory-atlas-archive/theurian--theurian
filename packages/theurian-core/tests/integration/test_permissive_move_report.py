"""``migrate validate`` and ``apply`` report an upsert that loosens an item (GHSA-v2qg-23fc-7fqp).

An update accepted after the item was withdrawn by an earlier-sorting
hand-authored migration replays the withdrawal first and the accepted upsert's
``status: approved`` / ``sensitivity: internal`` second, nullifying the reviewed
withdrawal; both commands exited 0 on it silently. The report names each replayed
upsert that itself loosens a label of an item its migration, at its end, leaves
more permissive than the replay held it when that migration began; exit codes are
unchanged, so existing histories keep applying.

``kind`` separates the race (``undoes``: before the upsert's migration, the field
was last changed by a write that tightened it) from an omitted label or a
hand-authored lowering (``lowers``: it was last changed by a write that did not,
such as a ``createItem`` or a declassification).

Every fixture sits on ROOT migrations (no ``dependsOn``): dependency ordering is
a separate defect (#863).
"""

from __future__ import annotations

import ast
import errno
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Final

import pytest
from git_harness import commit_migrations
from label_inheritance_support import (
    BODY,
    CREATE_ONLY_ITEM_ID,
    ITEM_ID,
    ROOT_MIGRATION_ID,
    ROOT_REVISION_ID,
    SORTS_BEFORE_A_DRAFT,
    LabelledProject,
    cli,
    cli_ok,
    cli_propose,
    create_only_project,
    deprecation,
    item_row,
    labelled_project,
    reclassification,
    runner,
)
from migration_fixtures import body_pin

from theurian.cli.main import app

pytestmark = pytest.mark.integration

_HAND_MIGRATION: Final = "01K1CCCCCC01234567890ABCDE"
_HAND_REVISION: Final = "01K1CCCREV01234567890ABCDE"
_EARLIER_MIGRATION: Final = "01K1BBBBBB01234567890ABCDE"
_EARLIER_REVISION: Final = "01K1BBBREV01234567890ABCDE"
_RESTORE_MIGRATION: Final = "01K1DDDDDD01234567890ABCDE"
_LATEST_MIGRATION: Final = "01K1EEEEEE01234567890ABCDE"
_LATEST_REVISION: Final = "01K1EEEREV01234567890ABCDE"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    assert "permissiveMoves" in payload, sorted(payload)
    rows = payload["permissiveMoves"]
    assert isinstance(rows, list), payload
    return rows


def _write_migration(project: LabelledProject, migration_id: str, body: str) -> None:
    """Write a hand-authored migration, and the body file of its own an upsert needs.

    One body file cannot back two revisions, so each hand upsert gets a file named
    for its revision, holding the same bytes.
    """
    (project.root / f".theurian/migrations/{migration_id}-hand.yaml").write_text(body)
    for revision in _hand_revisions(body):
        (project.root / f".theurian/knowledge/architecture/hand-{revision}.md").write_text(BODY)


def _hand_revisions(body: str) -> list[str]:
    return [
        line.split("hand-")[1].removesuffix(".md")
        for line in body.splitlines()
        if "contentFile: ../knowledge/architecture/hand-" in line
    ]


def _upsert_operation(
    item_id: str,
    revision_id: str,
    *,
    expected_revision: str | None,
    status: str,
    sensitivity: str | None,
) -> str:
    expected = f"    expectedRevision: {expected_revision}\n" if expected_revision else ""
    label = f"      sensitivity: {sensitivity}\n" if sensitivity else ""
    return f"""  - op: upsertRevision
    itemId: {item_id}
    revisionId: {revision_id}
{expected}    contentFile: ../knowledge/architecture/hand-{revision_id}.md
    contentSha256: {body_pin(BODY)}
    metadata:
      title: Authentication policy
      contentType: text/markdown
      kind: architecture
      namespace: backend
      status: {status}
      owner: platform-team
      trustLevel: reviewed
{label}      sourceAnchors:
        - provider: git
          sourceUri: git://demo/auth-policy.md
"""


def _migration(migration_id: str, *operations: str) -> str:
    return (
        f"apiVersion: theurian.dev/v1\nid: {migration_id}\n"
        "createdAt: 2026-09-30T10:00:00+09:00\nauthor: engineer@example.com\n"
        "operations:\n" + "".join(operations)
    )


def _upsert_migration(  # noqa: PLR0913 -- the labels an upsert can carry
    migration_id: str,
    revision_id: str,
    *,
    expected_revision: str | None,
    status: str,
    sensitivity: str | None,
    item_id: str = ITEM_ID,
) -> str:
    return _migration(
        migration_id,
        _upsert_operation(
            item_id,
            revision_id,
            expected_revision=expected_revision,
            status=status,
            sensitivity=sensitivity,
        ),
    )


def _reports(*, apply_too: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``migrate validate`` and (when asked) ``migrate apply``, both expected to exit 0."""
    validated = cli_ok("migrate", "validate")
    return validated, (cli_ok("migrate", "apply") if apply_too else None)


def _race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face: str
) -> tuple[LabelledProject, dict[str, Any], str]:
    """An accepted, applied update, then a withdrawal that sorts before it, unapplied."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted
    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the withdrawal must sort first"
    late = (
        deprecation(SORTS_BEFORE_A_DRAFT, ITEM_ID)
        if face == "status"
        else reclassification(SORTS_BEFORE_A_DRAFT, ITEM_ID, "confidential")
    )
    _write_migration(project, SORTS_BEFORE_A_DRAFT, late)
    return project, drafted, SORTS_BEFORE_A_DRAFT


def _text(*args: str) -> str:
    if args[:2] == ("migrate", "apply"):
        commit_migrations()
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


# -- The race ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "before", "after"),
    [("status", "deprecated", "approved"), ("sensitivity", "confidential", "internal")],
)
def test_an_update_accepted_after_a_withdrawal_that_sorts_first_is_reported_undoing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, before: str, after: str
) -> None:
    """The silent nullification of a reviewed withdrawal is the defect; a report is the remedy.

    Both commands must carry the row, and the item must still end loosened: the
    report is not a fix, and a test asserting otherwise would hide that.
    """
    project, drafted, withdrawal = _race(tmp_path, monkeypatch, field)
    expected = [
        {
            "migrationId": drafted["migrationId"],
            "itemId": ITEM_ID,
            "field": field,
            "before": before,
            "after": after,
            "undoes": withdrawal,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected
    row = item_row(project.root, ITEM_ID)
    assert (row["status"], row["sensitivity"]) == ("approved", "internal")


def test_the_text_report_carries_one_line_per_row_with_every_identifying_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator reading text must be able to tell which migration undid which."""
    _, drafted, withdrawal = _race(tmp_path, monkeypatch, "status")
    tokens = [drafted["migrationId"], ITEM_ID, "status", "deprecated", "approved", withdrawal]

    for command in (("migrate", "validate"), ("migrate", "apply")):
        lines = _text(*command).splitlines()

        assert any(all(token in line for token in tokens) and "undoes" in line for line in lines), (
            command,
            lines,
        )


# -- Controls: what is not reported -------------------------------------------------


def test_a_plain_update_reports_no_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A control: without it a report that fires on every update would pass the race tests."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted
    cli_ok("propose", "accept", drafted["proposalId"])

    validated, applied = _reports()

    assert applied is not None
    assert validated.get("permissiveMoves") == []
    assert applied.get("permissiveMoves") == []


def test_a_first_revision_taking_a_draft_item_to_approved_reports_no_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``draft`` is surfaceable for an operator, so draft to approved loosens nothing."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, project.item_id)
    assert code == 0, drafted
    cli_ok("propose", "accept", drafted["proposalId"])
    assert item_row(project.root, project.item_id)["status"] == "draft"

    validated, applied = _reports()

    assert applied is not None
    assert validated.get("permissiveMoves") == []
    assert applied.get("permissiveMoves") == []
    assert item_row(project.root, project.item_id)["status"] == "approved"


def test_a_proposed_item_taken_to_approved_by_a_later_migration_reports_no_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``proposed`` is surfaceable for an operator, so proposed to approved loosens nothing."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    _write_migration(
        project,
        _EARLIER_MIGRATION,
        _upsert_migration(
            _EARLIER_MIGRATION,
            _EARLIER_REVISION,
            expected_revision=None,
            status="proposed",
            sensitivity="internal",
            item_id=CREATE_ONLY_ITEM_ID,
        ),
    )
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=_EARLIER_REVISION,
            status="approved",
            sensitivity="internal",
            item_id=CREATE_ONLY_ITEM_ID,
        ),
    )

    validated, applied = _reports()

    assert applied is not None
    assert validated.get("permissiveMoves") == []
    assert applied.get("permissiveMoves") == []
    assert item_row(project.root, CREATE_ONLY_ITEM_ID)["status"] == "approved"


def _raising_upsert(project: LabelledProject) -> None:
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=ROOT_REVISION_ID,
            status="approved",
            sensitivity="confidential",
        ),
    )


def _retiring_upsert(project: LabelledProject) -> None:
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=ROOT_REVISION_ID,
            status="deprecated",
            sensitivity="internal",
        ),
    )


def _sanctioned_lowering(project: LabelledProject) -> None:
    _write_migration(
        project, _HAND_MIGRATION, reclassification(_HAND_MIGRATION, ITEM_ID, "internal")
    )


def _restoration(project: LabelledProject) -> None:
    _write_migration(project, _EARLIER_MIGRATION, deprecation(_EARLIER_MIGRATION, ITEM_ID))
    _write_migration(
        project,
        _RESTORE_MIGRATION,
        f"""apiVersion: theurian.dev/v1
id: {_RESTORE_MIGRATION}
createdAt: 2026-09-30T10:00:00+09:00
author: engineer@example.com
operations:
  - op: restoreItem
    itemId: {ITEM_ID}
""",
    )


@pytest.mark.parametrize(
    ("operation", "labels", "ends"),
    [
        (_raising_upsert, "internal", ("approved", "confidential")),
        (_retiring_upsert, "internal", ("deprecated", "internal")),
        (_sanctioned_lowering, "confidential", ("approved", "internal")),
        (_restoration, "internal", ("approved", "internal")),
    ],
    ids=["raising-upsert", "retiring-upsert", "sanctioned-lowering", "restoration"],
)
def test_only_a_loosening_upsert_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    labels: str,
    ends: tuple[str, str],
) -> None:
    """Raising and retiring upserts and the sanctioned operations: the one-directional controls."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity=labels)
    operation(project)

    validated, applied = _reports()

    assert applied is not None
    assert validated.get("permissiveMoves") == []
    assert applied.get("permissiveMoves") == []
    row = item_row(project.root, ITEM_ID)
    assert (row["status"], row["sensitivity"]) == ends, "the operation must have replayed"


# -- Loosening without a withdrawal ---------------------------------------------------


def test_a_hand_authored_upsert_lowering_an_earlier_upserts_sensitivity_is_reported_as_lowers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=ROOT_REVISION_ID,
            status="approved",
            sensitivity="internal",
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": ITEM_ID,
            "field": "sensitivity",
            "before": "confidential",
            "after": "internal",
            "undoes": ROOT_MIGRATION_ID,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


def test_an_upsert_omitting_sensitivity_lowers_what_the_create_item_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader defaults an omitted sensitivity to ``internal``, so omission is a lowering.

    The item is created in the root migration and upserted in a later one, so the
    before-state exists whichever moment the compare reads it.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=None,
            status="approved",
            sensitivity=None,
            item_id=CREATE_ONLY_ITEM_ID,
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": CREATE_ONLY_ITEM_ID,
            "field": "sensitivity",
            "before": "confidential",
            "after": "internal",
            "undoes": ROOT_MIGRATION_ID,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


# -- ``kind`` is decided by the effect of the last writer, not by its operation type -------
#
# A writer is a withdrawal iff its write tightened the field: status moved from
# surfaceable to non-surfaceable (``may_surface(..., include_unapproved=True)``), or
# sensitivity raised along ``DISCLOSURE_ORDER``. A write that leaves the field
# unchanged does not replace the recorded writer, and neither ``createItem`` nor any
# write in the migration that created the item counts.


def _accepted_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[LabelledProject, dict[str, Any]]:
    """An update accepted and applied on an ``internal`` item, before any withdrawal lands."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted
    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the withdrawal must sort first"
    return project, drafted


def test_a_declassification_is_not_a_withdrawal_so_a_further_lowering_is_reported_as_lowers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``changeSensitivity`` to ``internal`` lowers ``confidential``, so it is no withdrawal.

    By operation type the row said ``undoes``, naming a reviewed tightening that
    never happened; ``lowers`` is the only kind that does not mislead the reader.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project, _EARLIER_MIGRATION, reclassification(_EARLIER_MIGRATION, ITEM_ID, "internal")
    )
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=ROOT_REVISION_ID,
            status="approved",
            sensitivity="public",
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": ITEM_ID,
            "field": "sensitivity",
            "before": "internal",
            "after": "public",
            "undoes": _EARLIER_MIGRATION,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


#: (status, sensitivity) the in-place upsert states; the field, before and after the race row.
_IN_PLACE_WITHDRAWALS: Final = [
    ("rejected", "internal", "status", "rejected", "approved"),
    ("superseded", "internal", "status", "superseded", "approved"),
    ("deprecated", "internal", "status", "deprecated", "approved"),
    ("approved", "confidential", "sensitivity", "confidential", "internal"),
]


@pytest.mark.parametrize(
    "case",
    _IN_PLACE_WITHDRAWALS,
    ids=["rejected", "superseded", "deprecated-by-upsert", "sensitivity-raised-by-upsert"],
)
def test_an_in_place_withdrawal_by_upsert_is_what_an_accepted_update_is_reported_undoing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, str, str, str]
) -> None:
    """ADR-0024 decision 5's in-place withdrawal is an ``upsertRevision``, not a ``deprecateItem``.

    It re-declares the current revision with a tighter label (``docs/protocol/migrations.md``).
    Landing before an accepted update, it is the same race the operation-typed
    writers were reported for; calling it ``lowers`` hid the one reviewed
    withdrawal this report exists to surface.
    """
    status, sensitivity, field, before, after = case
    project, drafted = _accepted_update(tmp_path, monkeypatch)
    _write_migration(
        project,
        SORTS_BEFORE_A_DRAFT,
        _upsert_migration(
            SORTS_BEFORE_A_DRAFT,
            ROOT_REVISION_ID,
            expected_revision=None,
            status=status,
            sensitivity=sensitivity,
        ),
    )
    expected = [
        {
            "migrationId": drafted["migrationId"],
            "itemId": ITEM_ID,
            "field": field,
            "before": before,
            "after": after,
            "undoes": SORTS_BEFORE_A_DRAFT,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


def test_a_restated_status_does_not_replace_the_withdrawal_that_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upsert that leaves ``deprecated`` as it found it wrote nothing to the field.

    Were it recorded as the writer the row would name a restatement as what was
    undone, and ``kind`` would read ``lowers`` for a reviewed deprecation.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    _write_migration(project, _EARLIER_MIGRATION, deprecation(_EARLIER_MIGRATION, ITEM_ID))
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=ROOT_REVISION_ID,
            status="deprecated",
            sensitivity="internal",
        ),
    )
    _write_migration(
        project,
        _LATEST_MIGRATION,
        _upsert_migration(
            _LATEST_MIGRATION,
            _LATEST_REVISION,
            expected_revision=_HAND_REVISION,
            status="approved",
            sensitivity="internal",
        ),
    )
    expected = [
        {
            "migrationId": _LATEST_MIGRATION,
            "itemId": ITEM_ID,
            "field": "status",
            "before": "deprecated",
            "after": "approved",
            "undoes": _EARLIER_MIGRATION,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected
    assert item_row(project.root, ITEM_ID)["status"] == "approved"


_DEPRECATE: Final = "  - op: deprecateItem\n    itemId: {item}\n    reason: retired after review\n"


def _in_place(status: str) -> str:
    return _upsert_operation(
        ITEM_ID, ROOT_REVISION_ID, expected_revision=None, status=status, sensitivity="internal"
    )


#: (first withdrawal, second write between retired statuses, the status the update finds,
#: whether both are one migration). The second write changes the value, not surfaceability.
_RETIRED_BETWEEN_RETIRED: Final = [
    (_DEPRECATE.format(item=ITEM_ID), _in_place("rejected"), "rejected", False),
    (_DEPRECATE.format(item=ITEM_ID), _in_place("superseded"), "superseded", False),
    (_in_place("rejected"), _DEPRECATE.format(item=ITEM_ID), "deprecated", False),
    (_DEPRECATE.format(item=ITEM_ID), _in_place("superseded"), "superseded", True),
]


@pytest.mark.parametrize(
    "case",
    _RETIRED_BETWEEN_RETIRED,
    ids=[
        "deprecate-then-reject",
        "deprecate-then-supersede",
        "reject-then-deprecate",
        "deprecate-then-supersede-in-one-migration",
    ],
)
def test_a_retired_to_retired_write_does_not_displace_the_withdrawal_an_update_is_reported_undoing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, str, str, bool],
) -> None:
    """A write between two retired statuses leaves the item exactly as withheld as it was.

    The report and the accept floors judge a withdrawal by surfaceability. Recording
    the second write as the field's writer filed the readmitting update ``lowers``
    and named a restatement where the first write is the reviewed withdrawal.
    ``reject-then-deprecate`` read ``undoes`` under the operation-typed rule that preceded
    the effect rule, so it is the regression face.
    """
    first, second, found, one_migration = case
    project, drafted = _accepted_update(tmp_path, monkeypatch)
    first_id, second_id = SORTS_BEFORE_A_DRAFT, _HAND_MIGRATION
    assert second_id < drafted["migrationId"], "the second write must sort before the draft"
    if one_migration:
        _write_migration(project, first_id, _migration(first_id, first, second))
    else:
        _write_migration(project, first_id, _migration(first_id, first))
        _write_migration(project, second_id, _migration(second_id, second))
    expected = [
        {
            "migrationId": drafted["migrationId"],
            "itemId": ITEM_ID,
            "field": "status",
            "before": found,
            "after": "approved",
            "undoes": first_id,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


def test_a_create_item_is_never_a_withdrawal_so_a_later_lowering_is_reported_as_lowers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``createItem`` stating ``confidential`` is an initial labelling, not a tightening."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _upsert_migration(
            _HAND_MIGRATION,
            _HAND_REVISION,
            expected_revision=None,
            status="approved",
            sensitivity="internal",
            item_id=CREATE_ONLY_ITEM_ID,
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": CREATE_ONLY_ITEM_ID,
            "field": "sensitivity",
            "before": "confidential",
            "after": "internal",
            "undoes": ROOT_MIGRATION_ID,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


# -- Compared against the start of the migration (f4) --------------------------------------


def test_a_new_item_labelled_in_the_migration_that_creates_it_reports_no_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An initial labelling is not a lowering, and ``propose`` writes it for every new item.

    The createItem omits sensitivity (loader default ``internal``) and the same
    migration's upsert states ``public``. The dogfood corpus carried 26 such rows
    before f4.
    """
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            f"""  - op: createItem
    itemId: {ITEM_ID}
    kind: architecture
    namespace: backend
    owner: platform-team
""",
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=None,
                status="approved",
                sensitivity="public",
            ),
        ),
    )

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    assert item_row(project.root, ITEM_ID)["sensitivity"] == "public", (
        "the upsert must have replayed"
    )


def _corpus_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LabelledProject:
    """A registered project whose ``.theurian`` holds a copy of this checkout's tracked corpus.

    Key: ``git ls-files -- .theurian/migrations .theurian/knowledge
    .theurian/specifications`` in the checkout; the clone's own tree is only read.
    """
    repo = Path(__file__).resolve().parents[4]
    listed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no caller input
        [  # noqa: S607 -- git from PATH
            "git",
            "-C",
            str(repo),
            "ls-files",
            "-z",
            "--",
            ".theurian/migrations",
            ".theurian/knowledge",
            ".theurian/specifications",
        ],
        capture_output=True,
        check=True,
    )
    paths = [p for p in listed.stdout.decode("utf-8", "surrogateescape").split("\0") if p]
    assert any(p.startswith(".theurian/migrations/") for p in paths), "the corpus must be tracked"
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    for relative in paths:
        target = project.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    return project


def test_the_dogfood_corpus_reports_no_permissive_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repository's own history labels new items in the migration that creates them.

    26 rows were measured here before f4 (2026-10-02, ``migrate validate --json``);
    every one was an initial labelling. A report that fires on the maintainers' own
    corpus is noise nobody reads, so it must stay empty on it.
    """
    _corpus_project(tmp_path, monkeypatch)

    validated = cli_ok("migrate", "validate")

    assert _rows(validated) == []


def _withdraw_then_reassert(
    project: LabelledProject, withdrawal: str, status: str, label: str
) -> None:
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            withdrawal,
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status=status,
                sensitivity=label,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("labels", "withdrawal", "ends"),
    [
        (
            "internal",
            "  - op: deprecateItem\n    itemId: {item}\n    reason: retired after review\n",
            ("approved", "internal"),
        ),
        (
            "internal",
            "  - op: changeSensitivity\n    itemId: {item}\n    sensitivity: confidential\n"
            "    reason: reclassified after review\n",
            ("approved", "internal"),
        ),
    ],
    ids=["deprecate-then-approve", "confidential-then-internal"],
)
def test_a_migration_that_withdraws_and_reasserts_an_item_in_one_diff_is_not_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    labels: str,
    withdrawal: str,
    ends: tuple[str, str],
) -> None:
    """One migration is one reviewed diff: the reviewer sees both operations, so a migration
    that withdraws and re-asserts an item in the same diff is not reported.

    A recorded design limit of f4, not a gap: against the item's state at the start
    of the migration, its state at the end is no looser.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=labels)
    _withdraw_then_reassert(project, withdrawal.format(item=ITEM_ID), "approved", "internal")

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    row = item_row(project.root, ITEM_ID)
    assert (row["status"], row["sensitivity"]) == ends, "both operations must have replayed"


def _restore_then_upsert(project: LabelledProject) -> None:
    _write_migration(project, _EARLIER_MIGRATION, deprecation(_EARLIER_MIGRATION, ITEM_ID))
    _write_migration(
        project,
        _RESTORE_MIGRATION,
        _migration(
            _RESTORE_MIGRATION,
            f"  - op: restoreItem\n    itemId: {ITEM_ID}\n",
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="internal",
            ),
        ),
    )


def test_a_restore_and_content_update_in_one_migration_reports_no_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape READMIT_REMEDY's reader writes, adding the content update beside the restore.

    The upsert restates ``approved`` over a status ``restoreItem`` had already set
    in the same migration, so it loosened nothing; reporting it as undoing the
    earlier deprecation cries wolf on the remedy the product itself prints.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    _restore_then_upsert(project)

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    assert item_row(project.root, ITEM_ID)["status"] == "approved"


def test_an_upsert_raising_the_label_is_not_reported_after_the_migration_made_it_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upsert states ``internal`` over ``public``, a raise; the loosening was the
    sanctioned ``changeSensitivity`` beside it, so the upsert is not the move to report.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            "  - op: changeSensitivity\n    itemId: "
            + ITEM_ID
            + "\n    sensitivity: public\n    reason: reclassified after review\n",
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="internal",
            ),
        ),
    )

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    assert item_row(project.root, ITEM_ID)["sensitivity"] == "internal"


def test_an_upsert_lowering_below_the_start_is_reported_even_after_a_raise_in_its_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declassification made by a content update rather than by ``changeSensitivity``.

    The upsert itself lowered the label (``restricted`` to ``internal``) and the
    migration left the item below the ``confidential`` class it found, so the
    one-diff design limit does not cover it. The earlier writer is the root
    migration's ``createItem``/upsert, so ``kind`` is ``lowers``.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            "  - op: changeSensitivity\n    itemId: "
            + ITEM_ID
            + "\n    sensitivity: restricted\n    reason: reclassified after review\n",
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="internal",
            ),
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": ITEM_ID,
            "field": "sensitivity",
            "before": "confidential",
            "after": "internal",
            "undoes": ROOT_MIGRATION_ID,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


def test_an_upsert_lowering_that_a_later_operation_restores_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The net effect of one reviewed diff is nil, so it is not reported, which is the
    same reason as the withdraw-and-reassert design limit.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="internal",
            ),
            "  - op: changeSensitivity\n    itemId: "
            + ITEM_ID
            + "\n    sensitivity: confidential\n    reason: reclassified after review\n",
        ),
    )

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    assert item_row(project.root, ITEM_ID)["sensitivity"] == "confidential"


def test_a_reported_after_is_the_value_the_migration_ends_on_not_the_one_the_upsert_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row's ``after`` is the migration's end label; reporting what the upsert
    wrote would name ``public`` for an item that ends ``internal``.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="public",
            ),
            "  - op: changeSensitivity\n    itemId: "
            + ITEM_ID
            + "\n    sensitivity: internal\n    reason: reclassified after review\n",
        ),
    )
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": ITEM_ID,
            "field": "sensitivity",
            "before": "confidential",
            "after": "internal",
            "undoes": ROOT_MIGRATION_ID,
            "kind": "lowers",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected
    assert item_row(project.root, ITEM_ID)["sensitivity"] == "internal"


_DEPRECATE_OP: Final = f"  - op: deprecateItem\n    itemId: {ITEM_ID}\n    reason: retired\n"
_RESTORE_OP: Final = f"  - op: restoreItem\n    itemId: {ITEM_ID}\n"


def _readmit_then(project: LabelledProject, *later: str) -> None:
    _write_migration(project, _EARLIER_MIGRATION, deprecation(_EARLIER_MIGRATION, ITEM_ID))
    _write_migration(
        project,
        _HAND_MIGRATION,
        _migration(
            _HAND_MIGRATION,
            _upsert_operation(
                ITEM_ID,
                _HAND_REVISION,
                expected_revision=ROOT_REVISION_ID,
                status="approved",
                sensitivity="internal",
            ),
            *later,
        ),
    )


def test_a_deprecation_later_in_the_migration_decides_the_end_state_so_nothing_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upsert readmits, but the migration ends with the item deprecated: no looser."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    _readmit_then(project, _DEPRECATE_OP)

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == []
    assert _rows(applied) == []
    assert item_row(project.root, ITEM_ID)["status"] == "deprecated"


def test_a_restore_after_a_later_deprecation_in_the_migration_leaves_the_readmission_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the test above: only the end state differs, and the row appears."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    _readmit_then(project, _DEPRECATE_OP, _RESTORE_OP)
    expected = [
        {
            "migrationId": _HAND_MIGRATION,
            "itemId": ITEM_ID,
            "field": "status",
            "before": "deprecated",
            "after": "approved",
            "undoes": _EARLIER_MIGRATION,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected
    assert item_row(project.root, ITEM_ID)["status"] == "approved"


def test_the_race_row_survives_a_later_migration_that_writes_no_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last label-writing migration is settled by the read of ``moves``, not by a
    successor's label write; a trailing migration writing none must not drop its rows.
    """
    project, drafted, withdrawal = _race(tmp_path, monkeypatch, "status")
    trailing = "7ZZZZZZZZZ01234567890ABCDE"
    assert drafted["migrationId"] < trailing
    _write_migration(
        project,
        trailing,
        _migration(
            trailing,
            f"  - op: addAlias\n    alias: architecture.auth-policy-old\n    itemId: {ITEM_ID}\n",
        ),
    )
    expected = [
        {
            "migrationId": drafted["migrationId"],
            "itemId": ITEM_ID,
            "field": "status",
            "before": "deprecated",
            "after": "approved",
            "undoes": withdrawal,
            "kind": "undoes",
        }
    ]

    validated, applied = _reports()

    assert applied is not None
    assert _rows(validated) == expected
    assert _rows(applied) == expected


# -- A set that does not replay ---------------------------------------------------------

_REVISION_ONE: Final = "01K1AAAREV01234567890ABCDF"
_REVISION_ZERO: Final = "01K1AAAREV01234567890ABCDE"


def _a_set_that_validates_but_does_not_replay(project: LabelledProject) -> None:
    """The set ``test_draft_status_refusal.py`` builds: valid statically, a conflict at replay."""
    (project.root / ".theurian/knowledge/architecture/auth-policy.md").write_text(BODY)
    _write_migration(
        project,
        SORTS_BEFORE_A_DRAFT,
        f"""apiVersion: theurian.dev/v1
id: {SORTS_BEFORE_A_DRAFT}
createdAt: 2026-08-02T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {ITEM_ID}
    kind: architecture
    namespace: backend
    owner: platform-team
  - op: upsertRevision
    itemId: {ITEM_ID}
    revisionId: {_REVISION_ONE}
    expectedRevision: {_REVISION_ZERO}
    contentFile: ../knowledge/architecture/auth-policy.md
    contentSha256: {body_pin(BODY)}
    metadata:
      title: Authentication policy
      contentType: text/markdown
      kind: architecture
      namespace: backend
      status: approved
      owner: platform-team
      sourceAnchors:
        - provider: git
          sourceUri: git://demo/auth-policy.md
""",
    )


def test_validate_on_a_set_that_does_not_replay_stays_valid_and_says_the_report_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict and exit code must not change; the report is withheld in the engine's words."""
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    _a_set_that_validates_but_does_not_replay(project)

    code, payload = cli("migrate", "validate")

    assert code == 0, payload
    assert payload["valid"] is True
    assert "permissiveMoves" in payload, sorted(payload)
    assert payload["permissiveMoves"] is None
    assert "Revision conflict on" in str(payload.get("permissiveMovesUnavailable")), payload


def test_the_text_validate_on_a_set_that_does_not_replay_prints_the_report_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text mode has its own spelling of the withheld report; JSON mode's pin cannot reach it."""
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    _a_set_that_validates_but_does_not_replay(project)

    lines = _text("migrate", "validate").splitlines()

    assert any("unavailable" in line and "Revision conflict on" in line for line in lines), lines


def test_a_successful_validate_replay_carries_no_unavailable_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labelled_project(tmp_path, monkeypatch)

    payload = cli_ok("migrate", "validate")

    assert payload.get("permissiveMoves") == []
    assert "permissiveMovesUnavailable" not in payload


# -- Apply reports its own run --------------------------------------------------------


def test_a_second_apply_with_nothing_pending_reports_none_while_validate_still_reports_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply reports the upserts it applied in THIS run; validate reports the whole set."""
    _race(tmp_path, monkeypatch, "status")
    first = cli_ok("migrate", "apply")
    assert len(_rows(first)) == 1, first

    second = cli_ok("migrate", "apply")
    validated = cli_ok("migrate", "validate")

    assert _rows(second) == []
    assert len(_rows(validated)) == 1, validated


# -- Who does not print it ---------------------------------------------------------------


def test_propose_accept_does_not_print_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is the operator's replay, not an accept path.

    The accept output's next-steps prose names ``permissiveMoves`` once, to point
    at ``migrate validate``; the key it must not carry is the report itself.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", ROOT_REVISION_ID)
    assert code == 0, drafted

    accepted = cli_ok("propose", "accept", drafted["proposalId"])

    assert "permissiveMoves" not in accepted
    assert "permissiveMovesUnavailable" not in accepted


#: Every spelling of the report in code: the payload key, the module, the row type,
#: the writer map and the renderer module.
_REPORT_SPELLINGS: Final = (
    "permissiveMoves",
    "permissive_moves",
    "PermissiveMove",
    "LabelWriters",
    "permissive_move_report",
)

_REPO: Final = Path(__file__).resolve().parents[4]
_SERVED_PYTHON: Final = [
    _REPO / "packages/theurian-core/src/theurian/mcp",
    _REPO / "packages/theurian-core/src/theurian/daemon",
]


def test_no_served_path_mentions_the_report() -> None:
    """Guard, not a RED-first test: it passed before the report existed and must keep passing.

    A served path publishing the report would hand a caller the history of
    withdrawn labels. Key: any file under ``src/theurian/mcp/``,
    ``src/theurian/daemon/`` or ``schemas/mcp/``, read as text, holding any of
    :data:`_REPORT_SPELLINGS` (case-sensitive substrings). A served path
    importing the module would read the same history without printing the
    payload key; :func:`test_no_served_module_references_the_report_machinery`
    holds that by name.
    """
    roots = [*_SERVED_PYTHON, _REPO / "schemas/mcp"]
    scanned = [p for root in roots for p in root.rglob("*") if p.is_file()]
    assert len(scanned) > 10, "the scan must reach the served surface"

    offenders = [
        (str(p), spelling)
        for p in scanned
        for spelling in _REPORT_SPELLINGS
        if spelling in p.read_text(encoding="utf-8", errors="replace")
    ]

    assert offenders == []


_REPORT_NAMES: Final = frozenset(
    {
        "apply_migration_set",
        "permissive_moves_in",
        "ApplyReport",
        "LabelWriters",
        "PermissiveMove",
        "rehearse_migration_set",
        "Replay",
        "MigrationSetRehearsal",
    }
)


def _referenced_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                names.add(alias.name.rsplit(".", 1)[-1])
                if alias.asname:
                    names.add(alias.asname)
    return names


def test_a_referenced_names_walk_sees_every_way_a_module_can_name_the_machinery() -> None:
    """Positive controls: each spelling of a reference is found, and an unrelated module is not."""
    spellings = [
        "from theurian.cli.migration_pipeline import apply_migration_set",
        "import theurian.application.permissive_moves as moves\nmoves.LabelWriters",
        "x = pipeline.ApplyReport",
        "def f(a: PermissiveMove) -> None: ...",
        "from a import b as permissive_moves_in",
        "from theurian.cli.migration_pipeline import rehearse_migration_set",
        "from theurian.cli.migration_pipeline import Replay",
        "def f(r: MigrationSetRehearsal) -> None: ...",
        "rows = pipeline.Replay(items, moves, order).moves",
    ]

    assert all(_referenced_names(text) & _REPORT_NAMES for text in spellings)
    assert not _referenced_names("from theurian.cli.context import build\nbuild()") & _REPORT_NAMES


def test_no_served_module_references_the_report_machinery() -> None:
    """A transitive import through ``migration_engine`` is allowed; naming these is not.

    Key: every ``.py`` under ``src/theurian/mcp/`` and ``src/theurian/daemon/``,
    parsed, with each ``Name``, ``Attribute`` and import-alias name compared to
    :data:`_REPORT_NAMES`. A call through a getattr string or an alias of another
    name is invisible here.
    """
    modules = [p for root in _SERVED_PYTHON for p in root.rglob("*.py")]
    assert len(modules) > 5, "the walk must reach the served modules"

    offenders = [
        (str(p.relative_to(_REPO)), sorted(_referenced_names(p.read_text("utf-8")) & _REPORT_NAMES))
        for p in modules
        if _referenced_names(p.read_text("utf-8")) & _REPORT_NAMES
    ]

    assert offenders == []


# -- Validate's unavailable path ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "words"),
    [
        (OSError(errno.ENOSPC, "No space left on device"), "No space left on device"),
        (sqlite3.OperationalError("database or disk is full"), "database or disk is full"),
    ],
    ids=["oserror", "sqlite-error"],
)
def test_validate_stays_valid_when_the_scratch_replay_hits_an_os_or_sqlite_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raised: Exception, words: str
) -> None:
    """The verdict does not rest on the report replay, so its faults withhold the report only.

    ``permissive_moves_in`` catches ``TheurianError``, ``OSError`` and
    ``sqlite3.Error``; dropping either of the last two turns a full disk into a
    traceback and a failed ``migrate validate`` for a set that is valid.
    """
    labelled_project(tmp_path, monkeypatch)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise raised

    monkeypatch.setattr("theurian.cli.migration_pipeline.create_database", refuse)

    code, payload = cli("migrate", "validate")

    assert code == 0, payload
    assert payload["valid"] is True
    assert payload["permissiveMoves"] is None
    assert words in str(payload.get("permissiveMovesUnavailable")), payload
