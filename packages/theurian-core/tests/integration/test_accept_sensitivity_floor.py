"""``propose accept`` refuses a proposal that would lower an existing item's sensitivity.

The draft-time refusal in ``test_update_label_inheritance.py`` only covers a draft
written by a build that has it. ``accept`` lands whatever the proposal directory
says, and two kinds of proposal reach it past that check: one drafted by an
earlier build, whose ``upsertRevision`` omits ``sensitivity`` and so loads as
``internal``; and one that named the item's then-current sensitivity before a
hand-authored ``changeSensitivity`` raised it, where replay order decides which
of the two migrations lands last.

So the floor is checked on the union replay, per existing item, against the
sensitivity the item currently holds. A reclassification is still available: it
is the ``changeSensitivity`` migration, which is not a content update.

The operator's own ``accept`` runs on the unscoped store, so its refusal may name
both labels; what it must name is the way to declassify.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import pytest
from label_inheritance_support import (
    EVIDENCE,
    SORTS_AFTER_A_DRAFT,
    SORTS_BEFORE_A_DRAFT,
    UPDATED_BODY,
    LabelledProject,
    cli,
    cli_ok,
    item_row,
    labelled_project,
    land_reclassification,
)

pytestmark = pytest.mark.integration

_SENSITIVITY_LINE: Final = re.compile(r"^(?P<indent> +)sensitivity: \w+\n", re.MULTILINE)


def _draft_update(project: LabelledProject, *extra: str) -> dict[str, Any]:
    (project.root / "body.md").write_text(UPDATED_BODY, encoding="utf-8")
    payload = cli_ok(
        "propose",
        "--item-id",
        project.item_id,
        "--title",
        "Authentication policy",
        "--kind",
        "architecture",
        "--owner",
        "platform-team",
        "--author",
        "platform-team@example.com",
        "--description",
        "Tighten the token lifetime.",
        "--body-file",
        str(project.root / "body.md"),
        "--source-uri",
        "git://demo/auth-policy.md",
        "--agent-id",
        "claude-code",
        "--task-id",
        "task-7",
        "--model",
        "claude-opus-5",
        "--reasoning",
        EVIDENCE["reasoning"],
        "--expected-revision",
        project.revision_id,
        *extra,
    )
    assert payload["expectedRevision"] == project.revision_id, "this must be an update"
    return payload


def _migration_file(project: LabelledProject, drafted: dict[str, Any]) -> Path:
    return Path(project.root / drafted["proposalDirectory"] / drafted["migrationFile"])


def _rewrite(project: LabelledProject, drafted: dict[str, Any], *, sensitivity: str | None) -> None:
    """Edit the staged migration as an earlier build or a hand edit would have left it.

    Removes the ``sensitivity`` line (a draft that never wrote one has nothing to
    remove) or replaces its value, keeping the line's own indentation. Text, not a
    YAML round trip: every other value must reach ``accept`` byte for byte.
    """
    path = _migration_file(project, drafted)
    text = path.read_text(encoding="utf-8")
    assert sensitivity is None or _SENSITIVITY_LINE.search(text), "no sensitivity line to replace"
    edited = _SENSITIVITY_LINE.sub(
        lambda match: (
            "" if sensitivity is None else f"{match['indent']}sensitivity: {sensitivity}\n"
        ),
        text,
    )
    path.write_text(edited, encoding="utf-8")


def _landing_zone(project: LabelledProject) -> dict[str, bytes]:
    """Every file ``accept`` could move into or out of: proposals, migrations and knowledge."""
    found: dict[str, bytes] = {}
    for name in ("proposals", "proposals-local", "migrations", "knowledge"):
        base = project.root / ".theurian" / name
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    found[path.relative_to(project.root).as_posix()] = path.read_bytes()
    return found


def _assert_refused_and_nothing_moved(
    project: LabelledProject, drafted: dict[str, Any], before: dict[str, bytes]
) -> None:
    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload  # the exit `propose accept --help` documents for a floor refusal
    assert "changeSensitivity" in str(payload.get("remedy", "")), payload
    assert _landing_zone(project) == before, "a refused accept moved files"


# -- A proposal drafted before the fix, or edited by hand -----------------------


def test_accept_refuses_a_proposal_that_omits_sensitivity_for_a_confidential_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape a pre-fix draft has: the key is absent, so the loader says ``internal``.

    Such a proposal may already sit in a proposals directory, committed and under
    review. Accepting it would reclassify a confidential item with no review of
    the reclassification.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    drafted = _draft_update(project)
    _rewrite(project, drafted, sensitivity=None)
    assert "sensitivity:" not in _migration_file(project, drafted).read_text(encoding="utf-8")

    _assert_refused_and_nothing_moved(project, drafted, _landing_zone(project))


def test_accept_refuses_a_proposal_naming_a_lower_sensitivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand edit can name what the draft-time refusal would have refused."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    drafted = _draft_update(project, "--sensitivity", "confidential")
    _rewrite(project, drafted, sensitivity="internal")
    assert "sensitivity: internal" in _migration_file(project, drafted).read_text(encoding="utf-8")

    _assert_refused_and_nothing_moved(project, drafted, _landing_zone(project))


def test_accept_refuses_a_proposal_lowering_restricted_to_confidential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two highest levels are ordered, so the floor holds one step below ``restricted``."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="restricted")
    drafted = _draft_update(project, "--sensitivity", "restricted")
    _rewrite(project, drafted, sensitivity="confidential")

    _assert_refused_and_nothing_moved(project, drafted, _landing_zone(project))


def test_a_refused_accept_leaves_the_item_confidential_after_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing the refused proposal carried reaches the store."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    drafted = _draft_update(project)
    _rewrite(project, drafted, sensitivity=None)

    code, payload = cli("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    assert code != 0, payload
    row = item_row(project.root, project.item_id)
    assert (row["sensitivity"], row["current_revision_id"]) == ("confidential", project.revision_id)


@pytest.mark.parametrize(
    ("current", "named"),
    [
        pytest.param("confidential", "confidential", id="label-kept"),
        pytest.param("internal", "confidential", id="label-raised"),
        pytest.param("public", "internal", id="public-raised-to-internal"),
    ],
)
def test_accept_lands_an_update_that_keeps_or_raises_the_sensitivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current: str, named: str
) -> None:
    """The floor is on lowering: without these a refusal of every update would pass above."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)
    drafted = _draft_update(project, "--sensitivity", named)

    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    row = item_row(project.root, project.item_id)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["sensitivity"] == named


# -- The ordering race ------------------------------------------------------------


def _race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reclassification_id: str
) -> tuple[LabelledProject, dict[str, Any]]:
    """An update naming the current ``internal``, then a reclassification to ``confidential``.

    The draft names the label the item holds when it is drafted, which is correct
    then. The reclassification lands before ``accept``, so what ``accept`` replays
    is the draft and the reclassification in migration-id order.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    drafted = _draft_update(project, "--sensitivity", "internal")
    land_reclassification(project.root, reclassification_id, project.item_id, "confidential")
    assert item_row(project.root, project.item_id)["sensitivity"] == "confidential"
    return project, drafted


def test_accept_refuses_a_draft_whose_migration_replays_after_a_reclassification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reclassification sorts first, so the draft's ``internal`` would replay over it."""
    project, drafted = _race(tmp_path, monkeypatch, SORTS_BEFORE_A_DRAFT)
    assert drafted["migrationId"] > SORTS_BEFORE_A_DRAFT, "the reclassification must sort first"

    _assert_refused_and_nothing_moved(project, drafted, _landing_zone(project))


def test_accept_lands_a_draft_whose_migration_replays_before_a_reclassification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft sorts first, so the reclassification replays last: the item stays confidential."""
    project, drafted = _race(tmp_path, monkeypatch, SORTS_AFTER_A_DRAFT)
    assert drafted["migrationId"] < SORTS_AFTER_A_DRAFT, "the draft must sort first"

    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    row = item_row(project.root, project.item_id)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["sensitivity"] == "confidential"


# -- A proposal that carries its own changeSensitivity -----------------------------


def _add_change_sensitivity(project: LabelledProject, drafted: dict[str, Any], to: str) -> None:
    """Hand-edit the staged migration to end with a ``changeSensitivity`` operation."""
    path = _migration_file(project, drafted)
    path.write_text(
        path.read_text(encoding="utf-8") + f"- op: changeSensitivity\n  itemId: {project.item_id}\n"
        f"  sensitivity: {to}\n  reason: reclassified in the proposal\n",
        encoding="utf-8",
    )


def test_accept_refuses_a_proposal_whose_own_change_sensitivity_lowers_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor reads the post-replay state, whichever operation moved it.

    The draft keeps ``confidential`` and would pass a check on the upsert alone;
    its trailing ``changeSensitivity`` is what lowers the item, unreviewed as a
    reclassification.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    drafted = _draft_update(project)
    _add_change_sensitivity(project, drafted, "internal")

    _assert_refused_and_nothing_moved(project, drafted, _landing_zone(project))


def test_accept_lands_a_proposal_whose_own_change_sensitivity_raises_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this a refusal of every ``changeSensitivity`` would pass the test above."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="internal")
    drafted = _draft_update(project)
    _add_change_sensitivity(project, drafted, "confidential")

    cli_ok("propose", "accept", drafted["proposalId"])
    cli_ok("migrate", "apply")

    row = item_row(project.root, project.item_id)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["sensitivity"] == "confidential"
