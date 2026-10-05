"""``theurian propose`` and ``okf import`` refuse to draft for a retired item (GHSA-v2qg-23fc-7fqp).

``current_item_in`` is the one lookup that returns a retired item: the MCP and
candidate closures answer ``None`` for it, so a withheld item stays
indistinguishable from an absent id there and the refusal belongs to ``accept``.
Over the CLI and the OKF import the operator's own view is unscoped, and drafting
an update the accept-side floor will always refuse is wasted review.

The refusal text is a constant: it names no status, so it can be shared by every
surface that reaches it.

The same module holds the remedy of the two refusals ``current_item_in`` raises
when the landed set does not replay. ``theurian migrate validate``'s verdict does
not rest on a replay (it replays only to build its report), so it reports valid on
exactly such a set and cannot be what the remedy sends the reader to.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import pytest
from label_inheritance_support import (
    BODY,
    ITEM_ID,
    SORTS_BEFORE_A_DRAFT,
    LabelledProject,
    cli,
    cli_propose,
    create_only_project,
    labelled_project,
    land_deprecation,
    proposals_tree,
)
from migration_fixtures import body_pin

from theurian.application import proposal_service
from theurian.cli import migration_pipeline

pytestmark = pytest.mark.integration

_STATUS_WORD: Final = re.compile(
    r"deprecated|superseded|rejected|approved|draft|proposed", re.IGNORECASE
)
_RETIRED: Final = ["deprecated", "superseded", "rejected"]


def _deprecate(project: LabelledProject) -> None:
    land_deprecation(project.root, SORTS_BEFORE_A_DRAFT, project.item_id)


def _with_a_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> LabelledProject:
    """A project whose one item, with a revision, ends in ``status``."""
    if status == "deprecated":
        project = labelled_project(tmp_path, monkeypatch)
        _deprecate(project)
    else:
        project = labelled_project(tmp_path, monkeypatch, status=status)
    return project


def _propose_update(project: LabelledProject) -> tuple[int, dict[str, object]]:
    return cli_propose(project, project.item_id, "--expected-revision", project.revision_id)


# -- A retired item is not drafted for --------------------------------------------


@pytest.mark.parametrize(
    ("status", "shape"),
    [
        pytest.param("deprecated", "with-revision", id="deprecated-with-revision"),
        pytest.param("superseded", "with-revision", id="superseded-with-revision"),
        pytest.param("rejected", "with-revision", id="rejected-with-revision"),
        pytest.param("deprecated", "create-only", id="deprecated-create-only"),
    ],
)
def test_propose_refuses_to_draft_for_a_retired_item_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, shape: str
) -> None:
    """The update would be refused at ``accept``; a ``rejected`` item may hold a secret."""
    if shape == "create-only":
        assert status == "deprecated", "a create-only item has no revision to carry another status"
        project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
        _deprecate(project)
        extra: tuple[str, ...] = ()
    else:
        project = _with_a_revision(tmp_path, monkeypatch, status)
        extra = ("--expected-revision", project.revision_id)
    proposals_before = proposals_tree(project.root)
    gitignore_before = (project.root / ".gitignore").read_bytes()

    code, payload = cli_propose(project, project.item_id, *extra)

    assert code != 0, payload
    assert "restoreItem" in str(payload.get("remedy", "")), payload
    assert proposals_tree(project.root) == proposals_before, "a refused draft wrote a proposal"
    assert (project.root / ".gitignore").read_bytes() == gitignore_before


def test_the_draft_refusal_is_one_constant_that_names_no_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A message naming the status would say which of the retired kinds an item is."""
    answers: list[tuple[str, dict[str, object]]] = []
    for status in _RETIRED:
        base = tmp_path / status
        base.mkdir()
        project = _with_a_revision(base, monkeypatch, status)
        code, payload = _propose_update(project)
        assert code != 0, payload
        answers.append((status, payload))

    first = answers[0][1]
    assert "restoreItem" in str(first.get("remedy", "")), first
    for status, payload in answers:
        assert payload == first, (status, payload)
        assert _STATUS_WORD.search(str(payload["error"])) is None, payload


def test_the_draft_refusal_says_the_services_constant_and_that_no_update_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI surfaces the one constant, so an emptied or reworded one cannot pass unseen.

    "No update is written" is what tells the author the refusal is final for this
    command and nothing is waiting in ``.theurian/proposals/``.
    """
    project = _with_a_revision(tmp_path, monkeypatch, "deprecated")

    code, payload = _propose_update(project)

    assert code == 2, payload
    assert proposal_service.RETIRED_ITEM_MESSAGE, "the refusal text must not be empty"
    assert "no update is written" in proposal_service.RETIRED_ITEM_MESSAGE
    assert payload["error"] == proposal_service.RETIRED_ITEM_MESSAGE, payload


def test_a_retired_item_naming_a_lower_sensitivity_gets_the_retired_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retired check runs first: a lowering refusal would send the author to the wrong cure."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    _deprecate(project)
    proposals_before = proposals_tree(project.root)

    code, payload = cli_propose(
        project,
        project.item_id,
        "--expected-revision",
        project.revision_id,
        "--sensitivity",
        "internal",
    )

    assert code == 2, payload
    assert payload["error"] == proposal_service.RETIRED_ITEM_MESSAGE, payload
    assert "restoreItem" in str(payload.get("remedy", "")), payload
    assert proposals_tree(project.root) == proposals_before


@pytest.mark.parametrize("status", ["approved", "draft", "proposed"])
def test_propose_still_drafts_for_a_surfaceable_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Without these a refusal of every update would satisfy the tests above."""
    project = labelled_project(tmp_path, monkeypatch, status=status)

    code, payload = _propose_update(project)

    assert code == 0, payload


def test_propose_still_drafts_the_first_revision_of_a_create_only_draft_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``createItem`` leaves status ``draft``, which is surfaceable."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")

    code, payload = cli_propose(project, project.item_id)

    assert code == 0, payload


# -- OKF import ---------------------------------------------------------------------

_BUNDLE_OPTIONS: Final = (
    "--owner",
    "platform-team",
    "--author",
    "dana@example.com",
    "--agent-id",
    "claude-code",
    "--task-id",
    "task-okf",
    "--model",
    "claude-opus-5",
    "--reasoning",
    "Importing a bundle a teammate shared.",
)


def _bundle(root: Path, item_id: str) -> Path:
    bundle = root.parent / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "concept.md").write_text(
        f"---\ntype: architecture\ntitle: Imported concept\nstatus: stable\n"
        f"theurian_export_version: 1\ntheurian_item_id: {item_id}\n"
        f"theurian_content_type: text/markdown\n---\n\nBody prose.\n",
        encoding="utf-8",
    )
    return bundle


def test_okf_import_refuses_a_concept_whose_item_is_retired_locally_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create-only deprecated item has no revision, so the missing ``expectedRevision`` passes.

    Only the status check stops it: nothing else refuses the draft.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    _deprecate(project)
    bundle = _bundle(project.root, project.item_id)
    before = proposals_tree(project.root)

    code, payload = cli("okf", "import", str(bundle), *_BUNDLE_OPTIONS)

    assert code == 0, payload
    assert payload["conceptsAdmitted"] == 0, payload
    assert [(r["kind"], r["key"]) for r in payload["refusals"]] == [("draft", project.item_id)]
    assert proposals_tree(project.root) == before, "a refused import wrote a proposal"


def test_okf_import_still_drafts_a_concept_for_a_create_only_draft_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the same bundle over an item that is not retired is admitted."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    bundle = _bundle(project.root, project.item_id)

    code, payload = cli("okf", "import", str(bundle), *_BUNDLE_OPTIONS)

    assert code == 0, payload
    assert payload["conceptsAdmitted"] == 1, payload
    assert payload["refusals"] == [], payload


def test_okf_import_over_a_retired_item_with_a_revision_is_refused_as_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import never names ``expectedRevision``, so the cheap revision check refuses first.

    The retired refusal must win over it: its remedy ("Pass --expected-revision")
    is a step the retired refusal then blocks. The refusal record carries only
    the exception's class name, so the test reads the raised refusal off
    ``ProposalService.draft``.
    """
    project = _with_a_revision(tmp_path, monkeypatch, "deprecated")
    bundle = _bundle(project.root, project.item_id)
    before = proposals_tree(project.root)
    raised: list[BaseException] = []
    real_draft = proposal_service.ProposalService.draft

    def spy(self: proposal_service.ProposalService, *args: object, **kwargs: object) -> object:
        try:
            return real_draft(self, *args, **kwargs)  # type: ignore[arg-type]
        except proposal_service.ProposalError as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(proposal_service.ProposalService, "draft", spy)

    code, payload = cli("okf", "import", str(bundle), *_BUNDLE_OPTIONS)

    assert code == 0, payload
    assert payload["conceptsAdmitted"] == 0, payload
    assert [(r["kind"], r["key"]) for r in payload["refusals"]] == [("draft", ITEM_ID)]
    assert [str(exc) for exc in raised] == [proposal_service.RETIRED_ITEM_MESSAGE]
    assert proposals_tree(project.root) == before


# -- A landed set that loads and validates but does not replay ------------------------

_REVISION_ONE: Final = "01K1AAAREV01234567890ABCDF"
_REVISION_ZERO: Final = "01K1AAAREV01234567890ABCDE"


def _a_set_that_validates_but_does_not_replay(project: LabelledProject) -> None:
    """A first revision claiming to replace a revision the store never held.

    Schema-valid and passing every whole-set guard, so ``migrate validate``
    reports it valid; the engine refuses it at replay with a revision conflict.
    """
    (project.root / ".theurian/knowledge/architecture/auth-policy.md").write_text(BODY)
    (project.root / f".theurian/migrations/{SORTS_BEFORE_A_DRAFT}-conflict.yaml").write_text(
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
"""
    )


def test_a_draft_over_a_landed_set_that_does_not_replay_carries_the_engines_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy must not send the reader to a command that reports this set valid.

    ``migrate validate`` loads and runs the static guards; its verdict does not
    rest on a replay, so it says ``valid`` here and the reader is left with nothing. The engine's
    own message names the item and the two revisions, and the directory the
    fault is in is the one place left to look.
    """
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    _a_set_that_validates_but_does_not_replay(project)
    validated_code, validated = cli("migrate", "validate")
    assert (validated_code, validated["valid"]) == (0, True), validated
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, ITEM_ID, "--expected-revision", _REVISION_ONE)

    assert code != 0, payload
    assert "does not replay" in str(payload.get("error", "")), payload
    assert proposals_tree(project.root) == before
    error, remedy = str(payload.get("error", "")), str(payload.get("remedy", ""))
    assert f"Revision conflict on {ITEM_ID}" in error, payload
    assert "migrate validate" not in remedy, payload
    assert ".theurian/migrations/" in remedy, payload
    assert "`theurian migrate apply` runs the same replay" in remedy, payload


def test_migrate_apply_refuses_the_set_that_migrate_validate_passes_in_the_same_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy says `migrate apply` "runs the same replay"; this is what holds that sentence up.

    If apply passed this set, or refused it in other words than the refusal the
    draft carries, the reader sent there would learn nothing.
    """
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    _a_set_that_validates_but_does_not_replay(project)
    validated_code, validated = cli("migrate", "validate")
    assert (validated_code, validated["valid"]) == (0, True), validated
    drafted_code, drafted = cli_propose(project, ITEM_ID, "--expected-revision", _REVISION_ONE)
    assert drafted_code != 0, drafted
    engine_words = f"Revision conflict on {ITEM_ID}: migration expected {_REVISION_ZERO}"
    assert engine_words in str(drafted.get("error", "")), drafted

    code, payload = cli("migrate", "apply")

    assert code != 0, payload
    assert engine_words in json.dumps(payload), payload


def test_a_replay_that_holds_no_such_item_refuses_and_names_migrate_apply_not_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second refusal ``current_item_in`` raises, which no real set reaches.

    A set that names the item and replays cleanly always holds it, so the replay
    reader is driven to ``{}``. ``migrate validate``'s verdict does not rest on a replay and
    would report this set valid; the remedy sends the reader to ``migrate apply``, which does.
    """
    project = labelled_project(tmp_path, monkeypatch)
    monkeypatch.setattr(migration_pipeline, "_read_items", lambda *_: {})
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, ITEM_ID, "--expected-revision", project.revision_id)

    assert code != 0, payload
    assert "the replay holds no such item" in str(payload.get("error", "")), payload
    remedy = str(payload.get("remedy", ""))
    assert "migrate validate" not in remedy, payload
    assert "theurian migrate apply" in remedy, payload
    assert proposals_tree(project.root) == before
