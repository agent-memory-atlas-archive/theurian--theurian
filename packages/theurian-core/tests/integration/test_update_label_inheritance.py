"""An update of an existing item keeps the labels the item already holds (GHSA-v2qg-23fc-7fqp).

``KnowledgeItem.with_revision`` adopts status, owner, trust level, sensitivity,
namespace and kind from the revision a content-path update lands. Before the fix
the drafter wrote a governance label only when the caller named it, so an update
that omitted one re-asserted the loader's default: a ``confidential`` item became
``internal``, ``reviewed`` became ``unverified`` and ``backend`` became the item
id's own prefix, with no reviewer asked to decide any of it.

Every surface that calls ``ProposalService.draft`` is driven here -- the MCP
``knowledge.proposeChange`` tool, ``theurian propose`` and
``review.generateKnowledgeCandidate`` -- and each outcome is read two ways: from
the YAML the draft staged (what a reviewer reads) and from the item row after
``propose accept`` and ``migrate apply`` (what the store then serves).

An id nothing has created is unchanged: omitted keys stay absent from the document
and the loader's defaults apply. An item that exists, with or without a revision,
inherits its labels instead.

A draft that names a sensitivity below the item's current one is refused at draft
on every surface, and the refusal names the one declassification path: a
hand-authored migration with ``changeSensitivity``. Over MCP the refusal is a
constant, and for an item the caller may not see it must be indistinguishable
from a call about an id that does not exist.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
import review_candidate_fixtures as corpus
from label_inheritance_support import (
    CREATE_ONLY_ITEM_ID,
    EVIDENCE,
    ITEM_ID,
    ROOT_REVISION_ID,
    SORTS_BEFORE_A_DRAFT,
    UPDATED_BODY,
    LabelledProject,
    cli,
    cli_ok,
    create_only_project,
    item_row,
    labelled_project,
    land,
    land_deprecation,
    land_reclassification,
    landing_zone,
    proposals_tree,
    scrub,
    scrubbed_proposals_tree,
    serving_grant,
    staged_metadata,
)
from migration_fixtures import body_pin
from review_candidate_project import ServedProject, served_project

from theurian.daemon.runner import build_server
from theurian.infrastructure.sqlite import store as store_module
from theurian.infrastructure.sqlite.connection import open_read_connection

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration


def _mcp_arguments(item_id: str, revision: str, **named: str) -> dict[str, Any]:
    return {
        "projectId": "demo",
        "itemId": item_id,
        "title": "Authentication policy",
        "kind": "architecture",
        "owner": "platform-team",
        "author": "platform-team@example.com",
        "description": "Tighten the token lifetime.",
        "body": UPDATED_BODY,
        "contentType": "text/markdown",
        "evidence": EVIDENCE,
        "sourceAnchors": [{"provider": "git", "sourceUri": "git://demo/auth-policy.md"}],
        "expectedRevision": revision,
        **named,
    }


def _mcp_result(
    project: LabelledProject, tmp_path: Path, ceiling: str | None, **named: str
) -> dict[str, Any]:
    """The raw tool result of ``knowledge.proposeChange`` for the project's item."""
    grant = None if ceiling is None else serving_grant(project.data_dir, ceiling)
    arguments = _mcp_arguments(project.item_id, project.revision_id, **named)
    with mcp_session(build_server(project.registry, grant), tmp_path / "wire") as call:
        answer = call("knowledge.proposeChange", arguments)

    result: dict[str, Any] = answer["result"]
    return result


def _mcp_update(
    project: LabelledProject, tmp_path: Path, ceiling: str | None, **named: str
) -> dict[str, Any]:
    """``knowledge.proposeChange`` for the project's item, naming only ``named`` labels."""
    result = _mcp_result(project, tmp_path, ceiling, **named)
    assert result["isError"] is False, result
    payload: dict[str, Any] = result["structuredContent"]
    assert payload["expectedRevision"] == project.revision_id, "this must be an update"
    return payload


def _cli_update(project: LabelledProject, **named: str) -> dict[str, Any]:
    """``theurian propose`` for the project's item, naming only ``named`` labels."""
    (project.root / "body.md").write_text(UPDATED_BODY, encoding="utf-8")
    options: list[str] = []
    for key, value in named.items():
        options += [f"--{key.replace('_', '-')}", value]
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
        *options,
    )
    assert payload["expectedRevision"] == project.revision_id, "this must be an update"
    return payload


def _update(
    surface: str, project: LabelledProject, tmp_path: Path, ceiling: str | None, **named: str
) -> dict[str, Any]:
    if surface == "mcp":
        mcp_names = {"trust_level": "trustLevel", "sensitivity": "sensitivity"}
        return _mcp_update(
            project, tmp_path, ceiling, **{mcp_names.get(k, k): v for k, v in named.items()}
        )
    return _cli_update(project, **named)


# -- AC-1: an omitted sensitivity is the item's current one --------------------

#: (surface, the item's sensitivity, the ceiling the MCP server runs under).
#: A confidential item is only visible to an MCP caller under a raised ceiling;
#: the public one is driven under the default, which is the shipped configuration.
_SENSITIVITY_CASES: Final = [
    pytest.param("mcp", "confidential", "confidential", id="mcp-confidential-raised-ceiling"),
    pytest.param("cli", "confidential", None, id="cli-confidential"),
    pytest.param("mcp", "public", None, id="mcp-public-default-ceiling"),
    pytest.param("cli", "public", None, id="cli-public"),
]


@pytest.mark.parametrize(("surface", "current", "ceiling"), _SENSITIVITY_CASES)
def test_an_update_omitting_sensitivity_drafts_the_items_current_sensitivity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    current: str,
    ceiling: str | None,
) -> None:
    """The reviewer reads the label the revision will carry, not the loader's default.

    An absent key is read back as ``internal`` by the loader, so a reviewer of a
    confidential item's update was shown a document that said nothing about the
    label and approved one that reclassified it.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)
    assert item_row(project.root, ITEM_ID)["sensitivity"] == current

    drafted = _update(surface, project, tmp_path, ceiling)

    assert staged_metadata(project.root, drafted).get("sensitivity") == current


@pytest.mark.parametrize(("surface", "current", "ceiling"), _SENSITIVITY_CASES)
def test_an_update_omitting_sensitivity_leaves_the_items_sensitivity_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    current: str,
    ceiling: str | None,
) -> None:
    """GHSA-v2qg: a content update must not re-classify the item it updates."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)
    drafted = _update(surface, project, tmp_path, ceiling)

    land(drafted["proposalId"])

    row = item_row(project.root, ITEM_ID)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["sensitivity"] == current


def test_a_named_higher_sensitivity_wins_over_the_current_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inheritance fills an omitted label; it never overrides a raise the caller named."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity="public")

    drafted = _update("mcp", project, tmp_path, None, sensitivity="confidential")
    land(drafted["proposalId"])

    assert item_row(project.root, ITEM_ID)["sensitivity"] == "confidential"


# -- AC-2: an omitted trust level and namespace are the item's own -------------


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_an_update_omitting_trust_level_and_namespace_drafts_the_items_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """Both would otherwise be re-derived: ``unverified``, and the id's prefix."""
    project = labelled_project(tmp_path, monkeypatch, trust_level="reviewed", namespace="backend")

    drafted = _update(surface, project, tmp_path, None)

    metadata = staged_metadata(project.root, drafted)
    assert (metadata.get("trustLevel"), metadata["namespace"]) == ("reviewed", "backend")


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_an_update_omitting_trust_level_and_namespace_leaves_the_items_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """``reviewed`` stays ``reviewed`` and ``backend`` stays ``backend`` once applied."""
    project = labelled_project(tmp_path, monkeypatch, trust_level="reviewed", namespace="backend")
    drafted = _update(surface, project, tmp_path, None)

    land(drafted["proposalId"])

    row = item_row(project.root, ITEM_ID)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert (row["trust_level"], row["namespace"]) == ("reviewed", "backend")


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_an_update_that_names_a_trust_level_and_namespace_lands_what_it_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """Inheritance fills what was omitted and never overrides what the caller named."""
    project = labelled_project(tmp_path, monkeypatch, trust_level="reviewed", namespace="backend")

    drafted = _update(
        surface, project, tmp_path, None, trust_level="authoritative", namespace="platform"
    )
    land(drafted["proposalId"])

    row = item_row(project.root, ITEM_ID)
    assert (row["trust_level"], row["namespace"]) == ("authoritative", "platform")


# -- the next steps do not claim a schema default for an update -----------------


def _defaults_warnings(payload: dict[str, Any]) -> list[str]:
    return [step for step in payload["nextSteps"] if "schema default" in step]


@pytest.mark.parametrize(
    "named", [{}, {"trust_level": "reviewed"}, {"sensitivity": "confidential"}]
)
def test_an_update_omitting_labels_is_not_told_it_publishes_the_schema_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: dict[str, str]
) -> None:
    """The drafted YAML carries the item's labels, so no default is published."""
    project = labelled_project(
        tmp_path, monkeypatch, sensitivity="internal", trust_level="reviewed"
    )

    drafted = _cli_update(project, **named)

    assert _defaults_warnings(drafted) == []
    assert not [step for step in drafted["nextSteps"] if "will publish" in step]
    metadata = staged_metadata(project.root, drafted)
    assert metadata["trustLevel"] == named.get("trust_level", "reviewed")
    assert metadata["sensitivity"] == named.get("sensitivity", "internal")


def test_a_first_revision_omitting_labels_is_still_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers an id nothing has created, whose omitted labels the loader defaults."""
    project = labelled_project(tmp_path, monkeypatch)
    (project.root / "body.md").write_text(UPDATED_BODY, encoding="utf-8")

    drafted = cli_ok(
        "propose",
        "--item-id",
        "platform.another-policy",
        "--title",
        "Another policy",
        "--kind",
        "architecture",
        "--owner",
        "platform-team",
        "--author",
        "platform-team@example.com",
        "--description",
        "A different item.",
        "--body-file",
        str(project.root / "body.md"),
        "--source-uri",
        "git://demo/another-policy.md",
        "--agent-id",
        "claude-code",
        "--task-id",
        "task-7",
        "--model",
        "claude-opus-5",
        "--reasoning",
        EVIDENCE["reasoning"],
    )

    assert drafted["expectedRevision"] is None
    (warning,) = _defaults_warnings(drafted)
    assert "trustLevel: unverified and sensitivity: internal" in warning


# -- AC-3: an id nothing has created still leaves omitted keys out ----------------------


def test_a_first_revision_over_mcp_gains_no_label_the_caller_did_not_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id nothing created has no labels to inherit, so the draft stages only what was named.

    The CLI twin is ``test_propose_cli.py``'s
    ``test_a_draft_given_none_of_the_governed_options_stages_what_it_always_did``;
    an invented ``unverified`` would claim a judgement nobody made.
    """
    project = labelled_project(tmp_path, monkeypatch)
    arguments = {
        "projectId": "demo",
        "itemId": "architecture.retry-policy",
        "title": "Retry policy",
        "kind": "architecture",
        "owner": "platform-team",
        "author": "platform-team@example.com",
        "description": "Record the retry budget.",
        "body": "# Retry policy\n\nThree attempts.\n",
        "contentType": "text/markdown",
        "evidence": EVIDENCE,
        "sourceAnchors": [{"provider": "git", "sourceUri": "git://demo/retry.md"}],
    }
    with mcp_session(build_server(project.registry), tmp_path / "wire") as call:
        answer = call("knowledge.proposeChange", arguments)
    result = answer["result"]
    assert result["isError"] is False, result

    metadata = staged_metadata(project.root, result["structuredContent"])

    assert "sensitivity" not in metadata
    assert "trustLevel" not in metadata
    assert metadata["namespace"] == "architecture"


def test_the_mcp_result_of_an_update_publishes_no_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft names the current label in the YAML only; the tool result stays label-free.

    A published label in the result would hand a caller the sensitivity of an item
    the caller was only granted the right to propose a change to.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")

    drafted = _mcp_update(project, tmp_path, "confidential")

    assert "confidential" not in repr(drafted)
    assert not {"sensitivity", "trustLevel", "namespace"} & set(drafted)


# -- AC-7: the candidate path --------------------------------------------------

_CANDIDATE_RECLASSIFIED: Final = "01K1CCCCCC01234567890ABCDE"
_CANDIDATE_CREATE_ONLY: Final = "01K1AAAAAB01234567890ABCDE"


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ServedProject]:
    yield from served_project(
        tmp_path, monkeypatch, records=corpus.evidence_records(), withheld=frozenset()
    )


def _candidate(
    served: ServedProject, tmp_path: Path, item_id: str, grant_ceiling: str | None, **extra: Any
) -> dict[str, Any]:
    grant = None if grant_ceiling is None else serving_grant(tmp_path / "datadir", grant_ceiling)
    arguments: dict[str, Any] = {
        "projectId": "demo",
        "repository": corpus.REPOSITORY,
        "recordKey": corpus.THREAD_SATISFYING,
        "fixCommit": served.verifying,
        "itemId": item_id,
        "title": "Acquire locks after reads in retry-eligible paths",
        "body": "Acquire locks after reads, never before, in retry-eligible paths.\n",
        "kind": "convention",
        "category": "reliability-rule",
        "owner": "platform-team",
        "author": "dana@example.com",
        "description": "Generalise the deadlock thread into a locking rule",
        "evidence": EVIDENCE,
        "sourceAnchors": [
            {
                "provider": "github",
                "sourceUri": (
                    f"https://github.com/{corpus.REPOSITORY}/pull/"
                    f"{corpus.PULL_REQUEST_CI_PASSED}#discussion_r1"
                ),
                "repository": corpus.REPOSITORY,
                "filePath": corpus.FILE_PATH,
            }
        ],
        **extra,
    }
    with mcp_session(build_server(served.registry, grant), tmp_path / "wire") as call:
        answer = call("review.generateKnowledgeCandidate", arguments)

    result: dict[str, Any] = answer["result"]
    assert result["isError"] is False, result
    payload: dict[str, Any] = result["structuredContent"]
    return payload


def test_a_candidate_for_a_new_item_still_names_internal(
    served: ServedProject, tmp_path: Path
) -> None:
    """An id nothing has created has no label to inherit, and a candidate is ``internal``."""
    drafted = _candidate(served, tmp_path, "reliability.retry-lock-order", None)

    assert staged_metadata(served.root, drafted)["sensitivity"] == "internal"


def test_a_candidate_update_of_a_confidential_item_names_the_items_sensitivity(
    served: ServedProject, tmp_path: Path
) -> None:
    """The candidate path used to name ``internal`` on every update, lowering a confidential item.

    It now leaves the label out of an update, so the draft names the item's own.
    """
    land_reclassification(served.root, _CANDIDATE_RECLASSIFIED, ITEM_ID, "confidential")
    assert item_row(served.root, ITEM_ID)["sensitivity"] == "confidential"
    current = item_row(served.root, ITEM_ID)["current_revision_id"]

    drafted = _candidate(served, tmp_path, ITEM_ID, "confidential", expectedRevision=current)

    assert staged_metadata(served.root, drafted)["sensitivity"] == "confidential"


def test_a_candidate_update_of_a_confidential_item_leaves_it_confidential(
    served: ServedProject, tmp_path: Path
) -> None:
    """GHSA-v2qg through ``review.generateKnowledgeCandidate``, read from the store."""
    land_reclassification(served.root, _CANDIDATE_RECLASSIFIED, ITEM_ID, "confidential")
    current = item_row(served.root, ITEM_ID)["current_revision_id"]
    drafted = _candidate(served, tmp_path, ITEM_ID, "confidential", expectedRevision=current)

    land(drafted["proposalId"])

    row = item_row(served.root, ITEM_ID)
    assert row["current_revision_id"] == drafted["revisionId"], "the update must have landed"
    assert row["sensitivity"] == "confidential"


def test_a_candidate_for_a_deprecated_item_answers_like_an_id_nothing_created(
    served: ServedProject, tmp_path: Path
) -> None:
    """The candidate path's lookup answers ``None`` for a retired item.

    Same-corpus form: ``served_project`` always lands the one item, so there is
    no corpus without it to compare against. A deprecated item and a never-created
    id get the same call, and once the two ids are normalised the same answer and
    the same staged label.
    """
    land_deprecation(served.root, SORTS_BEFORE_A_DRAFT, ITEM_ID)
    assert item_row(served.root, ITEM_ID)["status"] == "deprecated"
    absent_id = "reliability.retry-lock-order"

    retired = _candidate(served, tmp_path, ITEM_ID, None)
    absent = _candidate(served, tmp_path, absent_id, None)

    def normalised(payload: dict[str, Any], item_id: str) -> str:
        text = json.dumps(payload, sort_keys=True)
        return scrub(text.replace(item_id, "<ID>").replace(item_id.replace(".", "/"), "<ID>"))

    assert normalised(retired, ITEM_ID) == normalised(absent, absent_id)
    retired_labels = staged_metadata(served.root, retired)
    absent_labels = staged_metadata(served.root, absent)
    # An id nothing created takes its namespace from the id's own prefix; so does this one.
    assert (retired_labels.pop("namespace"), absent_labels.pop("namespace")) == (
        "architecture",
        "reliability",
    )
    assert retired_labels == absent_labels


def _land_a_create_only_item(served: ServedProject, sensitivity: str) -> None:
    """An item with a ``createItem`` and no revision, the only shape the candidate path drafts
    without ``expectedRevision`` for an item the caller cannot see."""
    (served.root / f".theurian/migrations/{_CANDIDATE_CREATE_ONLY}-placeholder.yaml").write_text(
        f"""apiVersion: theurian.dev/v1
id: {_CANDIDATE_CREATE_ONLY}
createdAt: 2026-09-30T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {CREATE_ONLY_ITEM_ID}
    kind: architecture
    namespace: security
    owner: platform-team
    sensitivity: {sensitivity}
    trustLevel: reviewed
"""
    )
    cli_ok("migrate", "apply")
    row = item_row(served.root, CREATE_ONLY_ITEM_ID)
    assert (row["sensitivity"], row["current_revision_id"]) == (sensitivity, None)


def test_accept_refuses_a_candidate_for_a_deprecated_create_only_item_as_a_readmission(
    served: ServedProject, tmp_path: Path
) -> None:
    """The candidate lookup answers ``None`` for a retired item, so only ``accept`` can refuse.

    The draft is indistinguishable from one for an absent id, which stages
    ``status: approved`` over a deprecated item (GHSA-v2qg-23fc-7fqp). An item
    with a revision cannot reach this: the draft without ``expectedRevision``
    fails at accept's replay, and with it the lookup's ``None`` refuses at draft.
    """
    _land_a_create_only_item(served, "internal")
    land_deprecation(served.root, SORTS_BEFORE_A_DRAFT, CREATE_ONLY_ITEM_ID)
    assert item_row(served.root, CREATE_ONLY_ITEM_ID)["status"] == "deprecated"
    drafted = _candidate(served, tmp_path, CREATE_ONLY_ITEM_ID, None)
    before = landing_zone(served.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert "readmit" in error and CREATE_ONLY_ITEM_ID in error, payload
    assert "restoreItem" in str(payload.get("remedy", "")), payload
    assert landing_zone(served.root) == before, "a refused accept moved files"
    assert item_row(served.root, CREATE_ONLY_ITEM_ID)["status"] == "deprecated"


def test_accept_refuses_a_candidate_for_a_deprecated_item_with_a_revision_at_the_replay(
    served: ServedProject, tmp_path: Path
) -> None:
    """The sibling of the test above: with a revision, the refusal is the engine's, not the floor's.

    The candidate drafts a first revision for what the caller sees as an absent
    id, and the replay holds one already.
    """
    land_deprecation(served.root, SORTS_BEFORE_A_DRAFT, ITEM_ID)
    drafted = _candidate(served, tmp_path, ITEM_ID, None)
    before = landing_zone(served.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code != 0, payload
    assert f"Revision conflict on {ITEM_ID}" in str(payload.get("error", "")), payload
    assert landing_zone(served.root) == before, "a refused accept moved files"


def test_accept_refuses_a_candidate_for_a_withheld_create_only_item_as_a_lowering(
    served: ServedProject, tmp_path: Path
) -> None:
    """Above the serving ceiling the candidate lookup answers ``None``, so it drafts ``internal``.

    The item holds ``confidential`` on its ``createItem`` alone; ``accept`` is the
    only place that can see the draft would lower it.
    """
    _land_a_create_only_item(served, "confidential")
    drafted = _candidate(served, tmp_path, CREATE_ONLY_ITEM_ID, None)
    assert staged_metadata(served.root, drafted)["sensitivity"] == "internal"
    before = landing_zone(served.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    error = str(payload.get("error", ""))
    assert "lower the sensitivity" in error and CREATE_ONLY_ITEM_ID in error, payload
    assert landing_zone(served.root) == before, "a refused accept moved files"
    assert item_row(served.root, CREATE_ONLY_ITEM_ID)["sensitivity"] == "confidential"


# -- Refusal at draft: a lower sensitivity is not an update's to name ------------

_LEVELS: Final = ("public", "internal", "confidential", "restricted")
_LEVEL_WORD: Final = re.compile(r"\b(?:" + "|".join(_LEVELS) + r")\b", re.IGNORECASE)

#: (the item's sensitivity, the lower one the caller names, the ceiling the MCP
#: server runs under). Each item is visible to the caller, so the refusal is the
#: one about labels and not the one about a missing item.
_LOWERING_CASES: Final = [
    pytest.param("confidential", "internal", "confidential", id="confidential-to-internal"),
    pytest.param("confidential", "public", "confidential", id="confidential-to-public"),
    pytest.param("internal", "public", None, id="internal-to-public"),
    pytest.param("restricted", "confidential", "restricted", id="restricted-to-confidential"),
]


@pytest.mark.parametrize(("current", "named", "ceiling"), _LOWERING_CASES)
def test_an_mcp_update_naming_a_lower_sensitivity_is_refused_without_naming_a_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current: str,
    named: str,
    ceiling: str | None,
) -> None:
    """The refusal points at the declassification path and tells the caller nothing else.

    A message that named the item's current label would hand a caller the
    sensitivity of an item it may only propose changes to, so the text is
    constant: no level word appears in it, the one the caller named included.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)
    before = proposals_tree(project.root)

    result = _mcp_result(project, tmp_path, ceiling, sensitivity=named)

    assert result["isError"] is True, result
    text = result["content"][0]["text"]
    assert "changeSensitivity" in text, text
    assert _LEVEL_WORD.search(text) is None, text
    assert proposals_tree(project.root) == before, "a refused draft wrote into the proposals tree"


_EQUAL_OR_HIGHER: Final = [
    pytest.param("confidential", "confidential", "confidential", id="confidential-kept"),
    pytest.param("internal", "confidential", None, id="internal-raised-to-confidential"),
    pytest.param("public", "public", None, id="public-kept"),
    pytest.param("public", "internal", None, id="public-raised-to-internal"),
    pytest.param(
        "confidential", "restricted", "restricted", id="confidential-raised-to-restricted"
    ),
]


@pytest.mark.parametrize(("current", "named", "ceiling"), _EQUAL_OR_HIGHER)
def test_an_mcp_update_naming_an_equal_or_higher_sensitivity_drafts_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current: str,
    named: str,
    ceiling: str | None,
) -> None:
    """The floor refuses lowering only: keeping a label and raising it both draft.

    Without these a refusal of every sensitivity-naming update would satisfy the
    lowering tests above.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)

    drafted = _mcp_update(project, tmp_path, ceiling, sensitivity=named)

    assert staged_metadata(project.root, drafted)["sensitivity"] == named


# -- One corpus: a withheld id answers like an absent id ------------------------

_ABSENT_ITEM: Final = "architecture.never-existed"
_ABSENT_REVISION: Final = "01K1BBBREV01234567890ABCDE"

#: (the item's sensitivity, its status, the ceiling the server runs under, the
#: lower sensitivity the caller names). Each item is one the caller may not see:
#: above the default ceiling, retired, or above a raised ceiling. The retired items
#: sit *within* their ceiling, so it is the status gate alone that withholds them.
_WITHHELD_CASES: Final = [
    pytest.param(
        ("confidential", "approved", None, "internal"), id="above-default-ceiling-internal"
    ),
    pytest.param(("confidential", "approved", None, "public"), id="above-default-ceiling-public"),
    pytest.param(("internal", "deprecated", None, "public"), id="deprecated-internal"),
    pytest.param(("internal", "superseded", None, "public"), id="superseded-internal"),
    pytest.param(("internal", "rejected", None, "public"), id="rejected-internal"),
    pytest.param(
        ("confidential", "deprecated", "confidential", "internal"), id="deprecated-confidential"
    ),
    pytest.param(
        ("restricted", "approved", "confidential", "confidential"), id="above-raised-ceiling"
    ),
    pytest.param(
        ("restricted", "approved", "confidential", "public"), id="above-raised-ceiling-public"
    ),
]


@pytest.fixture
def selects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every SELECT a connection from ``open_read_connection`` executes from here on, in order.

    Counted where the statement reaches SQLite, not by the store method that issued
    it: a read added under any name opens its connection through
    ``open_read_connection`` and shows up here, which a spy on four method names did
    not see. A read on a raw ``sqlite3`` connection (a body read, say) is invisible.
    """
    statements: list[str] = []
    opened = open_read_connection

    def record(statement: str) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    def traced(path: Path) -> sqlite3.Connection:
        connection = opened(path)
        connection.set_trace_callback(record)
        return connection

    monkeypatch.setattr(store_module, "open_read_connection", traced)
    return statements


def _normalised(result: dict[str, Any], item_id: str, revision: str) -> str:
    return json.dumps(result, sort_keys=True).replace(item_id, "<ID>").replace(revision, "<REV>")


@pytest.mark.parametrize("case", _WITHHELD_CASES)
def test_naming_a_lower_sensitivity_on_a_withheld_item_answers_like_an_absent_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selects: list[str],
    case: tuple[str, str, str | None, str],
) -> None:
    """The label comparison must not run before the visibility gate.

    A refusal about *labels* for an item the caller may not see tells it the item
    exists, and which side of its label the named level falls. This is the
    same-corpus form: inside one project, a withheld id and a different never-stored
    id must produce the same response once the ids are normalised, read the store
    the same number of times, and write nothing. The two-corpora form, which needs
    no normalisation, is
    ``test_the_same_call_on_a_corpus_without_the_item_gets_the_same_bytes``.
    """
    current, status, ceiling, named = case
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current, status=status)
    row = item_row(project.root, ITEM_ID)
    assert (row["sensitivity"], row["status"]) == (current, status), "the item must be withheld"
    grant = None if ceiling is None else serving_grant(project.data_dir, ceiling)
    before = proposals_tree(project.root)

    with mcp_session(build_server(project.registry, grant), tmp_path / "wire") as call:
        selects.clear()
        withheld = call(
            "knowledge.proposeChange",
            _mcp_arguments(ITEM_ID, project.revision_id, sensitivity=named),
        )["result"]
        withheld_selects = [text.replace(ITEM_ID, "<ID>") for text in selects]
        selects.clear()
        absent = call(
            "knowledge.proposeChange",
            _mcp_arguments(_ABSENT_ITEM, _ABSENT_REVISION, sensitivity=named),
        )["result"]
        absent_selects = [text.replace(_ABSENT_ITEM, "<ID>") for text in selects]

    assert "has no current revision" in withheld["content"][0]["text"], withheld
    assert _normalised(withheld, ITEM_ID, project.revision_id) == _normalised(
        absent, _ABSENT_ITEM, _ABSENT_REVISION
    )
    assert absent_selects, "the trace never saw the lookup, so the equality below says nothing"
    assert len(withheld_selects) == len(absent_selects)
    assert withheld_selects == absent_selects
    assert proposals_tree(project.root) == before


def _propose_change_in_a_fresh_corpus(  # noqa: PLR0913 -- one corpus per call
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    ceiling: str | None,
    named: str,
    *,
    holds_item: bool,
    current: str,
    status: str,
) -> tuple[str, dict[str, bytes]]:
    """The raw ``knowledge.proposeChange`` answer from one corpus, and its proposals tree after."""
    base = tmp_path / name
    base.mkdir()
    project = labelled_project(
        base, monkeypatch, sensitivity=current, status=status, with_item=holds_item
    )
    grant = None if ceiling is None else serving_grant(project.data_dir, ceiling)
    before = proposals_tree(project.root)
    arguments = _mcp_arguments(ITEM_ID, ROOT_REVISION_ID, sensitivity=named)

    with mcp_session(build_server(project.registry, grant), base / "wire") as call:
        answer = call("knowledge.proposeChange", arguments)

    assert proposals_tree(project.root) == before, "a refused call wrote into the proposals tree"
    return json.dumps(answer["result"], sort_keys=True), before


@pytest.mark.parametrize("case", _WITHHELD_CASES)
def test_the_same_call_on_a_corpus_without_the_item_gets_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, str | None, str]
) -> None:
    """One call against two corpora must not tell the caller which holds the item.

    Corpus A holds an item the caller may not see; corpus B is identical except
    that the item never existed. The call is the same bytes to both: same item id,
    same expected revision, a sensitivity below the item's real one. Any difference
    between the answers, a label-dependent word or a different revision, is an
    existence oracle, so the comparison is on the raw response with no
    normalisation. Each corpus's proposals tree is also unchanged.
    """
    current, status, ceiling, named = case

    withheld, _ = _propose_change_in_a_fresh_corpus(
        tmp_path, monkeypatch, "a", ceiling, named, holds_item=True, current=current, status=status
    )
    absent, _ = _propose_change_in_a_fresh_corpus(
        tmp_path, monkeypatch, "b", ceiling, named, holds_item=False, current=current, status=status
    )

    assert "has no current revision" in withheld, withheld
    assert withheld == absent


def _first_revision_call_in_a_fresh_corpus(  # noqa: PLR0913 -- one corpus per call
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selects: list[str],
    name: str,
    shape: str,
    *,
    holds_item: bool,
    with_expected_revision: bool,
) -> tuple[str, dict[str, str], list[str]]:
    """One ``knowledge.proposeChange``, with or without ``expectedRevision``, from one corpus.

    Returns the answer, the proposals tree after it and the SELECTs it ran, each
    scrubbed of minted ids and instants. ``deprecated-create-only`` retires an item
    that has no revision; ``confidential-create-only`` holds one above the default
    ceiling; ``rejected`` holds an item whose revision says so.
    """
    base = tmp_path / name
    base.mkdir()
    if shape in ("deprecated-create-only", "confidential-create-only"):
        item_id = CREATE_ONLY_ITEM_ID
        if holds_item:
            deprecated = shape == "deprecated-create-only"
            project = create_only_project(
                base, monkeypatch, sensitivity="internal" if deprecated else "confidential"
            )
            if deprecated:
                land_deprecation(project.root, SORTS_BEFORE_A_DRAFT, item_id)
                assert item_row(project.root, item_id)["status"] == "deprecated"
            else:
                assert item_row(project.root, item_id)["sensitivity"] == "confidential"
        else:
            project = labelled_project(base, monkeypatch, with_item=False)
    else:
        item_id = ITEM_ID
        project = labelled_project(base, monkeypatch, status="rejected", with_item=holds_item)
        if holds_item:
            assert item_row(project.root, item_id)["status"] == "rejected"
    arguments = _mcp_arguments(item_id, ROOT_REVISION_ID)
    if not with_expected_revision:
        del arguments["expectedRevision"]

    with mcp_session(build_server(project.registry), base / "wire") as call:
        selects.clear()
        answer = call("knowledge.proposeChange", arguments)
        trace = [scrub(text) for text in selects]

    tree = scrubbed_proposals_tree(project.root)
    return scrub(json.dumps(answer["result"], sort_keys=True)), tree, trace


@pytest.mark.parametrize("with_expected_revision", [False, True], ids=["bare", "expected"])
@pytest.mark.parametrize(
    "shape", ["deprecated-create-only", "confidential-create-only", "rejected"]
)
def test_a_draft_for_a_withheld_item_matches_an_absent_id_across_two_corpora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selects: list[str],
    shape: str,
    with_expected_revision: bool,
) -> None:
    """The MCP lookup answers ``None`` for a withheld item, so the corpora must not differ.

    The status refusal belongs to the CLI, whose view is unscoped. Over MCP a
    retired item, or a create-only one above the ceiling, drafts or refuses exactly
    like an id nothing created; a lookup that returned it would refuse here, which
    tells the caller the id exists. Same call, two corpora, so the answer, the staged
    documents and the SQL read are compared with no per-id normalisation. The trace
    is of reads through ``open_read_connection``; see ``selects``.
    """
    held = _first_revision_call_in_a_fresh_corpus(
        tmp_path,
        monkeypatch,
        selects,
        "a",
        shape,
        holds_item=True,
        with_expected_revision=with_expected_revision,
    )
    absent = _first_revision_call_in_a_fresh_corpus(
        tmp_path,
        monkeypatch,
        selects,
        "b",
        shape,
        holds_item=False,
        with_expected_revision=with_expected_revision,
    )

    if with_expected_revision:
        assert "has no current revision" in absent[0], absent
    else:
        assert '"isError": false' in absent[0], absent
        assert absent[1], "the absent corpus must have staged a draft"
    assert absent[2], "the trace never saw the lookup, so the equality below says nothing"
    assert held[0] == absent[0]
    assert held[1] == absent[1]
    assert held[2] == absent[2]


# -- The CLI ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "named"),
    [
        pytest.param("confidential", "internal", id="confidential-to-internal"),
        pytest.param("internal", "public", id="internal-to-public"),
        pytest.param("restricted", "confidential", id="restricted-to-confidential"),
    ],
)
def test_theurian_propose_naming_a_lower_sensitivity_refuses_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current: str, named: str
) -> None:
    """The CLI refuses at draft and names the declassification path, as the tool does.

    The CLI is the operator's own surface, so unlike the tool's constant text its
    refusal is free to say more; what it owes is the path.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)
    before = proposals_tree(project.root)
    (project.root / "body.md").write_text(UPDATED_BODY, encoding="utf-8")

    code, payload = cli(
        "propose",
        "--item-id", project.item_id,
        "--title", "Authentication policy",
        "--kind", "architecture",
        "--owner", "platform-team",
        "--author", "platform-team@example.com",
        "--description", "Tighten the token lifetime.",
        "--body-file", str(project.root / "body.md"),
        "--source-uri", "git://demo/auth-policy.md",
        "--agent-id", "claude-code",
        "--task-id", "task-7",
        "--model", "claude-opus-5",
        "--reasoning", EVIDENCE["reasoning"],
        "--expected-revision", project.revision_id,
        "--sensitivity", named,
    )  # fmt: skip

    assert code != 0, payload
    assert "changeSensitivity" in json.dumps(payload), payload
    assert proposals_tree(project.root) == before, "a refused draft wrote into the proposals tree"


# -- The loader default for a hand-authored document is unchanged -----------------


def test_a_hand_authored_revision_omitting_sensitivity_and_trust_level_loads_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader's ``internal`` / ``unverified`` is what a document that says nothing means.

    Inheritance is the drafter's: a migration written by hand still states its own
    labels or accepts the defaults, and the schema and loader contract for it is
    not changed.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity=None, trust_level=None)

    row = item_row(project.root, ITEM_ID)

    assert (row["sensitivity"], row["trust_level"]) == ("internal", "unverified")


def test_a_hand_authored_update_omitting_sensitivity_and_trust_level_lands_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-written update still re-asserts the loader defaults, by decision.

    Inheritance is the drafter's, so this residual is by decision (ADR-0032 as
    amended; ``docs/protocol/migrations.md``'s "An upsertRevision re-labels the
    item"), and a change that closes it must change that decision first.
    """
    project = labelled_project(tmp_path, monkeypatch, sensitivity="confidential")
    assert item_row(project.root, ITEM_ID)["sensitivity"] == "confidential"
    body_name = "auth-policy.hand-authored.md"
    (project.root / f".theurian/knowledge/architecture/{body_name}").write_text(UPDATED_BODY)
    migration_id = SORTS_BEFORE_A_DRAFT
    (project.root / f".theurian/migrations/{migration_id}-hand-update.yaml").write_text(
        f"""apiVersion: theurian.dev/v1
id: {migration_id}
createdAt: 2026-09-30T10:00:00+09:00
author: engineer@example.com
operations:
  - op: upsertRevision
    itemId: {ITEM_ID}
    revisionId: 01K1BBBREV01234567890ABCDE
    expectedRevision: {project.revision_id}
    contentFile: ../knowledge/architecture/{body_name}
    contentSha256: {body_pin(UPDATED_BODY)}
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

    cli_ok("migrate", "apply")

    row = item_row(project.root, ITEM_ID)
    assert row["current_revision_id"] == "01K1BBBREV01234567890ABCDE", "the update must have landed"
    assert (row["sensitivity"], row["trust_level"]) == ("internal", "unverified")


@pytest.mark.parametrize(
    ("current", "named"),
    [
        pytest.param("public", "internal", id="public-raised-to-internal"),
        pytest.param("confidential", "restricted", id="confidential-raised-to-restricted"),
    ],
)
def test_theurian_propose_naming_a_higher_sensitivity_drafts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current: str, named: str
) -> None:
    """A refusal of every named sensitivity would satisfy the lowering test above."""
    project = labelled_project(tmp_path, monkeypatch, sensitivity=current)

    drafted = _cli_update(project, sensitivity=named)

    assert staged_metadata(project.root, drafted)["sensitivity"] == named
