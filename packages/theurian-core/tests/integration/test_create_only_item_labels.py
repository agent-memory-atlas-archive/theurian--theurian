"""An item with a ``createItem`` and no revision keeps its labels through a first revision.

``KnowledgeItem.with_revision`` overwrites sensitivity, trust level and namespace
from the first revision a content update lands, so a draft that treats such an
item as brand new re-labels it exactly as GHSA-v2qg-23fc-7fqp describes for an
item that already has a revision. The item *exists* and holds labels; having no
revision only means no ``expectedRevision`` is owed.

The item is a ``draft``, so it is visible to a caller who asks for unapproved
items, and above the deployment's ceiling it must still answer like an id that
never existed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

import pytest
from label_inheritance_support import (
    CREATE_ONLY_ITEM_ID,
    EVIDENCE,
    UPDATED_BODY,
    LabelledProject,
    cli,
    cli_propose,
    create_only_project,
    item_row,
    labelled_project,
    land,
    landing_zone,
    proposals_tree,
    scrub,
    scrubbed_proposals_tree,
    serving_grant,
    staged_metadata,
)

from theurian.daemon.runner import build_server

from mcp_wire_session import mcp_session  # isort: skip

pytestmark = pytest.mark.integration


_LEVEL_WORD: Final = re.compile(r"\b(?:public|internal|confidential|restricted)\b", re.IGNORECASE)
_GUESSED_REVISION: Final = "01K1BBBREV01234567890ABCDE"
_NO_REVISION_MESSAGE: Final = (
    "{item} has no current revision, so --expected-revision {revision} has nothing to replace."
)
_NO_REVISION_REMEDY: Final = (
    "Drop --expected-revision to draft its first revision, or correct --item-id."
)


def _mcp_arguments(item_id: str, expected: str | None, **named: str) -> dict[str, Any]:
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
        **({} if expected is None else {"expectedRevision": expected}),
        **named,
    }


def _mcp_result(
    project: LabelledProject,
    base: Path,
    ceiling: str | None,
    expected: str | None,
    **named: str,
) -> dict[str, Any]:
    grant = None if ceiling is None else serving_grant(project.data_dir, ceiling)
    with mcp_session(build_server(project.registry, grant), base / "wire") as call:
        answer = call("knowledge.proposeChange", _mcp_arguments(project.item_id, expected, **named))

    result: dict[str, Any] = answer["result"]
    return result


def test_the_create_only_fixture_is_a_labelled_item_with_no_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards every test below against a fixture that never reaches the create-only branch."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")

    row = item_row(project.root, CREATE_ONLY_ITEM_ID)

    assert row["current_revision_id"] is None
    assert (row["sensitivity"], row["trust_level"], row["namespace"], row["status"]) == (
        "confidential",
        "reviewed",
        "security",
        "draft",
    )


# -- the CLI ----------------------------------------------------------------------


def test_a_cli_draft_for_a_create_only_item_drafts_the_labels_its_create_item_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer reads the labels the first revision will carry, not the loader's defaults."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")

    code, drafted = cli_propose(project, project.item_id)

    assert code == 0, drafted
    metadata = staged_metadata(project.root, drafted)
    assert (metadata.get("sensitivity"), metadata.get("trustLevel"), metadata["namespace"]) == (
        "confidential",
        "reviewed",
        "security",
    )


def test_a_create_only_item_keeps_its_labels_after_its_first_revision_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GHSA-v2qg on the item that has no revision: accept, commit and apply re-label nothing."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    code, drafted = cli_propose(project, project.item_id)
    assert code == 0, drafted

    land(drafted["proposalId"])

    row = item_row(project.root, CREATE_ONLY_ITEM_ID)
    assert row["current_revision_id"] == drafted["revisionId"], "the revision must have landed"
    assert (row["sensitivity"], row["trust_level"], row["namespace"]) == (
        "confidential",
        "reviewed",
        "security",
    )


def test_a_cli_draft_naming_a_lower_sensitivity_for_a_create_only_item_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepting it was refused with a remedy that sent the author back to the same draft."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, project.item_id, "--sensitivity", "public")

    assert code != 0, payload
    assert "changeSensitivity" in json.dumps(payload), payload
    assert proposals_tree(project.root) == before, "a refused draft wrote into the proposals tree"


def test_a_cli_expected_revision_on_a_create_only_item_is_refused_with_the_no_revision_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The item exists but has no revision: the refusal every no-revision case gets."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")
    before = proposals_tree(project.root)

    code, payload = cli_propose(project, project.item_id, "--expected-revision", _GUESSED_REVISION)

    assert code != 0, payload
    assert payload["error"] == _NO_REVISION_MESSAGE.format(
        item=project.item_id, revision=_GUESSED_REVISION
    )
    assert payload["remedy"] == _NO_REVISION_REMEDY
    assert proposals_tree(project.root) == before, "a refused draft wrote into the proposals tree"


# -- MCP --------------------------------------------------------------------------


def test_an_mcp_draft_naming_a_lower_sensitivity_for_a_create_only_item_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is the constant, label-free one, and no ``expectedRevision`` is needed.

    The item has no revision, so the caller cannot name one; the refusal must not
    wait for a concurrency token that does not exist.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    before = proposals_tree(project.root)

    result = _mcp_result(project, tmp_path, None, None, sensitivity="public")

    assert result["isError"] is True, result
    text = result["content"][0]["text"]
    assert "changeSensitivity" in text, text
    assert _LEVEL_WORD.search(text) is None, text
    assert proposals_tree(project.root) == before, "a refused draft wrote into the proposals tree"


def test_an_mcp_draft_omitting_labels_for_a_create_only_item_drafts_its_create_item_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")

    result = _mcp_result(project, tmp_path, None, None)

    assert result["isError"] is False, result
    metadata = staged_metadata(project.root, result["structuredContent"])
    assert (metadata.get("sensitivity"), metadata.get("trustLevel"), metadata["namespace"]) == (
        "internal",
        "reviewed",
        "security",
    )


def test_an_mcp_expected_revision_on_an_in_view_create_only_item_matches_an_absent_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is the constant no-revision text, byte-identical to the absent id's.

    The item is internal under the default ceiling, so it is in view: a different
    message for it would tell a caller that the id exists and holds no revision.
    """
    held_base = tmp_path / "held"
    held_base.mkdir()
    held = create_only_project(held_base, monkeypatch, sensitivity="internal")
    absent_base = tmp_path / "absent"
    absent_base.mkdir()
    absent = labelled_project(absent_base, monkeypatch, with_item=False)
    absent = LabelledProject(absent.registry, absent.root, absent.data_dir, CREATE_ONLY_ITEM_ID, "")

    held_result = _mcp_result(held, held_base, None, _GUESSED_REVISION)
    absent_result = _mcp_result(absent, absent_base, None, _GUESSED_REVISION)

    assert held_result["isError"] is True, held_result
    assert (
        held_result["content"][0]["text"]
        == "Error executing tool knowledge.proposeChange: "
        + _NO_REVISION_MESSAGE.format(item=CREATE_ONLY_ITEM_ID, revision=_GUESSED_REVISION)
        + " "
        + _NO_REVISION_REMEDY
    ), held_result
    assert json.dumps(held_result, sort_keys=True) == json.dumps(absent_result, sort_keys=True)


def _answer_in_a_fresh_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    holds_item: bool,
    expected: str | None,
) -> tuple[str, dict[str, bytes], dict[str, bytes], dict[str, str]]:
    """The raw ``knowledge.proposeChange`` answer under the default ceiling, with the proposals
    tree before and after it, and the tree after it scrubbed of minted ids and instants."""
    base = tmp_path / name
    base.mkdir()
    if holds_item:
        project = create_only_project(base, monkeypatch, sensitivity="confidential")
    else:
        project = labelled_project(base, monkeypatch, with_item=False)
        project = LabelledProject(
            project.registry, project.root, project.data_dir, CREATE_ONLY_ITEM_ID, ""
        )
    before = proposals_tree(project.root)

    result = _mcp_result(project, base, None, expected, sensitivity="public")

    return (
        json.dumps(result, sort_keys=True),
        before,
        proposals_tree(project.root),
        scrubbed_proposals_tree(project.root),
    )


def test_a_create_only_item_above_the_ceiling_answers_like_an_id_that_never_existed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call, two corpora: the answer must not say which one holds the item.

    The call names a lower sensitivity and a revision, so the absent corpus refuses
    it as a missing item. A create-only item the caller may not see has to refuse
    in the same bytes; a label refusal here would confirm the id exists.
    """
    withheld = _answer_in_a_fresh_corpus(
        tmp_path, monkeypatch, "a", holds_item=True, expected=_GUESSED_REVISION
    )
    absent = _answer_in_a_fresh_corpus(
        tmp_path, monkeypatch, "b", holds_item=False, expected=_GUESSED_REVISION
    )

    assert "has no current revision" in absent[0], absent
    assert withheld[0] == absent[0]
    assert withheld[1] == withheld[2], "the withheld corpus's proposals tree moved"
    assert absent[1] == absent[2], "the absent corpus's proposals tree moved"
    assert withheld[1] == absent[1]


def test_a_first_revision_draft_for_a_withheld_create_only_item_matches_an_absent_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``expectedRevision`` both corpora draft, and the bytes must not differ.

    A gate bypassed for a create-only item would stage the item's own
    ``confidential`` / ``security`` labels in corpus A and the defaults in B; both
    answers are non-errors, so only the scrubbed response and staged documents
    can tell. The scrub replaces minted ids and instants and nothing else.
    """
    withheld = _answer_in_a_fresh_corpus(tmp_path, monkeypatch, "a", holds_item=True, expected=None)
    absent = _answer_in_a_fresh_corpus(tmp_path, monkeypatch, "b", holds_item=False, expected=None)

    for answer in (withheld[0], absent[0]):
        assert '"isError": false' in answer, answer
        assert "changeSensitivity" not in answer, answer
    assert absent[3], "the absent corpus must have staged a draft"
    assert scrub(withheld[0]) == scrub(absent[0])
    assert withheld[3] == absent[3]


# -- the next steps do not claim a schema default for an inherited label ---------


def test_a_cli_draft_for_a_create_only_item_is_not_told_it_publishes_the_schema_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft stages the item's own labels; a step saying it publishes defaults is false."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")

    code, drafted = cli_propose(project, project.item_id)

    assert code == 0, drafted
    metadata = staged_metadata(project.root, drafted)
    assert metadata.get("sensitivity") == "confidential", "the item's labels must be staged"
    assert [step for step in drafted["nextSteps"] if "the schema default" in step] == []


def test_a_cli_draft_for_a_brand_new_id_omitting_labels_is_still_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the test above: with no item to inherit from, the defaults warning is true."""
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")

    code, drafted = cli_propose(project, "platform.another-policy")

    assert code == 0, drafted
    (warning,) = [step for step in drafted["nextSteps"] if "the schema default" in step]
    assert "trustLevel: unverified and sensitivity: internal" in warning


# -- the recorded differential ---------------------------------------------------


def test_an_in_view_create_only_item_reads_as_absent_but_refuses_a_lowering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins both sides of the differential ADR-0032 decision 6 records.

    Its amendment says ``knowledge.get`` answers a create-only item as not present,
    in the same text it gives for an id nothing stored, while
    ``knowledge.proposeChange`` refuses a lowering for it and drafts the same call
    for an absent id: one bit about an item in the caller's view. If either side
    moves, that record must move with it.
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="internal")
    absent_id = "architecture.never-created"
    get_arguments = {"projectId": "demo", "includeUnapproved": True}
    with mcp_session(build_server(project.registry), tmp_path / "wire") as call:
        held_get = call("knowledge.get", {**get_arguments, "itemId": project.item_id})["result"]
        absent_get = call("knowledge.get", {**get_arguments, "itemId": absent_id})["result"]
        held_draft = call(
            "knowledge.proposeChange",
            _mcp_arguments(project.item_id, None, sensitivity="public"),
        )["result"]
        absent_draft = call(
            "knowledge.proposeChange", _mcp_arguments(absent_id, None, sensitivity="public")
        )["result"]

    assert held_get["isError"] is True, held_get
    assert json.dumps(held_get).replace(project.item_id, "ID") == json.dumps(absent_get).replace(
        absent_id, "ID"
    )
    assert held_draft["isError"] is True, held_draft
    assert "changeSensitivity" in held_draft["content"][0]["text"], held_draft
    assert absent_draft["isError"] is False, absent_draft


# -- accept is where an out-of-view lowering is refused ---------------------------


@pytest.mark.parametrize(
    "named",
    [
        pytest.param({"sensitivity": "public"}, id="names-public"),
        pytest.param({}, id="omits-sensitivity"),
    ],
)
def test_accept_refuses_the_lowering_an_mcp_draft_for_a_withheld_create_only_item_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: dict[str, str]
) -> None:
    """Over MCP a withheld item answers like an id nothing created, so the draft cannot refuse.

    The draft therefore stages ``public``, or the loader's ``internal`` when
    sensitivity is omitted, over a ``confidential`` item, and ``propose accept``
    is the only gate left between that and a lowering (GHSA-v2qg-23fc-7fqp).
    """
    project = create_only_project(tmp_path, monkeypatch, sensitivity="confidential")

    result = _mcp_result(project, tmp_path, None, None, **named)

    assert result["isError"] is False, result
    drafted = result["structuredContent"]
    staged = staged_metadata(project.root, drafted).get("sensitivity")
    assert staged == named.get("sensitivity", staged), "the draft must stage what was named"
    assert staged != "confidential", "the draft must not have inherited the withheld label"
    before = landing_zone(project.root)

    code, payload = cli("propose", "accept", drafted["proposalId"])

    assert code == 1, payload
    assert "lower the sensitivity" in str(payload.get("error", "")), payload
    assert project.item_id in str(payload.get("error", "")), payload
    assert landing_zone(project.root) == before, "a refused accept moved files"
    row = item_row(project.root, project.item_id)
    assert (row["sensitivity"], row["current_revision_id"]) == ("confidential", None)
