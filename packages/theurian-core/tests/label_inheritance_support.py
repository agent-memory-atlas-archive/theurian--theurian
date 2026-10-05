"""A registered project holding one item with chosen governance labels.

Shared by the tests that drive an update of an existing item through every
drafting surface (``test_update_label_inheritance.py``,
``test_accept_sensitivity_floor.py``). The item comes from a **root** migration,
one with no ``dependsOn``, so replay order is the migration ids' own and no test
here depends on how a dependency chain sorts.

Nothing here touches the developer's machine: the project, its git repository and
its data directory are created under the caller's ``tmp_path``, and the CLI runs
in-process with ``THEURIAN_DATA_DIR`` redirected.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from git_harness import commit_migrations
from migration_fixtures import body_pin
from typer.testing import CliRunner

from theurian.application.authorization import (
    AuthorizationGrant,
    StaticAuthorizationProvider,
    load_serving_profile,
    serving_profile_path,
)
from theurian.application.project_service import ProjectRegistry
from theurian.cli.main import app

runner = CliRunner()

ITEM_ID: Final = "architecture.auth-policy"
ROOT_MIGRATION_ID: Final = "01K1AAAAAA01234567890ABCDE"
ROOT_REVISION_ID: Final = "01K1AAAREV01234567890ABCDE"
BODY: Final = "# Authentication policy\n\nEvery call carries a signed token.\n"

#: A ULID that sorts after every id the clock mints today, and one that sorts
#: after the root migration but before them. The two ends of the replay order a
#: hand-authored migration can take relative to a draft minted now.
SORTS_AFTER_A_DRAFT: Final = "7ZZZZZZZZZ01234567890ABCDE"
SORTS_BEFORE_A_DRAFT: Final = "01K1BBBBBB01234567890ABCDE"

EVIDENCE: Final = {
    "agentId": "claude-code",
    "taskId": "task-7",
    "model": "claude-opus-5",
    "reasoning": "The alias review settled the new wording of the policy.",
}

UPDATED_BODY: Final = "# Authentication policy\n\nTokens live one hour.\n"


@dataclass(frozen=True)
class LabelledProject:
    registry: ProjectRegistry
    root: Path
    data_dir: Path
    item_id: str
    revision_id: str


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)  # noqa: S603, S607


def cli(*args: str) -> tuple[int, dict[str, Any]]:
    """One CLI command in-process, with ``migrate apply`` preceded by the commit it requires."""
    if args[:2] == ("migrate", "apply"):
        commit_migrations()
    result = runner.invoke(app, [*args, "--json"], catch_exceptions=False)
    stream = result.stdout if result.exit_code == 0 else (result.stderr or result.stdout)
    return result.exit_code, json.loads(stream) if stream.strip() else {}


def cli_ok(*args: str) -> dict[str, Any]:
    code, payload = cli(*args)
    assert code == 0, payload
    return payload


def root_migration(  # noqa: PLR0913 -- the labels a root migration can carry
    *,
    item_id: str,
    revision_id: str,
    namespace: str,
    sensitivity: str | None,
    trust_level: str | None,
    status: str,
) -> str:
    optional = ""
    if trust_level is not None:
        optional += f"      trustLevel: {trust_level}\n"
    if sensitivity is not None:
        optional += f"      sensitivity: {sensitivity}\n"
    return f"""apiVersion: theurian.dev/v1
id: {ROOT_MIGRATION_ID}
createdAt: 2026-08-02T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {item_id}
    kind: architecture
    namespace: {namespace}
    owner: platform-team
  - op: upsertRevision
    itemId: {item_id}
    revisionId: {revision_id}
    contentFile: ../knowledge/architecture/auth-policy.md
    contentSha256: {body_pin(BODY)}
    metadata:
      title: Authentication policy
      contentType: text/markdown
      kind: architecture
      namespace: {namespace}
      status: {status}
      owner: platform-team
{optional}      sourceAnchors:
        - provider: git
          sourceUri: git://demo/auth-policy.md
"""


def labelled_project(  # noqa: PLR0913 -- the labels a fixture item can carry
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sensitivity: str | None = None,
    trust_level: str | None = "reviewed",
    namespace: str = "backend",
    status: str = "approved",
    with_item: bool = True,
) -> LabelledProject:
    """A registered, applied ``demo`` project whose one item carries these labels.

    ``with_item=False`` builds the same project with no migration, so the item
    never existed: the second corpus of a withheld-versus-absent comparison.

    The process works inside the project (``chdir``) and against a redirected data
    directory, because the CLI resolves both from the environment.
    """
    root = tmp_path / "demo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")

    data_dir = tmp_path / "datadir"
    monkeypatch.setenv("THEURIAN_DATA_DIR", str(data_dir))
    monkeypatch.chdir(root)

    cli_ok("init")
    if with_item:
        (root / ".theurian/knowledge/architecture/auth-policy.md").write_text(BODY)
        (root / f".theurian/migrations/{ROOT_MIGRATION_ID}-auth.yaml").write_text(
            root_migration(
                item_id=ITEM_ID,
                revision_id=ROOT_REVISION_ID,
                namespace=namespace,
                sensitivity=sensitivity,
                trust_level=trust_level,
                status=status,
            )
        )
    cli_ok("project", "register")
    cli_ok("migrate", "apply")
    return LabelledProject(
        registry=ProjectRegistry.default(data_dir),
        root=root,
        data_dir=data_dir,
        item_id=ITEM_ID,
        revision_id=ROOT_REVISION_ID,
    )


CREATE_ONLY_ITEM_ID: Final = "architecture.placeholder"


def create_only_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sensitivity: str,
    trust_level: str = "reviewed",
    namespace: str = "security",
) -> LabelledProject:
    """A registered, applied ``demo`` project whose one item has a ``createItem`` and no revision.

    Such an item carries its labels on the ``createItem`` operation alone and its
    status is ``draft``. ``revision_id`` is empty because the item has no current
    revision to expect.
    """
    project = labelled_project(tmp_path, monkeypatch, with_item=False)
    (project.root / f".theurian/migrations/{ROOT_MIGRATION_ID}-placeholder.yaml").write_text(
        f"""apiVersion: theurian.dev/v1
id: {ROOT_MIGRATION_ID}
createdAt: 2026-08-02T10:00:00+09:00
author: engineer@example.com
operations:
  - op: createItem
    itemId: {CREATE_ONLY_ITEM_ID}
    kind: architecture
    namespace: {namespace}
    owner: platform-team
    sensitivity: {sensitivity}
    trustLevel: {trust_level}
"""
    )
    cli_ok("migrate", "apply")
    return LabelledProject(
        registry=project.registry,
        root=project.root,
        data_dir=project.data_dir,
        item_id=CREATE_ONLY_ITEM_ID,
        revision_id="",
    )


def cli_propose(project: LabelledProject, item_id: str, *extra: str) -> tuple[int, dict[str, Any]]:
    """``theurian propose`` for ``item_id`` with the fixed content, plus ``extra`` options."""
    (project.root / "body.md").write_text(UPDATED_BODY, encoding="utf-8")
    return cli(
        "propose",
        "--item-id", item_id,
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
        *extra,
    )  # fmt: skip


def landing_zone(root: Path) -> dict[str, bytes]:
    """Every file ``accept`` could move into or out of: proposals, migrations and knowledge."""
    found: dict[str, bytes] = {}
    for name in ("proposals", "proposals-local", "migrations", "knowledge"):
        base = root / ".theurian" / name
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def serving_grant(data_dir: Path, ceiling: str) -> AuthorizationGrant:
    """The grant the daemon would load from an operator-declared ceiling."""
    path = serving_profile_path(data_dir)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(ceiling, encoding="utf-8")
    path.chmod(0o600)
    return StaticAuthorizationProvider(load_serving_profile(data_dir)).deployment_grant()


def item_row(root: Path, item_id: str) -> dict[str, Any]:
    """The governance columns the item currently holds, read from the active state database."""
    state = root / ".theurian" / "state"
    active = json.loads((state / "active.json").read_text(encoding="utf-8"))
    database = state / active["databaseFilename"]
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT sensitivity, trust_level, namespace, status, current_revision_id "
            "FROM knowledge_items WHERE item_id = ?",
            (item_id,),
        ).fetchall()
    assert len(rows) == 1, rows
    return dict(rows[0])


def proposals_tree(root: Path) -> dict[str, bytes]:
    """Every file under both proposal locations, by relative path, with its bytes."""
    found: dict[str, bytes] = {}
    for parent in (root / ".theurian/proposals", root / ".theurian/proposals-local"):
        if parent.exists():
            for path in sorted(parent.rglob("*")):
                if path.is_file():
                    found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def staged_upsert(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """The ``upsertRevision`` operation of the migration a draft staged."""
    migration = root / payload["proposalDirectory"] / payload["migrationFile"]
    document = yaml.safe_load(migration.read_text(encoding="utf-8"))
    upsert = next(op for op in document["operations"] if op["op"] == "upsertRevision")
    assert isinstance(upsert, dict)
    return upsert


def staged_metadata(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = staged_upsert(root, payload)["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def land(proposal_id: str) -> None:
    """Accept a proposal, then apply the migration set."""
    cli_ok("propose", "accept", proposal_id)
    cli_ok("migrate", "apply")


def reclassification(migration_id: str, item_id: str, sensitivity: str) -> str:
    """A hand-authored ``changeSensitivity`` migration: the one declassification path."""
    return f"""apiVersion: theurian.dev/v1
id: {migration_id}
createdAt: 2026-09-30T10:00:00+09:00
author: engineer@example.com
operations:
  - op: changeSensitivity
    itemId: {item_id}
    sensitivity: {sensitivity}
    reason: reclassified after review
"""


def land_reclassification(root: Path, migration_id: str, item_id: str, sensitivity: str) -> None:
    """Write, commit and apply a ``changeSensitivity`` migration."""
    (root / f".theurian/migrations/{migration_id}-reclassify.yaml").write_text(
        reclassification(migration_id, item_id, sensitivity)
    )
    cli_ok("migrate", "apply")


def deprecation(migration_id: str, item_id: str) -> str:
    """A hand-authored ``deprecateItem`` migration: the item's status becomes ``deprecated``."""
    return f"""apiVersion: theurian.dev/v1
id: {migration_id}
createdAt: 2026-09-30T10:00:00+09:00
author: engineer@example.com
operations:
  - op: deprecateItem
    itemId: {item_id}
    reason: retired after review
"""


def land_deprecation(root: Path, migration_id: str, item_id: str) -> None:
    """Write, commit and apply a ``deprecateItem`` migration."""
    (root / f".theurian/migrations/{migration_id}-deprecate.yaml").write_text(
        deprecation(migration_id, item_id)
    )
    cli_ok("migrate", "apply")


_ULID: Final = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")
_INSTANT: Final = re.compile(r"\d{4}-\d\d-\d\dT[\d:.+Z-]+")


def scrub(text: str) -> str:
    """The text with every minted id and instant replaced, so two corpora compare on shape."""
    return _INSTANT.sub("<T>", _ULID.sub("<ULID>", text))


def scrubbed_proposals_tree(root: Path) -> dict[str, str]:
    """``proposals_tree`` with each path and document scrubbed of minted ids and instants."""
    return {scrub(path): scrub(data.decode("utf-8")) for path, data in proposals_tree(root).items()}
