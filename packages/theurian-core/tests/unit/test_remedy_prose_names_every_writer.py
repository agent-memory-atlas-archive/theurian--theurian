"""GHSA-fjqq-grr7-53cc: each site names every landed writer and the accept route.

Each site states every phrase in its stated tuple and none in its retired one:
not one migration, and not ``migrate apply`` for the hand-authored redraft. The
migration format's site also holds the refused-field definition and retires "the
refused fields being the reported row's". Exact phrases only; a reworded regression
passes. The migration format's site is its permissive-moves section: the
draft-side refusal of ``changeSensitivity`` and ``restoreItem``, which its
Operations section describes, still names ``migrate apply``.
"""

from __future__ import annotations

import pytest
from threat_model_claims import prose
from write_lock_claims import REPO_ROOT

pytestmark = pytest.mark.unit

_SITES = {
    "migrations.md": (
        "docs/protocol/migrations.md", "### Permissive moves are reported, not refused", "\n### ",
        ("lists, in replay order, every landed migration that writes a refused field and "
         "replays after this proposal's migration",
         "into this proposal's migration file in <proposal dir>/",
         "the refused fields being each field of a report row the proposal would introduce and "
         "each field it writes that a landed migration replaying after it leaves looser"),
        ("landed migration or migrations the error names", "apply it with theurian migrate apply",
         "the refused fields being the reported row's"),
    ),
    "T-28": (
        "docs/security/threat-model.md", "#### T-28 ", "\n#### ",
        ("names in dependsOn every landed migration that writes a refused field and replays "
         "after the proposal's",
         "otherwise this proposal's own migration file, edited and accepted again"),
        ("replays after the landed one", "the remedy is a new migration"),
    ),
    "ADR-0032": (
        "docs/adr/0032-the-write-intent-mcp-tool-surface.md",
        "**Amended by GHSA-fjqq-grr7-53cc", "\n## ",
        ("names in dependsOn every landed migration that writes a refused field and replays "
         "after the proposal's",
         "the hand-authored route goes back through theurian propose accept"),
        ("the error names", "the landed one"),
    ),
    "propose.md": (
        "plugins/claude-code/commands/propose.md", "", "\0",
        ("replays after every landed migration the remedy's dependsOn lists",
         "edited into its own migration file and, once the user agrees, accepted again"),
        ("replays after the landed one", "authored by hand and applied with"),
    ),
    "--help": (
        "packages/theurian-core/src/theurian/cli/propose_commands.py",
        "**What is checked before anything moves**", '"""',
        ("names in dependsOn every landed migration that writes a refused field and replays "
         "after this proposal's",
         "edited into this proposal's own migration file and the proposal accepted again"),
        ("applied with theurian migrate apply", "after the landed one"),
    ),
}  # fmt: skip


@pytest.mark.parametrize("site", _SITES)
def test_a_remedy_site_names_every_landed_writer_and_not_one(site: str) -> None:
    path, start, end, stated, retired = _SITES[site]
    raw = (REPO_ROOT / path).read_text(encoding="utf-8").replace("\n> ", "\n")
    text = prose(raw[raw.index(start) :].split(end, 1)[0])

    assert [phrase for phrase in stated if prose(phrase) not in text] == []
    assert [phrase for phrase in retired if prose(phrase) in text] == []
