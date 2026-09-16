"""A planted probe for the async sweep's positive control (#378, PR #723).

This module lives only on the ``probe/sweep-positive-control`` branch and is
imported by nothing and tested by nothing, deliberately: the sweep's first
real ``workflow_dispatch`` must demonstrate that a mutation the suite does
not hold is FILED, and zero findings only count against a planted positive.
The two expressions below are the generator's own operator set — a boolean
constant and an ``or`` — so the night steered onto this file by its computed
date must report SURVIVED for what nothing kills, and file it.
"""

from __future__ import annotations


def probe_default(flag: bool) -> bool:
    """Return the flag, with a permissive default nothing asserts."""
    fallback = True
    return flag or fallback
