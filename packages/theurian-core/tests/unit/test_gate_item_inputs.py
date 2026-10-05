"""The gates read exactly two item-derived inputs, and the accept floors compare those two.

ADR-0032's amendment closes GHSA-v2qg-23fc-7fqp's class by the gates' own input
set: ``status`` through ``may_surface`` and ``sensitivity`` through
``may_disclose``. A floor that compared fewer fields than the gates read would
leave a lowering or readmission the gates then act on, and a third gate input
would leave the floors one field short without any test going red.

Reach: the key is every public module attribute of ``theurian.domain.enums``
whose name starts ``may_`` and that is callable with ``__module__`` equal to
that module once unwrapped through ``__wrapped__`` (``functools.cache``,
``lru_cache``, ``wraps``). A plain function and a cache-wrapped one both count;
a ``may_*`` name imported into the module from elsewhere does not.

Blind spot: a gate that decides inline, outside ``domain/enums.py``, or whose
name does not start ``may_``, one bound through ``functools.partial`` (its
unwrapped ``__module__`` is ``functools``, so it is excluded), or one that reads an
item field through ``getattr`` or a helper it calls, is invisible here; ``test_gate_call_sites.py``
is what holds the call sites. The two floors are read by attribute access inside
their own function bodies only.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import typing
from collections.abc import Callable
from typing import Any, Final

import pytest

from theurian.application import item_labels
from theurian.domain import enums
from theurian.domain.enums import KnowledgeStatus, Sensitivity

pytestmark = pytest.mark.unit

#: The item field each gate decides on, by the gate's name.
_GATES: Final[dict[str, type]] = {
    "may_surface": KnowledgeStatus,
    "may_disclose": Sensitivity,
}


def _public_gates() -> dict[str, Callable[..., Any]]:
    """Every public callable named ``may_*`` that ``domain/enums.py`` defines, wrapped or not."""
    return {
        name: member
        for name, member in vars(enums).items()
        if name.startswith("may_")
        and callable(member)
        and inspect.unwrap(member).__module__ == enums.__name__
    }


def _label_fields_read(function: Callable[..., Any]) -> set[str]:
    """The ``ItemLabels`` fields a function reads as attributes, by its own source."""
    fields = {field.name for field in dataclasses.fields(item_labels.ItemLabels)}
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in fields
    }


def test_the_gates_are_exactly_may_surface_and_may_disclose() -> None:
    gates = _public_gates()

    assert set(gates) == set(_GATES), (
        f"a gate beyond {sorted(_GATES)} landed: {sorted(set(gates) - set(_GATES))}; the "
        "accept floors and ADR-0032's closure premise must cover its input"
    )


@pytest.mark.parametrize(("name", "gated_type"), sorted(_GATES.items()))
def test_each_gate_decides_on_one_item_field_type_in_its_first_parameter(
    name: str, gated_type: type
) -> None:
    function = _public_gates()[name]
    first = next(iter(inspect.signature(function).parameters.values()))

    hints = typing.get_type_hints(function)

    assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert hints[first.name] is gated_type


def test_the_accept_floors_compare_exactly_the_two_fields_the_gates_decide_on() -> None:
    lowered = _label_fields_read(item_labels.lowered_sensitivities)
    readmitted = _label_fields_read(item_labels.readmitted_items)

    assert lowered == {"sensitivity"}
    assert readmitted == {"status"}
