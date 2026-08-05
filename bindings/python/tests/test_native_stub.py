"""The stub for the native module must describe the native module.

`_warpllm.pyi` is the one hand-written type file left in this package, and a
stub is inert: nothing imports it, nothing runs it, and a name that goes stale
in it is invisible. It went stale exactly that way once, still declaring `echo`
and `serve` after both were removed from `src/lib.rs`. These compare the stub
against the extension that actually loads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import warpllm._warpllm as native

STUB = ast.parse(
    (Path(native.__file__).parent / "_warpllm.pyi").read_text(),
)


def _declared() -> set[str]:
    return {
        node.name
        for node in STUB.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _exported() -> set[str]:
    return {name for name in dir(native) if not name.startswith("_")}


def test_the_stub_declares_nothing_the_module_lacks():
    assert not _declared() - _exported()


def test_the_stub_declares_everything_the_module_exports():
    assert not _exported() - _declared()


def test_client_methods_match():
    stub_methods = {
        node.name
        for cls in STUB.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Client"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    real = {name for name in dir(native.Client) if not name.startswith("__")}
    assert stub_methods - {"__init__"} == real
