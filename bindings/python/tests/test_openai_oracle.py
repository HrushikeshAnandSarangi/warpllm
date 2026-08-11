"""Drift alarm: warpllm's reply shapes measured against OpenAI's own.

`openai` is a dev dependency and an ORACLE, never a contract. Nothing here is
re-exported and the wheel never ships it -- warpllm's wire types are its own,
and this only asserts they still name everything the vendor names. When OpenAI
adds a field, that field still reaches callers through the Rust
`unknown_fields` catch-all; it reaches them untyped, and this test is what says
so out loud instead of leaving it to be noticed in someone's stream.

It walks BOTH reply shapes to the leaves rather than checking a hand-listed few
levels. The pairing is structural: for each field the two sides agree on, the
models nested inside the upstream annotation are matched against the TypedDicts
nested inside ours, and the walk recurses. So a shape nobody thought to list --
`CompletionUsage`, the moderation results, a tool call's `function` -- is
covered the day it appears, and a pair that cannot be matched up is reported
rather than skipped in silence.

The other half of the question -- does everything OpenAI can emit FIT warpllm's
shape, nulls included -- is asked in TypeScript, where assignability answers it
directly at every level in one line: `bindings/node/__test__/openai-oracle.ts`.
Neither half has an exception today, and a new one belongs written down at the
check.
"""

from __future__ import annotations

from typing import Any, get_args, is_typeddict

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel
from typing_extensions import get_type_hints
from warpllm._generated.response import CreateChatCompletionResponse
from warpllm._generated.stream_response import (
    CreateChatCompletionStreamResponse,
    StreamChoice,
)

# Where the walk starts. Everything else is reached from here.
ROOTS: list[tuple[type[BaseModel], type[Any]]] = [
    (ChatCompletion, CreateChatCompletionResponse),
    (ChatCompletionChunk, CreateChatCompletionStreamResponse),
]

# Fields OpenAI models that warpllm deliberately does not, keyed by the
# upstream model's name. Empty, and a deletion from it is a decision: an entry
# here means callers reach that field through `unknown_fields` with no type to
# guide them.
UNMODELLED: dict[str, set[str]] = {}


def models_in(annotation: Any) -> list[type[BaseModel]]:
    """Every pydantic model an annotation reaches, in declaration order.

    `Optional[Moderation]`, `list[ToolCall]` and `Results | Error` all answer
    with the models inside them, so a field's shape is found the same way no
    matter how it is wrapped.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for arg in get_args(annotation) for model in models_in(arg)]


def typed_dicts_in(annotation: Any) -> list[type[Any]]:
    """The same, for warpllm's generated side."""
    if is_typeddict(annotation):
        return [annotation]
    return [td for arg in get_args(annotation) for td in typed_dicts_in(arg)]


class Walk:
    def __init__(self) -> None:
        self.checked: list[tuple[str, str, set[str]]] = []
        self.unpaired: list[str] = []
        self.seen: set[tuple[str, str]] = set()

    def visit(self, upstream: type[BaseModel], ours: type[Any]) -> None:
        if (upstream.__name__, ours.__name__) in self.seen:
            return
        self.seen.add((upstream.__name__, ours.__name__))

        # Resolved, not `__annotations__`: the generated modules use
        # `from __future__ import annotations`, so the raw values are strings.
        hints = get_type_hints(ours)
        self.checked.append(
            (
                upstream.__name__,
                ours.__name__,
                set(upstream.model_fields) - set(hints),
            )
        )

        for name, field in upstream.model_fields.items():
            if name not in hints:
                continue
            theirs, mine = models_in(field.annotation), typed_dicts_in(
                hints[name]
            )
            if not theirs:
                continue
            if len(theirs) != len(mine):
                self.unpaired.append(
                    f"{upstream.__name__}.{name}: {len(theirs)} upstream "
                    f"model(s) against {len(mine)} warpllm TypedDict(s)"
                )
                continue
            for nested_upstream, nested_ours in zip(theirs, mine):
                self.visit(nested_upstream, nested_ours)


def walk_both_shapes() -> Walk:
    walk = Walk()
    for upstream, ours in ROOTS:
        walk.visit(upstream, ours)
    return walk


def test_every_field_openai_models_is_named_by_warpllm() -> None:
    walk = walk_both_shapes()

    for upstream, ours, missing in walk.checked:
        assert missing == UNMODELLED.get(upstream, set()), (
            f"{ours} does not name {sorted(missing)}, which {upstream} models. "
            "Add the field to the Rust type in crates/warpllm/src/protocol/"
            "openai_compat/chat_completions/types.rs and regenerate, or record "
            "it in UNMODELLED."
        )


def test_the_walk_reaches_the_leaves() -> None:
    """A walk that silently stopped at the roots would pass the check above.

    The floor is deliberately well under the real count: this is here to catch
    a walk that broke, not to be rewritten whenever OpenAI adds an object.
    """
    walk = walk_both_shapes()
    pairs = {ours for _, ours, _ in walk.checked}

    assert (
        not walk.unpaired
    ), f"shapes that could not be matched up: {walk.unpaired}"
    assert len(walk.checked) >= 15, sorted(pairs)
    # Named because they are the leaves furthest from either root, and reaching
    # them means the recursion went through a list, a union and an optional.
    for leaf in (
        "TopLogprob",
        "ModerationResultBody",
        "CompletionTokensDetails",
    ):
        assert leaf in pairs, f"{leaf} was never reached: {sorted(pairs)}"


def test_the_oracle_would_notice_a_missing_field() -> None:
    """...and the comparison itself has to be capable of failing."""
    assert set(ChatCompletionChunk.model_fields) >= {"id", "choices", "usage"}
    assert set(ChatCompletionChunk.model_fields) - set(
        get_type_hints(StreamChoice)
    ), "comparing mismatched shapes must fail, or the check proves nothing"
