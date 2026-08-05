"""What the generated types are supposed to say. Checked by `test_typing.py`,
never executed.

Each `# type: ignore[...]` is an assertion that the line IS an error. mypy runs
here with `--warn-unused-ignores`, so a suppression that stops being needed
fails the run -- the same bargain as TypeScript's `@ts-expect-error`, and the
reason these read backwards.
"""

from __future__ import annotations

from warpllm import CreateChatCompletionRequest, WarpLLM


def a_plain_dict_is_a_request(client: WarpLLM) -> None:
    client.chat_completion(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 64,
        }
    )


def an_extension_parameter_is_accepted_at_the_boundary(
    client: WarpLLM,
) -> None:
    client.chat_completion(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "future_provider_parameter": {"enabled": True},
        }
    )


def the_generated_request_is_strict_when_requested() -> None:
    request: CreateChatCompletionRequest = {
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "hi"}],
        "temperture": 0.2,  # type: ignore[typeddict-unknown-key]
    }
    assert request


def the_response_is_typed(client: WarpLLM) -> None:
    completion = client.chat_completion({"model": "m", "messages": []})

    # Reading the response needs no import and no cast...
    content: str | None = completion["choices"][0]["message"]["content"]
    finish: str = completion["choices"][0]["finish_reason"]
    assert content is not None and finish

    # ...and a misspelled field is an error rather than `Any`.
    completion["choicez"]  # type: ignore[typeddict-item]


def a_nullable_field_stays_nullable(client: WarpLLM) -> None:
    """`TopLogprob.bytes` is `Option<Vec<u8>>` in Rust with no
    `skip_serializing_if`, so it really does arrive as `null`.

    It was `list[int]` here until the schema emitter started spelling nullables
    as `anyOf`; the annotation was flatly wrong and nothing said so.
    """
    completion = client.chat_completion({"model": "m", "messages": []})
    logprobs = completion["choices"][0].get("logprobs")
    assert logprobs is not None
    content = logprobs.get("content")
    assert content is not None
    first = content[0]["top_logprobs"][0]["bytes"]
    len(first)  # type: ignore[arg-type]
