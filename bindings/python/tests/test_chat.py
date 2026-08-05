import pytest
from pytest_httpserver import HTTPServer
from warpllm import WarpLLM, WarpLLMError

MESSAGES = [{"role": "user", "content": "hi"}]


def request(model: str = "openai/gpt-5.6", **extra) -> dict:
    return {"model": model, "messages": MESSAGES, **extra}


def test_sync_openai_happy_path(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = client.chat_completion(request())

    assert completion["choices"][0]["message"]["content"] == "Hello there!"
    assert completion["choices"][0]["finish_reason"] == "stop"
    assert completion["model"] == "openai/gpt-5.6"
    assert completion["usage"]["total_tokens"] == 21
    assert completion["service_tier"] == "default"
    assert completion["system_fingerprint"] == "fp_44709d6fcb"
    assert completion["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert (
        completion["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 2
    )
    assert (
        completion["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == 5
    )

    sent = httpserver.log[0][0].get_json()
    assert sent["model"] == "gpt-5.6"  # provider prefix stripped outbound
    assert sent["messages"] == MESSAGES


async def test_async_openai_happy_path(
    async_client, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = await async_client.chat_completion(request())

    assert completion["choices"][0]["message"]["content"] == "Hello there!"
    assert completion["model"] == "openai/gpt-5.6"
    assert completion["usage"]["total_tokens"] == 21


def test_the_response_is_not_narrowed_on_the_way_through(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    """What Rust serialized is what the caller gets.

    The wrapper used to re-hydrate the body into dataclasses, which meant a
    field warpllm learned to pass through was still dropped here until a
    Python class gained it too. Handing back the parsed body makes that
    class of bug unreachable rather than merely tested for.
    """
    openai_completion_body["some_field_python_never_heard_of"] = {
        "nested": [1]
    }
    httpserver.expect_request("/chat/completions").respond_with_json(
        openai_completion_body
    )

    completion = client.chat_completion(request())

    assert completion["some_field_python_never_heard_of"] == {"nested": [1]}


def test_401_reports_authentication(client: WarpLLM, httpserver: HTTPServer):
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
        status=401,
    )

    with pytest.raises(WarpLLMError) as exc_info:
        client.chat_completion(request())
    assert exc_info.value.status_code == 401
    assert "Incorrect API key" in str(exc_info.value)
    # The provider's own slug reaches the caller, not warpllm's spelling
    # of it -- warpllm would have called this one `authentication`.
    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.type == "invalid_request_error"


def test_quota_exhaustion_is_not_reported_as_a_rate_limit(
    client: WarpLLM, httpserver: HTTPServer
):
    """A quota exhaustion arrives as a 429 and reads exactly like a rate
    limit, but no amount of backing off buys credit.

    A retry loop keyed on `code == "rate_limited"` must not fire here --
    that is how a billing failure becomes an infinite retry loop.
    """
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "You exceeded your current quota",
                "type": "invalid_request_error",
                "code": "insufficient_quota",
            }
        },
        status=429,
    )

    with pytest.raises(WarpLLMError) as exc_info:
        client.chat_completion(request())
    error = exc_info.value
    assert error.code == "insufficient_quota"
    assert (
        error.code != "rate_limit_exceeded"
    ), "a backoff loop would swallow this"
    assert error.status_code == 429


def test_rate_limit_carries_the_providers_request_id(
    client: WarpLLM, httpserver: HTTPServer
):
    """The upstream's request id reaches the caller. It lives only in a
    header, so it proves the transport kept it."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Rate limit reached",
                "type": "rate_limit_error",
            }
        },
        status=429,
        headers={"Retry-After": "30", "x-request-id": "req-abc"},
    )

    with pytest.raises(WarpLLMError) as exc_info:
        client.chat_completion(request())
    assert exc_info.value.type == "rate_limit_error"
    assert exc_info.value.request_id == "req-abc"


def test_context_overflow_is_classified(
    client: WarpLLM, httpserver: HTTPServer
):
    """A context overflow must not read as a plain bad request: the remedy
    is a shorter prompt or a bigger model, not a corrected payload."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "maximum context length is 8192 tokens",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
        status=400,
    )

    with pytest.raises(WarpLLMError) as exc_info:
        client.chat_completion(request())
    assert exc_info.value.code == "context_length_exceeded"


def test_code_separates_the_providers_rejection_from_warpllms(
    client: WarpLLM, httpserver: HTTPServer
):
    """A provider rejecting the request and warpllm rejecting it read
    almost alike -- both 400, both `invalid_request_error` -- and the
    remedy is not the same: one edits the payload, the other may just need
    a different model. `code` is what tells them apart, since `origin` is
    warpllm's own vocabulary and stays in Rust."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {"error": {"message": "bad payload", "type": "invalid_request_error"}},
        status=400,
    )

    with pytest.raises(WarpLLMError) as upstream:
        client.chat_completion(request())

    # ...and warpllm's own rejection never left the process.
    with pytest.raises(WarpLLMError) as local:
        client.chat_completion(request(model="mistral/large"))

    assert upstream.value.type == local.value.type == "invalid_request_error"
    assert upstream.value.code == "provider_invalid_request"
    assert local.value.code == "invalid_request"


def test_unknown_provider_rejected(client: WarpLLM):
    with pytest.raises(WarpLLMError, match="no registered model spec"):
        client.chat_completion(request(model="mistral/large"))


def test_bare_model_rejected(client: WarpLLM):
    with pytest.raises(WarpLLMError, match="no registered model spec"):
        client.chat_completion(request(model="gpt-5.6"))


def test_missing_key_names_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = WarpLLM()
    with pytest.raises(WarpLLMError, match="OPENAI_API_KEY") as exc_info:
        client.chat_completion(request())
    assert exc_info.value.code == "missing_api_key"


def test_stream_reports_not_implemented(client: WarpLLM):
    with pytest.raises(WarpLLMError, match="streaming") as exc_info:
        client.chat_completion(request(stream=True))
    assert exc_info.value.code == "not_implemented"
