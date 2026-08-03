import pytest
from pytest_httpserver import HTTPServer
from warpllm import (
    APIStatusError,
    AuthenticationError,
    ChatCompletion,
    InvalidRequestError,
    QuotaExceededError,
    RateLimitError,
    WarpLLM,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def test_sync_openai_happy_path(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = client.chat.completions.create(
        model="openai/gpt-5.6", messages=MESSAGES
    )

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Hello there!"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "openai/gpt-5.6"
    assert completion.usage.total_tokens == 21
    assert completion.service_tier == "default"
    assert completion.system_fingerprint == "fp_44709d6fcb"
    assert completion.usage.prompt_tokens_details.cached_tokens == 3
    assert completion.usage.prompt_tokens_details.cache_write_tokens == 2
    assert completion.usage.completion_tokens_details.reasoning_tokens == 5

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

    completion = await async_client.chat.completions.create(
        model="openai/gpt-5.6", messages=MESSAGES
    )

    assert completion.choices[0].message.content == "Hello there!"
    assert completion.model == "openai/gpt-5.6"
    assert completion.usage.total_tokens == 21


def test_401_raises_authentication_error(
    client: WarpLLM, httpserver: HTTPServer
):
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
            }
        },
        status=401,
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == "openai"
    assert "Incorrect API key" in str(exc_info.value)
    # The normalized taxonomy, and the provider's own spelling beside it.
    assert exc_info.value.code == "authentication"
    assert exc_info.value.error_type == "invalid_request_error"


def test_quota_exhaustion_is_not_raised_as_a_rate_limit(
    client: WarpLLM, httpserver: HTTPServer
):
    """A quota exhaustion arrives as a 429 and reads exactly like a rate
    limit, but no amount of backing off buys credit.

    `except RateLimitError` must NOT catch it — that is how a billing
    failure becomes an infinite retry loop.
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

    with pytest.raises(QuotaExceededError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )
    error = exc_info.value
    assert not isinstance(
        error, RateLimitError
    ), "a backoff loop written against RateLimitError would swallow this"
    assert error.status_code == 429
    assert error.code == "quota_exceeded"
    # The evidence the classification was read from survives.
    assert error.provider_code == "insufficient_quota"


def test_rate_limit_carries_the_providers_retry_after(
    client: WarpLLM, httpserver: HTTPServer
):
    """The provider's own delay and request id reach the caller — both live
    only in headers, so they prove the transport kept them."""
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

    with pytest.raises(RateLimitError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )
    error = exc_info.value
    assert error.code == "rate_limited"
    assert error.retry_after_seconds == 30
    assert error.request_id == "req-abc"


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

    with pytest.raises(APIStatusError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )
    assert exc_info.value.code == "context_length_exceeded"


def test_origin_separates_the_providers_failures_from_warpllms(
    client: WarpLLM, httpserver: HTTPServer
):
    """The two halves of one flat code space. A provider rejecting the
    request and warpllm rejecting it read almost alike, and the remedy is
    not the same — one edits the payload, the other may just need a
    different model."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {"error": {"message": "bad payload", "type": "invalid_request_error"}},
        status=400,
    )

    with pytest.raises(APIStatusError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )
    assert exc_info.value.origin == "provider"
    assert exc_info.value.code == "provider_invalid_request"

    # ...and warpllm's own rejection never left the process.
    with pytest.raises(InvalidRequestError) as local:
        client.chat.completions.create(
            model="mistral/large", messages=MESSAGES
        )
    assert local.value.origin == "gateway"
    assert local.value.code == "invalid_request"


def test_unknown_provider_rejected(client: WarpLLM):
    with pytest.raises(InvalidRequestError, match="no registered model spec"):
        client.chat.completions.create(
            model="mistral/large", messages=MESSAGES
        )


def test_bare_model_rejected(client: WarpLLM):
    with pytest.raises(InvalidRequestError, match="no registered model spec"):
        client.chat.completions.create(model="gpt-5.6", messages=MESSAGES)


def test_missing_key_names_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = WarpLLM()
    with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES
        )


def test_stream_raises_not_implemented(client: WarpLLM):
    with pytest.raises(NotImplementedError, match="streaming"):
        client.chat.completions.create(
            model="openai/gpt-5.6", messages=MESSAGES, stream=True
        )
