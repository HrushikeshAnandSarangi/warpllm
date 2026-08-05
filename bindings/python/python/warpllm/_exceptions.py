from __future__ import annotations

import json
from typing import Any, NoReturn


class WarpLLMError(Exception):
    """Every failure warpllm reports, carrying what the official OpenAI SDK
    carries on `APIError` -- and nothing else.

    warpllm's own taxonomy (its `code` slugs, `origin`, the provider
    evidence) deliberately stays in Rust. You program against this class
    directly, so an attribute named here is one warpllm owes compatibility
    on, and that taxonomy is not settled enough to promise.

    Branch on `code`, never on `status_code`. The statuses lie in both
    directions: a 403 permission failure and a 401 bad key are both
    credential problems, while one 429 is a rate limit and another is a
    billing failure that no amount of backing off will clear -- OpenAI
    spells that second one `insufficient_quota`::

        try:
            client.chat_completion({"model": "openai/gpt-5.6", "messages": m})
        except WarpLLMError as e:
            if e.code == "insufficient_quota":
                top_up()

    Attributes:
        status_code: HTTP status of the response that caused the error.
        type: OpenAI error family, e.g. `"invalid_request_error"`.
        code: The failure's own slug -- the provider's when an upstream
            named it, so a quota exhaustion stays `insufficient_quota`.
            Free-form, per OpenAI.
        param: Which request parameter was at fault. warpllm does not model
            this yet, so it is always `None`.
        request_id: The upstream's request id, when it sent one.
    """

    def __init__(
        self, message: str, wire: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        wire = wire or {}
        error = wire.get("error") or {}
        self.message = message
        self.status_code: int | None = wire.get("status")
        self.request_id: str | None = wire.get("request_id")
        self.type: str | None = error.get("type")
        self.code: str | None = error.get("code")
        self.param: str | None = error.get("param")


def raise_from_wire(raw: str) -> NoReturn:
    """Turns the native layer's wire-format JSON into a `WarpLLMError`.

    The wire form is shaped like the arguments the OpenAI SDK builds an
    `APIError` from: the error object exactly as OpenAI spells it, beside
    what an SDK would otherwise read off the HTTP response.
    """
    try:
        wire = json.loads(raw)
    except ValueError:
        # Not our JSON -- surface it whole rather than inventing a shape.
        raise WarpLLMError(raw) from None
    raise WarpLLMError(
        (wire.get("error") or {}).get("message", raw), wire
    ) from None
