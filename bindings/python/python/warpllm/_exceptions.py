from __future__ import annotations

import json
from typing import NoReturn


class WarpLLMError(Exception):
    """Base for all warpllm errors.

    `code` is a stable machine-readable slug, one per failure — warpllm's own
    and the provider's live in one flat space, so `code` alone identifies a
    failure. `origin` says which side it came from: `"provider"` means an
    upstream answered and said no, `"gateway"` means the call never got that
    far. Locally raised errors are always `"gateway"`.
    """

    def __init__(
        self, message: str, *, code: str = "error", origin: str = "gateway"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.origin = origin


class InvalidRequestError(WarpLLMError):
    """warpllm rejected the call before sending it.

    Distinct from a provider rejecting the request, which raises
    `APIStatusError` with `code == "provider_invalid_request"` — one is fixed
    by editing the call, the other may just need a different model.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_request")


class APIConnectionError(WarpLLMError):
    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, code="connection_error")
        self.provider = provider


class APIStatusError(WarpLLMError):
    """An upstream provider returned an error.

    Two groups of attributes:

    * What happened — `code`, the normalized taxonomy (`"rate_limited"`,
      `"quota_exceeded"`, `"context_length_exceeded"`, …), and `origin`.
      Branch on `code` rather than on the status: a quota exhaustion and a
      rate limit both arrive as a 429 and are not the same failure.
    * What the provider said — `error_type`, `provider_code`, `request_id`,
      `retry_after_seconds`, and `provider_raw`.

    Nothing here says what to DO about a failure; that is the caller's
    policy. `retry_after_seconds` is the provider's own figure, carrying no
    claim from warpllm that waiting will help.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        error_type: str | None = None,
        retry_after_seconds: int | None = None,
        provider_code: str | None = None,
        request_id: str | None = None,
        provider_raw: str | None = None,
        code: str = "error",
        origin: str = "provider",
    ) -> None:
        super().__init__(message, code=code, origin=origin)
        self.status_code = status_code
        self.provider = provider
        self.error_type = error_type
        self.retry_after_seconds = retry_after_seconds
        self.provider_code = provider_code
        self.request_id = request_id
        self.provider_raw = provider_raw


class AuthenticationError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    """Requests arrived faster than the account may send them.

    Deliberately NOT raised for quota exhaustion — see `QuotaExceededError`.
    """


class QuotaExceededError(APIStatusError):
    """The account is out of credit or past a spend cap.

    A sibling of `RateLimitError`, never a subclass, because the two need
    opposite handling and providers report them under the same 429. Catching
    this as a rate limit is how a billing failure turns into an infinite
    backoff loop, so `except RateLimitError` must not swallow it.
    """


class PermissionDeniedError(APIStatusError):
    """The credential is valid but not entitled to what was asked for."""


def raise_from_wire(raw: str) -> NoReturn:
    """Translates the native layer's wire-format JSON into typed exceptions."""
    try:
        data = json.loads(raw)
    except ValueError:
        raise WarpLLMError(raw) from None

    code = data.get("code", "error")
    message = data.get("message", raw)
    provider = data.get("provider")

    if code == "not_implemented":
        raise NotImplementedError(message)
    if code == "invalid_request":
        raise InvalidRequestError(message)
    if code == "missing_api_key":
        raise AuthenticationError(
            message, provider=provider, code=code, origin="gateway"
        )
    if code == "connection_error":
        raise APIConnectionError(message, provider=provider)
    if data.get("origin") == "provider":
        # Branch on the normalized code, never the raw status. The statuses
        # lie in both directions: a 403 permission failure and a 401 bad key
        # are both credential problems, while one 429 is a rate limit and
        # another is a billing failure.
        raise _BY_CODE.get(code, APIStatusError)(
            message,
            code=code,
            status_code=data.get("status"),
            provider=provider,
            error_type=data.get("error_type"),
            retry_after_seconds=data.get("retry_after_seconds"),
            provider_code=data.get("provider_code"),
            request_id=data.get("request_id"),
            provider_raw=data.get("provider_raw"),
        )
    raise WarpLLMError(message, code=code)


#: Codes that earn an exception class of their own. Anything absent raises
#: `APIStatusError`, which still carries `code` — a new failure kind is never
#: a crash, just a less specific class. Mirrors the Rust `Error` variants,
#: which is why the two lists are the same shape.
_BY_CODE = {
    "authentication": AuthenticationError,
    "permission_denied": PermissionDeniedError,
    "rate_limited": RateLimitError,
    "quota_exceeded": QuotaExceededError,
}
