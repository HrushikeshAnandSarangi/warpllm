from __future__ import annotations

import json
from typing import Any, NoReturn


class APIError(Exception):
    """The errors warpllm raises, matching the shape and the dispatch of the
    official OpenAI SDK.

    warpllm's own failure taxonomy is not here and never crosses from Rust.
    It does its work before this point: warpllm knows a quota exhaustion from
    a rate limit, and knows DeepSeek's 402 for an exhausted balance means what
    OpenAI calls 429 `insufficient_quota` -- so what reaches you is what
    OpenAI would have said, whichever provider served the request::

        try:
            client.chat_completion({"model": "deepseek/deepseek-chat", ...})
        except RateLimitError as e:
            # A 429 covers two failures that need opposite handling. Waiting
            # fixes one of them; only `code` says which.
            if e.code == "insufficient_quota":
                top_up()
            else:
                back_off()

    Attributes:
        message: What went wrong.
        code: The failure's own slug, free-form per OpenAI. The only field
            that separates failures sharing a status.
        param: Which request parameter was at fault. warpllm does not model
            this yet, so it is always `None`.
        type: OpenAI error family, e.g. `"invalid_request_error"`.
        request_id: The upstream's request id, when it sent one.
        body: The error object as it came off the wire.
    """

    def __init__(
        self,
        message: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        body = body or {}
        self.message = message
        self.headers = headers or {}
        self.body = body
        self.code: str | None = body.get("code")
        self.param: str | None = body.get("param")
        self.type: str | None = body.get("type")
        self.request_id: str | None = self.headers.get("x-request-id")


class APIConnectionError(APIError):
    """warpllm never reached the provider, so there is no status and no body."""


class APIStatusError(APIError):
    """A failure an upstream answered with, carrying its status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, body=body, headers=headers)
        self.status_code = status_code


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class PermissionDeniedError(APIStatusError):
    pass


class NotFoundError(APIStatusError):
    pass


class ConflictError(APIStatusError):
    pass


class UnprocessableEntityError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    """A 429 -- and NOT only a rate limit.

    An exhausted quota arrives here too, because OpenAI reports both under
    one status and one class. Backing off clears the first and never clears
    the second, so read `code` before retrying: `insufficient_quota` means
    somebody has to pay.
    """


class InternalServerError(APIStatusError):
    pass


#: Rust has already interpreted the status. This table only turns its stable
#: discriminant into the corresponding Python class object.
_BY_CLASS = {
    "api_status": APIStatusError,
    "bad_request": BadRequestError,
    "authentication": AuthenticationError,
    "permission_denied": PermissionDeniedError,
    "not_found": NotFoundError,
    "conflict": ConflictError,
    "unprocessable_entity": UnprocessableEntityError,
    "rate_limit": RateLimitError,
    "internal_server": InternalServerError,
}


def raise_from_wire(raw: str) -> NoReturn:
    """Rebuilds the error from the native layer's JSON message.

    Nothing is interpreted here. Rust already decided what the failure is and
    what OpenAI calls it; this only instantiates the class named by the wire
    discriminant.
    """
    try:
        wire = json.loads(raw)
    except ValueError:
        # Not our JSON -- surface it whole rather than inventing a shape.
        raise APIError(raw) from None

    body = wire.get("error") or {}
    headers = wire.get("headers") or {}
    message = body.get("message", raw)
    if wire.get("class") == "api_connection":
        raise APIConnectionError(message, body=body, headers=headers) from None
    status = wire.get("status")
    Class = _BY_CLASS.get(wire.get("class"), APIStatusError)
    if status is None:
        raise APIError(message, body=body, headers=headers) from None
    raise Class(
        message, status_code=status, body=body, headers=headers
    ) from None
