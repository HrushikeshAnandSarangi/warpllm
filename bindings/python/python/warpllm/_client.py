from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping, cast

from warpllm._warpllm import Client as _NativeClient
from warpllm._warpllm import WarpLLMNativeError

from ._exceptions import raise_from_wire

if TYPE_CHECKING:
    from .types import CreateChatCompletionResponse


def _native_client(base_url: str | None, timeout: int | None) -> _NativeClient:
    config = {"base_url": base_url, "timeout_secs": timeout}
    try:
        return _NativeClient(
            json.dumps({k: v for k, v in config.items() if v is not None})
        )
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


class WarpLLM:
    """Synchronous client. Model strings are `provider/model`, e.g.
    `"openai/gpt-5.6"`. API keys come from the environment
    (`OPENAI_API_KEY`); a provider's key is only required when a request
    targets that provider.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._native = _native_client(base_url, timeout)

    def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse:
        """One method, mirroring Rust's `client.chat_completions(request)`.

        The request crosses verbatim -- its fields are Rust's, so nothing
        here renames them and nothing here has to learn a field warpllm
        gains. The response comes back as Rust serialized it: Rust has
        already parsed and validated it, and re-hydrating it into Python
        objects would re-do that work to hand back the same fields under the
        same names.

        `warpllm.types.CreateChatCompletionRequest` is available when callers
        want strict authoring help. This boundary accepts any mapping because
        Rust deliberately forwards fields it does not model.
        """
        try:
            raw = self._native.chat_completions(json.dumps(dict(request)))
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        return cast("CreateChatCompletionResponse", json.loads(raw))


class AsyncWarpLLM:
    """Async client; `await client.chat_completions(...)`."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._native = _native_client(base_url, timeout)

    async def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse:
        try:
            raw = await self._native.async_chat_completions(
                json.dumps(dict(request))
            )
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        return cast("CreateChatCompletionResponse", json.loads(raw))
