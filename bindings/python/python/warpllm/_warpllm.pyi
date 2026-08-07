"""Type stubs for the PyO3 extension module.

Hand-written, and the only file here that is: PyO3 emits no stubs, and the
surface is one class and one function. `tests/test_native_stub.py` compares it
against the module that actually loads, because a stub is inert -- it drifted
once already, still declaring an `echo` and a `serve` that had been removed.
"""

from collections.abc import Awaitable

def version() -> str: ...

class WarpLLMNativeError(Exception): ...

class Client:
    def __init__(self, config_json: str) -> None: ...
    def chat_completions(self, request_json: str) -> str: ...
    def async_chat_completions(self, request_json: str) -> Awaitable[str]: ...
