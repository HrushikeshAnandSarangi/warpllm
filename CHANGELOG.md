# Changelog

Notable changes to warpllm. One version releases all three packages together —
the crate, the PyPI package, and the npm package share a version number, so a
release note here applies to all of them unless it says otherwise.

Versions follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a breaking change bumps the MINOR number: `0.1.x` and `0.2.x` are
incompatible, and `^0.1` will not upgrade you into one.

## [0.2.0] - 2026-08-08

The first release with a provider registry. 0.1.4 could reach OpenAI; this can
route to any provider on the roster, and it decides at construction which of
them the environment can actually authenticate.

This release rewrites most of the public surface. Everything under "Changed"
and "Removed" is breaking.

### Added

- **Provider registry.** Providers and models live in `specs.yaml`, compiled
  into the binary. Model strings are `provider/model`. The registry **fails
  closed**: a name no entry claims is an error, never a guess at an upstream
  default, so a typo cannot become a live, billed request. There is no
  wildcard — `openai/*` registers a model literally named `*`.
- **Every model declares the API surfaces it serves**, and inherits nothing
  from its provider:

  ```yaml
  openai/gpt-5.6:
    supported_apis:
      - {api: openai_compat_chat_completions}
  ```

  A provider is a host, not a capability — one host commonly serves chat
  completions, embeddings, and moderation from disjoint sets of models — so
  there is nothing at that level to route on. A request for a surface the model
  does not list is refused before the network, rather than discovered as a 404
  upstream. A surface name carries the protocol it is spoken in, which is what
  lets a model one day list `anthropic_messages` beside the entry above.
- **Provider entries are transport only**: `base_url`, `env_api_key`, `models`.
  There is no `protocol:` field — the surfaces a model lists already say which
  wire format is in play, so one host may serve models over different
  protocols.
- **DeepSeek and OpenRouter providers**, alongside OpenAI. Adding an
  OpenAI-compatible provider is a YAML edit and no Rust.
- **Environment-driven provider discovery.** Building a client reads each
  roster provider's `env_api_key` variable once and keeps the providers it can
  authenticate. The set is reported through `tracing` — silent unless the host
  installs a subscriber. A request is admitted only when the roster registers
  the model *and* the client holds a key for the provider serving it.
- **Error taxonomy with provenance.** `Error::origin()` separates a warpllm
  rejection from a provider's, and `Error::code()` is a stable slug for
  bindings. Provider failures carry `ProviderError` with the upstream status,
  `retry_after`, and request id.
- **Errors normalized into OpenAI's vocabulary**, once, in Rust — so a quota
  exhaustion reads the same whichever provider served it. Python and Node raise
  exception classes mirroring the official OpenAI SDK.
- Registry read surface in Rust: `fetch_model`, `ProviderSpec`, `ModelSpec`,
  `Capabilities`, `SupportedApi`, `Api`. Every field is private and there is no
  public constructor, so a spec is read-only outside the crate.
- `JsonClient`, the JSON boundary both native bindings share.
- Quickstart examples for all three languages in `examples/`.

### Changed

- **`chat_completion` is now `chat_completions`** (`chatCompletion` →
  `chatCompletions` in Node). This is the rename most callers will hit.
- **API keys resolve at construction, not per request.** A key exported after a
  client is built is not picked up, and a rotated key needs a new client. Long
  running processes that build one client at startup must restart to pick up a
  rotated key.
- **Rust wire types moved** from `types::openai::chat::completions::*` to
  `protocol::openai_compat::chat_completions::types`. The crate root no longer
  re-exports them with a glob; it names the three types you need to make a call
  and hold its result: `CreateChatCompletionRequest`,
  `ChatCompletionRequestMessage`, `CreateChatCompletionResponse`.
- **Binding types are generated from Rust** rather than hand-written, so the
  three languages cannot drift. Python's `ChatCompletion` is now
  `CreateChatCompletionResponse`; Node re-exports the generated names under
  OpenAI's spellings.
- Unknown request and response fields pass through in both directions rather
  than being dropped, so a provider parameter warpllm does not model still
  reaches it.

### Removed

- **`echo`** from every package. It was a connectivity probe, not API.
- **`WarpLLMError`, `InvalidRequestError`, `NotImplementedError`** in Python and
  Node, replaced by the OpenAI-SDK-shaped hierarchy (`APIError`,
  `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`,
  `NotFoundError`, `ConflictError`, `UnprocessableEntityError`,
  `RateLimitError`, `InternalServerError`, `APIConnectionError`).
- Python's re-exports of response internals (`Choice`, `CompletionUsage`,
  `Annotation`, and the rest). Reading `completion["choices"][0]` names none of
  them, so they are no longer part of the public surface.

### Not in this release

Streaming, retries, failover, load balancing, and caching are still
unimplemented. Supplying API keys through client configuration rather than the
environment is not supported. The OpenAI-compatible HTTP gateway
(`warpllm-server`) is in the repository but is not published.

## [0.1.4] and earlier

Early SDK releases serving OpenAI chat completions only, before the provider
registry existed. See the [release tags](https://github.com/warpllm/warpllm/tags).

[0.2.0]: https://github.com/warpllm/warpllm/compare/v0.1.4...v0.2.0
