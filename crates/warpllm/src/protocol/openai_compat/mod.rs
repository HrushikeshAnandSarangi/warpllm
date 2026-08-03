//! The OpenAI-compatible protocol: every provider whose spec selects
//! `Protocol::OpenAiCompat` is spoken through here, parameterized by the
//! provider's name, base URL, and key.
//!
//! `openai_compat` is warpllm's OpenAI-*compatible* protocol: a permissive
//! superset (unknown-field passthrough) of the OpenAI request, which many
//! providers speak. The authoritative *OpenAI* shape is defined upstream in
//! <https://github.com/openai/openai-openapi>; we track it and contribute
//! changes there rather than fork it. A provider with its own wire format gets
//! a sibling protocol module here (e.g. `anthropic::messages`), and the
//! conversions that translate between the two via the gateway request live
//! under `crate::gateway`.
//!
//! Providers that speak this protocol but diverge in places — a response field
//! carrying meaning OpenAI has no name for, a parameter spelled differently —
//! do NOT get shapes of their own. The divergence is handled in the
//! conversions under `crate::gateway`; the shapes here stay the protocol's.
//!
//! SHAPES AND TRANSPORT ONLY. Turning a provider's error body into warpllm's
//! canonical failure form is an ingest conversion, and it lives with every
//! other one at `crate::gateway::openai_compat::error` — the same rule that
//! keeps `transport` handing a non-2xx back as data rather than deciding what
//! it means.

pub mod chat_completions;
