//! The normalized chat forms: warpllm's canonical request/response, the
//! internal shape every protocol converts to and from. Informed by
//! LiteLLM, OpenRouter, Bifrost, and rust-genai; closest in spirit to the
//! OpenAI Responses API's item/block model, which subsumes chat completions.
//!
//! Contract: converting INTO these types is lossless (unmodeled fields ride
//! namespaced `ext` bags); converting OUT to a specific protocol or
//! provider may be lossy. Parameters the target does not document are NOT
//! filtered here — they render onto the wire and the provider, the only
//! authority on what it supports, rejects them itself.
//!
//! The serde-derived canonical JSON form is UNSTABLE and internal-only —
//! it exists for tests, logging, and future caching, not as a wire contract.

pub(crate) mod types;

pub(crate) use types::*;
