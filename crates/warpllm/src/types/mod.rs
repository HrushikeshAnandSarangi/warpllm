//! Wire shapes, grouped by shape family then endpoint — pure serde data,
//! zero I/O. The module path mirrors the HTTP path: the `chat.completion`
//! types live at `openai_compat::chat::completions`.
//!
//! `openai_compat` is warpllm's OpenAI-*compatible* shape: a permissive
//! superset (unknown-field passthrough) of the OpenAI request, which many
//! providers speak. The authoritative *OpenAI* shape is defined upstream in
//! <https://github.com/openai/openai-openapi>; we track it and contribute
//! changes there rather than fork it. A provider with its own wire format
//! gets a sibling family here (e.g. `anthropic::messages`) and its impl
//! under `crate::providers` translates via the normalized request.

pub mod openai_compat;

/// Deprecated alias preserving the pre-rename `types::openai` module path.
#[deprecated(note = "renamed to `types::openai_compat`")]
pub use openai_compat as openai;
