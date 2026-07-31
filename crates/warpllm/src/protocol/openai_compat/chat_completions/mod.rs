//! `Api::ChatCompletions` as OpenAI-compatible providers speak it: the wire
//! shapes in [`types`], the HTTP binding in `transport`.
//!
//! The conversions to and from the normalized forms live at
//! `crate::normalized::openai_compat::chat_completions`.

pub mod types;

pub(crate) mod transport;

pub use types::*;
