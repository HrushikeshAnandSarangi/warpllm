//! DeepSeek's divergences from `openai_compat`.
//!
//! One module per thing it does differently, so a second divergence lands
//! beside the first rather than widening it. There is nothing here but those
//! modules: a provider states its DELTA, never a restatement of the protocol.
//!
//! <https://api-docs.deepseek.com> (checked 2026-08-04).

pub(crate) mod error;
