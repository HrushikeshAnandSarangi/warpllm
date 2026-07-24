//! Protocols: how each wire dialect is spoken. A protocol owns the
//! four conversions between its wire shapes and the normalized forms —
//! ingest_request / render_request / ingest_response / render_response —
//! plus the HTTP transport and error envelope. The module path mirrors the
//! HTTP path: `POST /chat/completions` lives at
//! `openai_compat::chat::completions`.
//!
//! Provider-specific logic does NOT live here: per-model specs are
//! contributed by `crate::providers` and consulted by the client around
//! these conversions.

pub(crate) mod openai_compat;
