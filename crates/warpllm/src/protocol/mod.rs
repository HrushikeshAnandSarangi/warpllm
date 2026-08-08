//! Protocols: the wire shapes each protocol defines, and the transport that
//! puts them on the wire. A protocol owns its request/response types, its HTTP
//! binding (URL, auth scheme), and its error envelope.
//!
//! The conversions between these shapes and the gateway forms do NOT live
//! here — they live in `crate::gateway`, one child module per protocol,
//! because a conversion is a statement about the canonical model as much as
//! about the wire. That makes this module a leaf: it depends on nothing but
//! `error` and `http`, and `gateway` depends on it.
//!
//! The module path spells out [`crate::types::Api`]'s own name, not the HTTP
//! path: `Api::OpenAiCompatChatCompletions` deserializes from
//! `"openai_compat_chat_completions"` and is implemented at
//! `openai_compat::chat_completions`, protocol segment and all. The URL an API
//! is reached at is a transport detail and lives in that module's
//! `transport.rs` — two protocols may serve the same API at different paths.
//!
//! [`crate::types::Api`] itself is declared in `crate::types`, above this
//! module and `crate::gateway` both, since both layers speak it and neither
//! owns it. What lives here are the protocols its names refer to.
//!
//! Provider-specific logic does NOT live here either: per-model specs are
//! contributed by `crate::registry`, and per-provider conversion deltas belong
//! with the conversions under `crate::gateway`.

pub mod openai_compat;
