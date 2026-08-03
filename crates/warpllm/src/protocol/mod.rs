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
//! The module path is [`crate::types::Api`]'s snake_case name, not the HTTP
//! path: `Api::ChatCompletions` deserializes from `"chat_completions"` and is
//! implemented at `openai_compat::chat_completions`. The URL an API is reached
//! at is a transport detail and lives in that module's `transport.rs` — two
//! protocols may serve the same API at different paths.
//!
//! The names themselves — [`crate::types::Protocol`] and `Api` — are declared
//! in `crate::types`, above this module and `crate::gateway` both, since both
//! layers speak them and neither owns them. What lives here are the protocols
//! those names refer to.
//!
//! Provider-specific logic does NOT live here either: per-model specs are
//! contributed by `crate::registry`, and per-provider conversion deltas belong
//! with the conversions under `crate::gateway`.

pub mod openai_compat;
