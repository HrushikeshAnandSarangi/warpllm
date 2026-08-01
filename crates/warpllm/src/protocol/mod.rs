//! Protocols: the wire shapes each dialect defines, and the transport that
//! puts them on the wire. A protocol owns its request/response types, its HTTP
//! binding (URL, auth scheme), and its error envelope.
//!
//! The conversions between these shapes and the normalized forms do NOT live
//! here — they live in `crate::normalized`, one child module per protocol,
//! because a conversion is a statement about the normalized model as much as
//! about the wire. That makes this module a leaf: it depends on nothing but
//! `error` and `http`, and `normalized` depends on it.
//!
//! The module path is [`Api`]'s snake_case name, not the HTTP path:
//! `Api::ChatCompletions` deserializes from `"chat_completions"` and is
//! implemented at `openai_compat::chat_completions`. The URL an API is reached
//! at is a transport detail and lives in that module's `transport.rs` — two
//! protocols may serve the same API at different paths.
//!
//! Provider-specific logic does NOT live here either: per-model specs are
//! contributed by `crate::registry`, and per-provider conversion deltas belong
//! with the conversions under `crate::normalized`. [`Protocol`] is the one pointing
//! the other way — the registry names a wire format, so the name is declared
//! here with the modules that implement it.

pub mod openai_compat;

/// How a provider's wire protocol is spoken. One variant per wire format,
/// not per provider — the exhaustive `match` in `client.rs` stays small
/// forever while adding a model is one entry in the registry YAML. A provider
/// that diverges from its dialect is handled in that dialect's conversions,
/// never with a variant of its own.
///
/// `non_exhaustive` because this enum exists to grow: a new wire format is a
/// normal addition, and without it every downstream `match` would break on a
/// release that adds one. Inside warpllm the attribute has no effect, so the
/// crate's own matches stay exhaustive and a new variant still fails to
/// compile until every arm handles it.
///
/// The `rename` is spelled out rather than derived: `rename_all =
/// "snake_case"` reads the camel hump in `OpenAi` and produces
/// `open_ai_compat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[non_exhaustive]
pub enum Protocol {
    /// The OpenAI-compatible chat completions wire format
    /// ([`openai_compat`]).
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
}

impl Protocol {
    /// This dialect's name as a string: the spelling the registry YAML uses,
    /// and the key its passthrough fields are filed under in the normalized
    /// `ext` bags.
    ///
    /// One source of truth for that string, so the YAML vocabulary and the ext
    /// namespace cannot drift apart — `protocol_name_matches_serde` pins them
    /// together, which matters because the `serde(rename)` above is hand-written.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Protocol::OpenAiCompat => "openai_compat",
        }
    }

    /// The APIs this wire format defines.
    ///
    /// A registry entry naming an API its protocol does not serve is a
    /// protocol mismatch, which is what `registry::lint` reads this for.
    /// Widening a protocol's surface is an edit here, beside the module that
    /// implements it — not in the roster.
    ///
    /// While `OpenAiCompat` is the only protocol and serves every [`Api`],
    /// that lint cannot fail; it earns its keep the moment a protocol lands
    /// that serves a subset, which is why it is written now rather than
    /// retrofitted then.
    ///
    /// `cfg(test)` because that lint is its only reader, and the lint itself
    /// is test-only. Nothing on a request path consults this: a spec carries
    /// its OWN api list, already checked against this one.
    #[cfg(test)]
    pub(crate) fn apis(self) -> &'static [Api] {
        match self {
            Protocol::OpenAiCompat => &[
                Api::ChatCompletions,
                Api::ChatCompletionsStream,
                Api::Responses,
            ],
        }
    }
}

/// One API surface a provider can serve, as named in a registry provider's
/// `supported_apis`.
///
/// A capability, not a URL: two protocols can serve the same API at different
/// paths, so the path belongs to the protocol module that implements it. Being
/// an enum is what makes a misspelling fail to LOAD rather than 404 against a
/// live provider at request time.
///
/// The snake_case name is also the module name that implements the API, so
/// `chat_completions` is reached at `openai_compat::chat_completions` and the
/// two cannot drift by a rename.
///
/// `non_exhaustive` for the same reason as [`Protocol`]: this exists to grow
/// as warpllm implements more of each provider's surface, and without it every
/// downstream `match` would break on a release that adds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Api {
    /// Chat completions: one request, one whole reply. The only surface
    /// warpllm serves today.
    ChatCompletions,
    /// Chat completions asked to stream, delivering the reply as incremental
    /// chunks.
    ///
    /// Separate from [`Api::ChatCompletions`] because a model can serve one
    /// without the other, so declaring that one never implies this one.
    ChatCompletionsStream,
    /// OpenAI's newer, stateful successor to chat completions.
    Responses,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, for the checks below. A new protocol goes here — the
    /// same standing obligation [`Protocol::apis`] carries.
    const ALL: &[Protocol] = &[Protocol::OpenAiCompat];

    /// [`Protocol::as_str`] keys the ext bags; the `serde(rename)` keys the
    /// registry YAML. They are the same string by contract, and hand-written
    /// in two places — so a rename that touches one and not the other has to
    /// fail here rather than in a passthrough field that silently stops
    /// round-tripping.
    #[test]
    fn protocol_name_matches_serde() {
        for &protocol in ALL {
            let from_name: Protocol =
                serde_json::from_value(serde_json::json!(protocol.as_str())).unwrap();
            assert_eq!(
                from_name,
                protocol,
                "{} deserializes to something else",
                protocol.as_str()
            );
        }
    }
}
