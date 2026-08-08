//! The vocabulary three layers agree on: which wire format a provider speaks,
//! and which API surfaces a model serves.
//!
//! These live above both [`crate::protocol`] and [`crate::gateway`] rather than
//! inside either, because both need them and neither owns them. [`Protocol`]
//! names a protocol, keys the gateway `ext` bags, and is the word the registry
//! YAML uses; [`Api`] names one protocol's surface and is the module path that
//! implements it. Filing them under the protocol layer would make the canonical
//! forms in `gateway::types` — which are supposed to be protocol-neutral — reach
//! into the module tree that defines the protocols to name their own bag keys.
//!
//! # An [`Api`] names its protocol
//!
//! `openai_chat_completions`, not `chat_completions`. The surface alone is
//! ambiguous the moment a second protocol serves something comparable: chat
//! completions and Anthropic's messages are the same idea in two wire formats,
//! and a model may well serve both. Qualifying the name is what lets one model
//! list `openai_chat_completions` and `anthropic_messages` side by side without
//! either the roster or this enum having to decide they are the same thing.
//!
//! [`Api`] itself is a bare name. What a roster records ABOUT a surface lives
//! on [`crate::SupportedApi`], one struct holding every surface's fields — so a
//! field like `input_modalities` is declared once and every surface has it,
//! rather than being added to three payloads and kept in step by hand.

/// How a provider's wire protocol is spoken. One variant per wire format,
/// not per provider — the exhaustive `match` in `client.rs` stays small
/// forever while adding a model is one entry in the registry YAML. A provider
/// that diverges from its protocol is handled in that protocol's conversions,
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
    /// The OpenAI-compatible chat completions wire format, implemented at
    /// [`crate::protocol::openai_compat`].
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
}

impl Protocol {
    /// This protocol's name as a string: the spelling the registry YAML uses,
    /// and the key its passthrough fields are filed under in the gateway
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
}

/// One API surface a model can serve — the `api:` of an entry in a registry
/// model's `supported_apis`.
///
/// A capability, not a URL: the path a surface is reached at belongs to the
/// protocol module implementing it. Being an enum is what makes a misspelling
/// fail to LOAD rather than 404 against a live provider at request time.
///
/// A bare name, carrying nothing. What the roster records about a surface is
/// [`crate::SupportedApi`]'s, so that a field belongs to every surface at once
/// — see this module's own docs. The name is also the module path that
/// implements it, so `openai_chat_completions` is reached at
/// `protocol::openai_compat::chat_completions`, and the two cannot drift by a
/// rename.
///
/// The renames are spelled out rather than derived for the same reason
/// [`Protocol`]'s is: `rename_all = "snake_case"` reads the camel hump in
/// `OpenAi` and would give `open_ai_chat_completions`. `api_names_match_serde`
/// pins every one of them.
///
/// `non_exhaustive` for the same reason as [`Protocol`]: this exists to grow
/// as warpllm implements more of each provider's surface, and without it every
/// downstream `match` would break on a release that adds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize)]
#[non_exhaustive]
pub enum Api {
    /// Chat completions as OpenAI-compatible providers speak them: one
    /// request, one whole reply. The only surface warpllm serves today.
    #[serde(rename = "openai_chat_completions")]
    OpenAiChatCompletions,
    /// The same request asked to stream, delivered as incremental chunks.
    ///
    /// Separate from [`Api::OpenAiChatCompletions`] because a model can serve
    /// one without the other, so declaring that one never implies this one.
    #[serde(rename = "openai_chat_completions_stream")]
    OpenAiChatCompletionsStream,
    /// OpenAI's newer, stateful successor to chat completions.
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

impl Api {
    /// This surface's name as a string: the spelling the registry YAML uses,
    /// and the one an error names it by.
    ///
    /// One source of truth for that string, so a message cannot drift from the
    /// roster line a reader would go and fix — `Debug` would say
    /// `OpenAiChatCompletions`, which appears nowhere a contributor can act on.
    /// `api_names_match_serde` pins it to the `serde(rename)` above, which
    /// matters because both are hand-written.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Api::OpenAiChatCompletions => "openai_chat_completions",
            Api::OpenAiChatCompletionsStream => "openai_chat_completions_stream",
            Api::OpenAiResponses => "openai_responses",
        }
    }

    /// The protocol this surface is spoken in — read off the variant, which
    /// names it.
    ///
    /// The inverse of the list [`Protocol`] used to carry, and the better
    /// direction: a surface belongs to exactly one protocol, while a protocol
    /// serves many. `registry::lint` reads it to hold a model's surfaces
    /// against the provider serving them.
    ///
    /// `cfg(test)` because that lint is its only reader and is itself
    /// test-only. Nothing on a request path consults it — dispatch reads the
    /// provider's own [`Protocol`].
    #[cfg(test)]
    pub(crate) fn protocol(self) -> Protocol {
        match self {
            Api::OpenAiChatCompletions
            | Api::OpenAiChatCompletionsStream
            | Api::OpenAiResponses => Protocol::OpenAiCompat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, for the checks below. A new protocol goes here — the
    /// same standing obligation [`Api::protocol`] carries in the other
    /// direction.
    const ALL: &[Protocol] = &[Protocol::OpenAiCompat];

    /// Every surface, so a new variant has one place to be added for the
    /// checks below to cover it.
    const EVERY_API: &[(&str, Api)] = &[
        ("openai_chat_completions", Api::OpenAiChatCompletions),
        (
            "openai_chat_completions_stream",
            Api::OpenAiChatCompletionsStream,
        ),
        ("openai_responses", Api::OpenAiResponses),
    ];

    /// The roster spells these by hand, and so do the `serde(rename)` on each
    /// variant and [`Api::as_str`]. All three are the same string by contract,
    /// so a rename that touches one and not the others has to fail here rather
    /// than as an "unknown variant" against a contributor's perfectly good
    /// roster line — or, worse, as an error message naming a surface that
    /// appears nowhere in the file.
    #[test]
    fn api_names_match_serde() {
        for &(name, expected) in EVERY_API {
            let parsed: Api = serde_json::from_value(serde_json::json!(name))
                .unwrap_or_else(|e| panic!("`api: {name}` is what the roster writes: {e}"));
            assert_eq!(parsed, expected, "`{name}` deserialized to something else");
            assert_eq!(parsed.as_str(), name, "`{name}` renders as something else");
            // The variant names its own protocol, which is the whole reason
            // the surface is spelled with a prefix.
            assert_eq!(parsed.protocol(), Protocol::OpenAiCompat, "`{name}`");
        }
    }

    /// A surface warpllm has never heard of fails to parse, which is what
    /// keeps a misspelling out of the roster. The message names the whole
    /// vocabulary so the line can be fixed without opening this file.
    #[test]
    fn an_unknown_surface_is_rejected() {
        let err = serde_json::from_value::<Api>(serde_json::json!("anthropic_messages"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown variant"), "{err}");
        for (known, _) in EVERY_API {
            assert!(err.contains(known), "vocabulary missing {known}: {err}");
        }
    }

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
