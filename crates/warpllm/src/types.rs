//! The vocabulary for API surfaces, and the wire formats they are spoken in.
//!
//! [`Api`] is what the registry YAML writes and what an entrypoint gates on:
//! one surface a model serves. It lives here rather than under
//! [`crate::protocol`] because the `gateway` layer's canonical forms — which
//! are supposed to be protocol-neutral — would otherwise reach into the module
//! tree that defines the protocols in order to name a surface.
//!
//! `Protocol` is the smaller thing, and crate-internal: a wire format's name,
//! used inside `gateway` to key the `ext` passthrough bags and to record what a
//! value was ingested from. The roster no longer mentions it — a surface names
//! its own protocol, so recording it a second time on the provider could only
//! ever restate or contradict.
//!
//! # An [`Api`] names its protocol
//!
//! `openai_compat_chat_completions`, not `chat_completions`. The surface alone
//! is ambiguous the moment a second protocol serves something comparable: chat
//! completions and Anthropic's messages are the same idea in two wire formats,
//! and a model may well serve both. Qualifying the name is what lets one model
//! list `openai_compat_chat_completions` and `anthropic_messages` side by side
//! without either the roster or this enum having to decide they are the same
//! thing — and it is the only place the roster now says which wire format is in
//! play.
//!
//! [`Api`] itself is a bare name. What a roster records ABOUT a surface lives
//! on [`crate::SupportedApi`], one struct holding every surface's fields — so a
//! field like `input_modalities` is declared once and every surface has it,
//! rather than being added to three payloads and kept in step by hand.

/// How a provider's wire protocol is spoken. One variant per wire format,
/// not per provider — a provider that diverges from its protocol is handled in
/// that protocol's conversions, never with a variant of its own.
///
/// Crate-internal, and read only by [`crate::gateway`]: it keys the `ext`
/// passthrough bags and tags an ingested value with the format it came from.
/// A caller reaches a surface through [`Api`], which names its protocol, so
/// there is nothing left for a public [`Protocol`] to answer.
///
/// Not deserialized from anything. The roster used to write `protocol:
/// openai_compat` on every provider; [`Api`] carries that now, so this is a tag
/// the code constructs rather than a value a file supplies.
///
/// `non_exhaustive` because this enum exists to grow: a new wire format is a
/// normal addition. Inside warpllm the attribute has no effect, so the crate's
/// own matches stay exhaustive and a new variant still fails to compile until
/// every arm handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum Protocol {
    /// The OpenAI-compatible chat completions wire format, implemented at
    /// [`crate::protocol::openai_compat`].
    OpenAiCompat,
}

impl Protocol {
    /// This protocol's name as a string: the key its passthrough fields are
    /// filed under in the gateway `ext` bags, and the provenance an ingested
    /// value carries.
    ///
    /// One source of truth for that string, so the three call sites that key a
    /// bag cannot drift apart.
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
/// implements it, so `openai_compat_chat_completions` is reached at
/// `protocol::openai_compat::chat_completions`, and the two cannot drift by a
/// rename.
///
/// The renames are spelled out rather than derived: `rename_all = "snake_case"`
/// reads the camel hump in `OpenAiCompat` and would give
/// `open_ai_compat_chat_completions`. `api_names_match_serde` pins every one of
/// them.
///
/// `non_exhaustive` because this exists to grow as warpllm implements more of
/// each provider's surface, and without it every downstream `match` would break
/// on a release that adds one. Inside warpllm the attribute has no effect, so
/// the crate's own matches stay exhaustive and a new variant still fails to
/// compile until every arm handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize)]
#[non_exhaustive]
pub enum Api {
    /// Chat completions as OpenAI-compatible providers speak them: one
    /// request, one whole reply. The only surface warpllm serves today.
    #[serde(rename = "openai_compat_chat_completions")]
    OpenAiCompatChatCompletions,
    /// The same request asked to stream, delivered as incremental chunks.
    ///
    /// Separate from [`Api::OpenAiCompatChatCompletions`] because a model can
    /// serve one without the other, so declaring that one never implies this
    /// one.
    #[serde(rename = "openai_compat_chat_completions_stream")]
    OpenAiCompatChatCompletionsStream,
    /// OpenAI's newer, stateful successor to chat completions.
    #[serde(rename = "openai_compat_responses")]
    OpenAiCompatResponses,
}

impl Api {
    /// This surface's name as a string: the spelling the registry YAML uses,
    /// and the one an error names it by.
    ///
    /// One source of truth for that string, so a message cannot drift from the
    /// roster line a reader would go and fix — `Debug` would say
    /// `OpenAiCompatChatCompletions`, which appears nowhere a contributor can
    /// act on. `api_names_match_serde` pins it to the `serde(rename)` above,
    /// which matters because both are hand-written.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Api::OpenAiCompatChatCompletions => "openai_compat_chat_completions",
            Api::OpenAiCompatChatCompletionsStream => "openai_compat_chat_completions_stream",
            Api::OpenAiCompatResponses => "openai_compat_responses",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every surface with the protocol it is spoken in, so a new variant has
    /// one place to be added for the checks below to cover it.
    ///
    /// The protocol is test data rather than a method on [`Api`]: nothing on a
    /// request path asks which protocol a surface belongs to — an entrypoint
    /// knows its own — so the mapping exists only to hold the naming
    /// convention below.
    const EVERY_API: &[(&str, Api, Protocol)] = &[
        (
            "openai_compat_chat_completions",
            Api::OpenAiCompatChatCompletions,
            Protocol::OpenAiCompat,
        ),
        (
            "openai_compat_chat_completions_stream",
            Api::OpenAiCompatChatCompletionsStream,
            Protocol::OpenAiCompat,
        ),
        (
            "openai_compat_responses",
            Api::OpenAiCompatResponses,
            Protocol::OpenAiCompat,
        ),
    ];

    /// The roster spells these by hand, and so do the `serde(rename)` on each
    /// variant and [`Api::as_str`]. All three are the same string by contract,
    /// so a rename that touches one and not the others has to fail here rather
    /// than as an "unknown variant" against a contributor's perfectly good
    /// roster line — or, worse, as an error message naming a surface that
    /// appears nowhere in the file.
    #[test]
    fn api_names_match_serde() {
        for &(name, expected, _) in EVERY_API {
            let parsed: Api = serde_json::from_value(serde_json::json!(name))
                .unwrap_or_else(|e| panic!("`api: {name}` is what the roster writes: {e}"));
            assert_eq!(parsed, expected, "`{name}` deserialized to something else");
            assert_eq!(parsed.as_str(), name, "`{name}` renders as something else");
        }
    }

    /// Every surface names the protocol it is spoken in, which is the whole
    /// reason the roster no longer records one per provider. With the
    /// provider-level field gone, the name is the ONLY place a reader learns
    /// the wire format — so a variant that dropped the prefix would leave the
    /// roster saying nothing about it at all.
    #[test]
    fn every_api_name_carries_its_protocol() {
        for &(name, _, protocol) in EVERY_API {
            assert!(
                name.starts_with(protocol.as_str()),
                "`{name}` does not name `{}`, the protocol it is spoken in",
                protocol.as_str()
            );
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
        for (known, _, _) in EVERY_API {
            assert!(err.contains(known), "vocabulary missing {known}: {err}");
        }
    }
}
