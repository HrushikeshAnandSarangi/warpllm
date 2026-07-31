//! The protocol enum: which wire dialect a model speaks, and what that
//! dialect defines.
//!
//! It lives here rather than with the registry because the registry only
//! REFERENCES a protocol — the conversions and transport that implement one
//! are the sibling modules next door.

/// How a provider's wire protocol is spoken. One variant per wire format,
/// not per provider — the exhaustive `match` in `client.rs` stays small
/// forever while adding a model is one entry in the registry YAML.
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
    /// (`crate::types::openai_compat`).
    #[serde(rename = "openai_compat")]
    OpenAiCompat,
}

impl Protocol {
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
