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
    /// The endpoint paths this wire format defines.
    ///
    /// A registry entry naming anything else is a typo or a protocol
    /// mismatch, which is what `registry::lint` reads this for. Widening a
    /// protocol's surface is an edit here, beside the module that implements
    /// it — not in the roster.
    ///
    /// `cfg(test)` because that lint is its only reader, and the lint itself
    /// is test-only. Nothing on a request path consults this: a spec carries
    /// its OWN endpoint list, already checked against this one.
    #[cfg(test)]
    pub(crate) fn endpoints(self) -> &'static [&'static str] {
        match self {
            Protocol::OpenAiCompat => &[
                "/chat/completions",
                "/embeddings",
                "/moderations",
                "/responses",
            ],
        }
    }
}
