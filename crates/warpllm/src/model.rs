//! Parsing of `provider/model` strings into the spec for that model, plus
//! the model roster itself.

use crate::error::{Error, Result};
use crate::providers;

/// What a model can do. Deliberately NOT coupled to the request shape:
/// parameter support is passthrough — the provider is the authority and
/// rejects what it doesn't accept. A field is added here only when a real
/// consumer need arrives with it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Capabilities {
    /// Endpoint paths under the provider's `base_url` this model serves,
    /// e.g. `"/chat/completions"`.
    pub supported_endpoints: &'static [&'static str],
    #[allow(dead_code)] // spec data; read once a consumer need arrives
    pub max_input_tokens: Option<u32>,
    #[allow(dead_code)] // spec data; read once a consumer need arrives
    pub max_output_tokens: Option<u32>,
    /// Requests this model will serve at once. `None` means undocumented,
    /// NOT unlimited. Account tier can move these, so treat the roster
    /// value as the default a config surface would later override.
    #[allow(dead_code)] // spec data; read once a consumer need arrives
    pub max_concurrent_requests: Option<u32>,
}

impl Capabilities {
    pub(crate) fn supports_endpoint(&self, endpoint: &str) -> bool {
        self.supported_endpoints.contains(&endpoint)
    }
}

/// How a provider's wire protocol is spoken. One variant per wire format,
/// not per provider — the exhaustive `match` in `client.rs` stays small
/// forever while adding a model is one line of data in [`MODELS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Protocol {
    /// The OpenAI-compatible chat completions wire format
    /// (`crate::types::openai_compat`).
    OpenAiCompat,
}

/// Everything warpllm knows about one model: how to reach it and what it
/// can do. The spec is per MODEL, not per provider, because that is the
/// granularity at which support diverges — two models from the same
/// provider can differ — so this is the only layer to maintain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelSpec<'a> {
    /// The prefix in `provider/model` strings; also names the provider in
    /// errors and its `ext` namespace.
    pub provider: &'static str,
    /// Upstream model name, provider prefix stripped. [`ANY_MODEL`] in a
    /// roster entry marks the provider's base spec.
    pub model: &'a str,
    /// API root including any version prefix (e.g. `/v1`); endpoint impls
    /// append only their own path (e.g. `/chat/completions`).
    pub base_url: &'static str,
    /// Environment variable holding the provider's API key.
    pub env_key: &'static str,
    pub protocol: Protocol,
    pub capabilities: Capabilities,
}

/// The `model` of a provider's base spec: what every model from that
/// provider resolves to unless it has its own entry.
pub(crate) const ANY_MODEL: &str = "*";

/// The roster, one slice per provider module. Entries are model specs;
/// a provider contributes its base spec plus one entry per model that
/// diverges from it. Inheritance is plain `..BASE` struct-update syntax
/// in each roster — a model entry restates only its deltas.
// Linear scan; switch to a `phf::Map` when the roster reaches hundreds.
const MODELS: &[&[ModelSpec<'static>]] = &[
    providers::openai::models::MODELS,
    providers::deepseek::models::MODELS,
];

/// Resolves a `"provider/model"` string into the spec for that model: its
/// own entry if the roster has one, else the provider's [`ANY_MODEL`] base
/// spec with the requested name written in.
///
/// Bare model names (no `/`) are rejected rather than defaulting to a
/// provider: silently assuming OpenAI becomes a footgun once many providers
/// exist. `split_once` keeps any `/` inside the model name intact.
pub(crate) fn parse_model(requested: &str) -> Result<ModelSpec<'_>> {
    let invalid = || Error::InvalidModel {
        given: requested.to_string(),
    };
    let Some((provider, model)) = requested.split_once('/') else {
        return Err(invalid());
    };
    if model.is_empty() {
        return Err(invalid());
    }
    let find = |name: &str| {
        MODELS
            .iter()
            .flat_map(|specs| specs.iter())
            .find(|spec| spec.provider == provider && spec.model == name)
    };
    let spec = find(model)
        .or_else(|| find(ANY_MODEL))
        .ok_or_else(invalid)?;
    // The base spec carries ANY_MODEL; the caller's name is what ships.
    Ok(ModelSpec { model, ..*spec })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai() {
        let spec = parse_model("openai/gpt-4o").unwrap();
        assert_eq!(spec.provider, "openai");
        assert_eq!(spec.model, "gpt-4o");
    }

    #[test]
    fn parses_deepseek() {
        let spec = parse_model("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(spec.provider, "deepseek");
        assert_eq!(spec.model, "deepseek-v4-flash");
        assert_eq!(spec.base_url, "https://api.deepseek.com");
        assert_eq!(spec.env_key, "DEEPSEEK_API_KEY");
        assert_eq!(spec.protocol, Protocol::OpenAiCompat);
    }

    #[test]
    fn capabilities_resolve_for_the_addressed_model() {
        let openai = parse_model("openai/gpt-4o").unwrap();
        assert!(openai.capabilities.supports_endpoint("/chat/completions"));
        assert!(openai.capabilities.supports_endpoint("/responses"));

        let deepseek = parse_model("deepseek/deepseek-v4-flash").unwrap();
        assert!(deepseek.capabilities.supports_endpoint("/chat/completions"));
        assert!(!deepseek.capabilities.supports_endpoint("/responses"));
    }

    /// Every model resolves through its provider's base spec until it
    /// earns an entry of its own — and the caller's name always ships,
    /// never the `ANY_MODEL` marker.
    #[test]
    fn unlisted_models_fall_back_to_the_providers_base_spec() {
        let spec = parse_model("openai/some-future-model").unwrap();
        assert_eq!(spec.model, "some-future-model");
        assert_eq!(spec.base_url, "https://api.openai.com/v1");
        assert!(spec.capabilities.supports_endpoint("/responses"));
    }

    /// A listed model resolves to its own entry, not the provider base.
    /// Iterates the roster so new entries are covered automatically.
    #[test]
    fn listed_models_resolve_to_their_own_entry() {
        let listed: Vec<_> = MODELS
            .iter()
            .flat_map(|specs| specs.iter())
            .filter(|spec| spec.model != ANY_MODEL)
            .collect();
        assert!(!listed.is_empty(), "roster has no per-model entries");
        for spec in listed {
            let requested = format!("{}/{}", spec.provider, spec.model);
            let resolved = parse_model(&requested).unwrap();
            assert_eq!(resolved.model, spec.model);
            assert_eq!(resolved.base_url, spec.base_url);
            assert_eq!(
                resolved.capabilities.max_concurrent_requests,
                spec.capabilities.max_concurrent_requests
            );
        }
    }

    /// The divergence that motivated per-model entries: same provider,
    /// same everything else, 5x the concurrency.
    #[test]
    fn concurrency_limits_resolve_per_model() {
        let flash = parse_model("deepseek/deepseek-v4-flash").unwrap();
        let pro = parse_model("deepseek/deepseek-v4-pro").unwrap();
        assert_eq!(flash.capabilities.max_concurrent_requests, Some(2500));
        assert_eq!(pro.capabilities.max_concurrent_requests, Some(500));
        assert_eq!(flash.base_url, pro.base_url);

        // An unlisted model inherits the base, where the limit is unknown.
        let unlisted = parse_model("deepseek/deepseek-v5").unwrap();
        assert_eq!(unlisted.capabilities.max_concurrent_requests, None);
    }

    #[test]
    fn every_provider_has_exactly_one_base_spec() {
        let mut bases: Vec<_> = MODELS
            .iter()
            .flat_map(|specs| specs.iter())
            .filter(|spec| spec.model == ANY_MODEL)
            .map(|spec| spec.provider)
            .collect();
        let count = bases.len();
        bases.sort_unstable();
        bases.dedup();
        assert_eq!(bases.len(), count, "a provider declared two base specs");
    }

    #[test]
    fn keeps_slashes_in_model_name() {
        let spec = parse_model("openai/org/custom-model").unwrap();
        assert_eq!(spec.provider, "openai");
        assert_eq!(spec.model, "org/custom-model");
    }

    #[test]
    fn rejects_bare_model() {
        let msg = parse_model("gpt-4o").unwrap_err().to_string();
        assert!(msg.contains("not a supported provider"), "{msg}");
    }

    #[test]
    fn rejects_unknown_provider() {
        let msg = parse_model("mistral/large").unwrap_err().to_string();
        assert!(msg.contains("mistral/large"), "{msg}");
        assert!(msg.contains("not a supported provider"), "{msg}");
    }

    #[test]
    fn rejects_empty_model_name() {
        assert!(parse_model("openai/").is_err());
    }
}
