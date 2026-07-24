//! DeepSeek's model roster. Entries inherit [`BASE`] with `..BASE` and
//! restate only what differs, so a model is one line plus its deltas.

use crate::model::{ANY_MODEL, Capabilities, ModelSpec, Protocol};

/// What holds for every DeepSeek model, per
/// <https://api-docs.deepseek.com/api/create-chat-completion> (checked
/// 2026-07-24). Also the spec any unlisted model resolves to.
const BASE: ModelSpec<'static> = ModelSpec {
    provider: "deepseek",
    model: ANY_MODEL,
    base_url: "https://api.deepseek.com",
    env_key: "DEEPSEEK_API_KEY",
    protocol: Protocol::OpenAiCompat,
    capabilities: Capabilities {
        supported_endpoints: &["/chat/completions"],
        max_input_tokens: None,
        max_output_tokens: None,
        max_concurrent_requests: None,
    },
};

/// The V4 pair (the legacy `deepseek-chat` / `deepseek-reasoner` names were
/// discontinued 2026-07-24 and map to V4-Flash modes). They diverge on
/// concurrency: Flash serves 5x what Pro does.
pub(crate) const MODELS: &[ModelSpec<'static>] = &[
    BASE,
    ModelSpec {
        model: "deepseek-v4-flash",
        capabilities: Capabilities {
            max_concurrent_requests: Some(2500),
            ..BASE.capabilities
        },
        ..BASE
    },
    ModelSpec {
        model: "deepseek-v4-pro",
        capabilities: Capabilities {
            max_concurrent_requests: Some(500),
            ..BASE.capabilities
        },
        ..BASE
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(model: &str) -> &'static ModelSpec<'static> {
        MODELS.iter().find(|s| s.model == model).unwrap()
    }

    #[test]
    fn the_v4_pair_diverges_only_on_concurrency() {
        let flash = spec("deepseek-v4-flash");
        let pro = spec("deepseek-v4-pro");
        assert_eq!(flash.capabilities.max_concurrent_requests, Some(2500));
        assert_eq!(pro.capabilities.max_concurrent_requests, Some(500));
        // Everything else is inherited from BASE, not restated.
        for model in [flash, pro] {
            assert_eq!(model.base_url, BASE.base_url);
            assert_eq!(model.env_key, BASE.env_key);
            assert_eq!(
                model.capabilities.supported_endpoints,
                BASE.capabilities.supported_endpoints
            );
        }
    }

    #[test]
    fn the_base_spec_covers_unlisted_models() {
        let base = spec(ANY_MODEL);
        assert_eq!(base.provider, "deepseek");
        assert_eq!(base.env_key, "DEEPSEEK_API_KEY");
        assert!(base.capabilities.supports_endpoint("/chat/completions"));
        assert!(!base.capabilities.supports_endpoint("/responses"));
        // Undocumented, NOT unlimited.
        assert_eq!(base.capabilities.max_concurrent_requests, None);
    }
}
