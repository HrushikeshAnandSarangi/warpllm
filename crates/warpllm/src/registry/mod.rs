//! The model roster and the lookup over it.
//!
//! The roster is pure data, edited by the community in `specs.yaml` next
//! door. Nothing here is hand-written per provider: adding a provider or a
//! model is a YAML edit, never a Rust edit.
//!
//! The file is compiled into the binary with `include_str!` and loaded once,
//! lazily, on the first lookup — a few kilobytes, turned into two `HashMap`s
//! that then answer every request without walking anything. It used to be
//! compiled to Rust source by a build script; that bought a build-time gate on
//! a bad roster and cost a code generator writing Rust by string
//! concatenation, a types file `include!`d into two crates, and leaked
//! `&'static` strings to make the result a `const`. The gate now lives in
//! `the_shipped_registry_loads_and_lints`, which CI runs on every PR.
//!
//! Three modules and no more: `types` is what a spec IS, `load` turns the
//! file into the tables, `lint` holds what is merely true of a tidy roster.
//! Each keeps its own tests.

use std::sync::LazyLock;

use crate::error::{Error, Result};

mod load;
mod types;

#[cfg(test)]
mod lint;
#[cfg(test)]
mod testing;

use types::Registry;
pub use types::{Capabilities, ModelSpec, ProviderSpec};

/// The shipped roster, loaded on first use.
///
/// The panic is deliberate and stays out of [`fetch_model`]'s signature: the
/// input is compiled into this binary, cannot vary at runtime, and is held to
/// both gates by `the_shipped_registry_loads_and_lints`. Threading "warpllm's
/// own roster is malformed" through the public API would put an arm at every
/// call site that no caller could ever act on.
static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    load::load(include_str!("specs.yaml")).unwrap_or_else(|e| panic!("specs.yaml: {e}"))
});

/// What warpllm knows about a `model_str` such as `"openai/gpt-5.6"`: the
/// provider that serves it, and the model itself.
///
/// Two rows rather than one merged spec, because the roster keeps them at two
/// levels — how the API is reached and what it serves is the provider's, the
/// upstream name and the published limits are the model's. Nothing is copied
/// between them, so a provider's transport is stated once no matter how many
/// models it serves.
///
/// The key matches exactly or not at all. Nothing routes on a guess — no
/// pattern, no catch-all, no fallback — so a name no entry claims is an error,
/// and a typo cannot reach a provider as a live, billed request. That includes
/// a bare name with no `/`, since silently assuming OpenAI becomes a footgun
/// once many providers exist.
///
/// # Errors
///
/// [`Error::InvalidModel`] if nothing is registered for `model_str`.
///
/// # Examples
///
/// ```
/// let (provider, model) = warpllm::fetch_model("openai/gpt-5.6")?;
/// assert_eq!(provider.name(), "openai");
/// assert!(provider.supports_api(warpllm::Api::ChatCompletions));
/// assert_eq!(model.model(), "gpt-5.6");
///
/// // A name nobody registered is an error, never a guess.
/// assert!(warpllm::fetch_model("openai/nonexistent").is_err());
/// # Ok::<(), warpllm::Error>(())
/// ```
pub fn fetch_model(model_str: &str) -> Result<(&'static ProviderSpec, &'static ModelSpec)> {
    resolve(&REGISTRY, model_str).ok_or_else(|| Error::InvalidModel {
        given: model_str.to_string(),
    })
}

/// The model row filed under `model_str`, and the provider row it names.
///
/// One hash lookup, then a second. There is no second chance at the first: a
/// key the table does not hold is a model warpllm does not serve, and the
/// caller hears that rather than a provider hearing a guess.
///
/// Split out from [`fetch_model`] so the matching can be tested against a
/// fixture roster rather than only the shipped one.
fn resolve<'a>(
    registry: &'a Registry,
    model_str: &str,
) -> Option<(&'a ProviderSpec, &'a ModelSpec)> {
    let model = registry.models.get(model_str)?;
    let provider = registry
        .providers
        .get(&model.provider)
        .expect("load registers a provider for every model it holds");
    Some((provider, model))
}

#[cfg(test)]
mod tests {
    use super::testing::{clean, keys, providers, with};
    use super::*;
    use crate::protocol::{Api, Protocol};

    // ------------------------------------------------------------- matching

    /// The rule the roster documents: a key matches its own entry or nothing
    /// at all, and what ships upstream is that entry's name.
    #[test]
    fn a_key_matches_its_own_entry_or_nothing() {
        let registry = clean(&with(
            "      demo/pinned:\n        model: pinned-2024\n      demo/plain: {}\n",
        ));
        assert_eq!(
            resolve(&registry, "demo/pinned").unwrap().1.model(),
            "pinned-2024"
        );
        assert_eq!(resolve(&registry, "demo/plain").unwrap().1.model(), "plain");
        // One character off, and there is nothing to fall back to.
        assert!(resolve(&registry, "demo/pinnedd").is_none());
        assert!(resolve(&registry, "demo/PLAIN").is_none());
    }

    /// Whichever entry matched, the provider row comes back with it — that is
    /// the whole point of handing back both halves.
    #[test]
    fn a_match_carries_its_provider() {
        let registry = clean(&with("      demo/plain: {}\n"));
        let (provider, _) = resolve(&registry, "demo/plain").unwrap();
        assert_eq!(provider.name(), "demo");
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
        assert_eq!(provider.protocol(), Protocol::OpenAiCompat);
    }

    /// The registry is closed, so an unlisted name is an error — the whole
    /// reason a typo cannot become a billed upstream request.
    ///
    /// A pattern is just another unlisted name. Nothing reads `*` as anything
    /// but a character, so asking for one matches exactly as much as asking
    /// for a misspelling: nothing.
    #[test]
    fn an_unlisted_name_is_rejected() {
        let registry = clean(&with("      demo/plain: {}\n"));
        assert!(resolve(&registry, "demo/plain").is_some());
        for unlisted in ["demo/unlisted", "demo/*", "demo/pl*", "*"] {
            assert!(resolve(&registry, unlisted).is_none(), "`{unlisted}`");
        }
    }

    /// A name grouped under a prefix is registered like any other, and the
    /// grouping buys it no fallback.
    #[test]
    fn a_grouped_name_matches_only_its_own_entry() {
        let registry = clean(&with("      demo/org/one: {}\n      demo/plain: {}\n"));
        assert!(resolve(&registry, "demo/org/one").is_some());
        assert!(resolve(&registry, "demo/org/two").is_none());
        assert!(resolve(&registry, "demo/org").is_none());
    }

    /// A provider is not a model. Its key names transport, and no request
    /// routes to it.
    #[test]
    fn a_provider_name_is_not_routable() {
        let registry = clean(&with("      demo/plain: {}\n"));
        assert!(resolve(&registry, "demo/").is_none());
        assert!(resolve(&registry, "demo").is_none());
    }

    // -------------------------------------------------------- shipped roster

    /// The shipped roster is the one case that has to keep working, and the
    /// only place every policy is enforced against real content. This test is
    /// what replaced the build-time gate: it is why [`REGISTRY`] can panic on a
    /// malformed file without that ever reaching anyone.
    #[test]
    fn the_shipped_registry_loads_and_lints() {
        let yaml = include_str!("specs.yaml");
        lint::check(yaml).unwrap_or_else(|e| panic!("specs.yaml: {e}"));
        let registry = load::load(yaml).unwrap();
        assert_eq!(providers(&registry), vec!["deepseek", "openai"]);
        assert_eq!(
            keys(&registry),
            vec![
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-v4-pro",
                "openai/gpt-5.6",
                "openai/gpt-5.6-luna",
                "openai/gpt-5.6-sol",
                "openai/gpt-5.6-terra",
            ]
        );
    }

    #[test]
    fn resolves_openai() {
        let (provider, model) = fetch_model("openai/gpt-5.6").unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
        assert_eq!(model.model(), "gpt-5.6");
    }

    #[test]
    fn resolves_deepseek() {
        let (provider, model) = fetch_model("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(provider.name(), "deepseek");
        assert_eq!(provider.base_url(), "https://api.deepseek.com");
        assert_eq!(provider.env_api_key(), Some("DEEPSEEK_API_KEY"));
        assert_eq!(provider.protocol(), Protocol::OpenAiCompat);
        assert_eq!(model.model(), "deepseek-v4-flash");
    }

    /// The whole GPT-5.6 family is routable and answered by one provider row;
    /// the entries exist only to register the names.
    #[test]
    fn the_gpt_5_6_family_is_registered() {
        let (base, _) = fetch_model("openai/gpt-5.6").unwrap();
        for name in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"] {
            let (provider, model) = fetch_model(&format!("openai/{name}")).unwrap();
            assert_eq!(model.model(), name);
            assert!(
                std::ptr::eq(provider, base),
                "`{name}` got a second copy of its provider"
            );
        }
    }

    /// API surfaces are the provider's claim, and differ between providers.
    #[test]
    fn supported_apis_resolve_for_the_addressed_provider() {
        let (openai, _) = fetch_model("openai/gpt-5.6").unwrap();
        assert!(openai.supports_api(Api::ChatCompletions));
        assert!(openai.supports_api(Api::Responses));

        let (deepseek, _) = fetch_model("deepseek/deepseek-v4-flash").unwrap();
        assert!(deepseek.supports_api(Api::ChatCompletions));
        assert!(!deepseek.supports_api(Api::Responses));
    }

    /// Every registered model resolves to its own entry. Iterates the registry
    /// so a new `specs.yaml` entry is covered automatically.
    #[test]
    fn registered_models_resolve_to_their_own_entry() {
        assert!(!REGISTRY.models.is_empty(), "the registry is empty");
        for (model_str, spec) in &REGISTRY.models {
            let (provider, resolved) = fetch_model(model_str).unwrap();
            assert_eq!(resolved.model(), spec.model());
            assert_eq!(provider.name(), spec.provider);
            assert_eq!(
                resolved.capabilities().max_concurrent_requests(),
                spec.capabilities().max_concurrent_requests()
            );
        }
    }

    /// What per-model entries are FOR: the V4 pair shares one provider row and
    /// differs only in the limit that motivated splitting them.
    #[test]
    fn model_entries_carry_only_what_differs() {
        let (flash_provider, flash) = fetch_model("deepseek/deepseek-v4-flash").unwrap();
        let (pro_provider, pro) = fetch_model("deepseek/deepseek-v4-pro").unwrap();
        assert!(std::ptr::eq(flash_provider, pro_provider));
        // The divergence that motivated per-model entries: 5x the concurrency.
        assert_eq!(flash.capabilities().max_concurrent_requests(), Some(2500));
        assert_eq!(pro.capabilities().max_concurrent_requests(), Some(500));
    }

    /// The registry is closed: an unlisted name is an error, not a fallback.
    /// `openai/*` is in the list because a pattern matches nothing either.
    #[test]
    fn unregistered_models_are_rejected() {
        for model_str in ["openai/gpt-4o", "deepseek/deepseek-v5", "openai/*"] {
            let msg = fetch_model(model_str).unwrap_err().to_string();
            assert!(msg.contains(model_str), "{msg}");
            assert!(msg.contains("no registered model spec"), "{msg}");
        }
    }

    /// A slash-containing name needs an entry of its own, and the registry
    /// has none under `openai/org/`.
    #[test]
    fn unregistered_grouped_names_are_rejected() {
        assert!(fetch_model("openai/org/custom-model").is_err());
    }

    #[test]
    fn rejects_bare_model() {
        let msg = fetch_model("gpt-5.6").unwrap_err().to_string();
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    #[test]
    fn rejects_unknown_provider() {
        let msg = fetch_model("mistral/large").unwrap_err().to_string();
        assert!(msg.contains("mistral/large"), "{msg}");
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    /// A provider key is transport, not a model, so it cannot be routed to.
    #[test]
    fn rejects_a_provider_key() {
        assert!(fetch_model("openai/").is_err());
        assert!(fetch_model("openai").is_err());
    }

    /// Nothing but routable entries reaches the model table: every key
    /// carrying the provider it is filed under, every row agreeing with the
    /// key it sits at, and no key reaching for a pattern.
    ///
    /// That last one is asserted HERE and nowhere else. `load` reads a key
    /// literally and has no opinion about `*`, so `openai/*` would register a
    /// model named `*` and quietly serve nothing — this is what keeps the
    /// shipped file from acquiring one, whether by a stale roster or by
    /// somebody expecting the catch-all warpllm used to have.
    ///
    /// The fixtures elsewhere assert the same over hand-written rosters. This
    /// asserts it over [`REGISTRY`], which is the table a caller actually
    /// routes against — a resolver that passed every other test and still
    /// built a bad table would fail here and only here.
    #[test]
    fn the_table_holds_only_routable_entries() {
        for (model_str, spec) in &REGISTRY.models {
            let (provider, name) = model_str
                .split_once('/')
                .unwrap_or_else(|| panic!("`{model_str}` names no provider"));
            assert_eq!(spec.provider, provider, "`{model_str}`");
            assert!(REGISTRY.providers.contains_key(provider), "`{model_str}`");
            assert!(!name.is_empty(), "`{model_str}` names no model");
            assert!(
                !model_str.contains('*') && !spec.model().contains('*'),
                "`{model_str}`: a `*` reached the table, which matches nothing"
            );
        }
    }
}
