//! The model roster and the lookup over it.
//!
//! The roster is pure data, edited by the community in `specs.yaml` next
//! door. Nothing here is hand-written per provider: adding a provider or a
//! model is a YAML edit, never a Rust edit.
//!
//! The file is compiled into the binary with `include_str!` and resolved once,
//! lazily, on the first lookup — a few kilobytes, turned into a `HashMap` that
//! then answers every request without walking anything. It used to be compiled
//! to Rust source by a build script; that bought a build-time gate on a bad
//! roster and cost a code generator writing Rust by string concatenation, a
//! types file `include!`d into two crates, and leaked `&'static` strings to
//! make the result a `const`. The gate now lives in
//! `the_shipped_registry_loads_and_lints`, which CI runs on every PR.
//!
//! Three modules and no more: `types` is what an entry IS, `load` turns the
//! file into the table, `lint` holds what is merely true of a tidy roster.
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
pub use types::{Capabilities, ModelSpec};

/// The shipped roster, resolved on first use.
///
/// The panic is deliberate and stays out of [`model_spec`]'s signature: the
/// input is compiled into this binary, cannot vary at runtime, and is held to
/// both gates by `the_shipped_registry_loads_and_lints`. Threading "warpllm's
/// own roster is malformed" through the public API would put an arm at every
/// call site that no caller could ever act on.
static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    load::load(include_str!("specs.yaml")).unwrap_or_else(|e| panic!("specs.yaml: {e}"))
});

/// The spec warpllm has for a `model_str` such as `"openai/gpt-5.6"`.
///
/// The most specific entry wins: an exact key first, then the `*` of the
/// namespace the name sits directly in. Nothing routes on a guess beyond
/// that — a name no entry claims is an error rather than a fallback, so a typo
/// cannot reach a provider as a live request, and that includes a bare name
/// with no `/`, since silently assuming OpenAI becomes a footgun once many
/// providers exist.
///
/// A wildcard is the one deliberate hole in that, and only for namespaces that
/// opted in by declaring a `*`. Under one of those a typo IS routable, which is
/// the trade the roster makes knowingly — see [`ModelSpec::is_wildcard`].
///
/// # Errors
///
/// [`Error::InvalidModel`] if nothing is registered for `model_str`.
///
/// # Examples
///
/// ```
/// let spec = warpllm::model_spec("openai/gpt-5.6")?;
/// assert_eq!(spec.provider(), "openai");
/// assert!(spec.capabilities().supports_api(warpllm::Api::ChatCompletions));
///
/// // A name nobody registered, under a namespace with no `*`, is an error.
/// assert!(warpllm::model_spec("openai/nonexistent").is_err());
/// # Ok::<(), warpllm::Error>(())
/// ```
pub fn model_spec(model_str: &str) -> Result<&'static ModelSpec> {
    resolve(&REGISTRY, model_str).ok_or_else(|| Error::InvalidModel {
        given: model_str.to_string(),
    })
}

/// Exact key, else the wildcard of the namespace directly above.
///
/// Split out from [`model_spec`] so the matching can be tested against a
/// fixture roster rather than only the shipped one.
fn resolve<'a>(registry: &'a Registry, model_str: &str) -> Option<&'a ModelSpec> {
    // A `*` in a REQUEST is never part of a name. Without this a caller could
    // address a wildcard entry by its own key and ship `*` upstream as the
    // model name.
    if model_str.contains(load::WILDCARD) {
        return None;
    }
    if let Some(spec) = registry.get(model_str) {
        return Some(spec);
    }
    // One level, never a walk: `openai/org/x` may fall back to `openai/org/*`
    // and never to `openai/*`, so a nested namespace cannot be swallowed by a
    // wildcard declared above it.
    let (namespace, name) = model_str.rsplit_once('/')?;
    if name.is_empty() {
        return None; // a namespace key is merge material, not a model
    }
    registry.get(&format!("{namespace}/{}", load::WILDCARD))
}

#[cfg(test)]
mod tests {
    use super::testing::{clean, keys, with};
    use super::*;
    use crate::protocol::{Api, Protocol};

    // ------------------------------------------------------------- matching

    /// The rule the roster documents: an exact entry beats the wildcard that
    /// would otherwise have served the same name.
    #[test]
    fn an_exact_entry_beats_a_wildcard() {
        let registry = clean(&with(
            "  demo/*: {}\n  demo/pinned:\n    model: pinned-2024\n",
        ));
        let exact = resolve(&registry, "demo/pinned").unwrap();
        assert!(!exact.is_wildcard());
        assert_eq!(exact.wire_model("demo/pinned"), "pinned-2024");

        let matched = resolve(&registry, "demo/anything-else").unwrap();
        assert!(matched.is_wildcard());
        assert_eq!(matched.wire_model("demo/anything-else"), "anything-else");
    }

    /// Without a `*` the registry stays closed: an unlisted name is an error.
    #[test]
    fn a_namespace_without_a_wildcard_rejects_unlisted_names() {
        let registry = clean(&with("  demo/plain: {}\n"));
        assert!(resolve(&registry, "demo/plain").is_some());
        assert!(resolve(&registry, "demo/unlisted").is_none());
    }

    /// A wildcard matches exactly one segment. A nested name falls back to ITS
    /// namespace's `*` and to nothing else, so `demo/*` cannot quietly serve
    /// everything below `demo/org/`.
    #[test]
    fn a_wildcard_matches_one_segment_and_does_not_cross_a_slash() {
        let registry = clean(&with(
            "  demo/*: {}\n  demo/org/:\n    env_api_key: DEMO_API_KEY\n  demo/org/*: {}\n",
        ));
        assert!(resolve(&registry, "demo/anything").unwrap().is_wildcard());
        assert!(resolve(&registry, "demo/org/thing").unwrap().is_wildcard());
        assert_eq!(
            resolve(&registry, "demo/org/thing")
                .unwrap()
                .wire_model("demo/org/thing"),
            "thing"
        );
        // Three levels deep, and only `demo/a/*` could serve it.
        assert!(resolve(&registry, "demo/a/b").is_none());
    }

    /// A nested name with no wildcard of its own gets nothing, even when an
    /// outer namespace has one.
    #[test]
    fn an_outer_wildcard_does_not_serve_a_nested_name() {
        let registry = clean(&with("  demo/*: {}\n  demo/org/: {}\n  demo/org/one: {}\n"));
        assert!(resolve(&registry, "demo/org/one").is_some());
        assert!(
            resolve(&registry, "demo/org/two").is_none(),
            "`demo/*` reached across a `/`"
        );
    }

    /// A wildcard entry is not addressable by its own key: `*` in a request is
    /// never a name, and matching it would ship `*` upstream.
    #[test]
    fn a_request_naming_a_star_is_rejected() {
        let registry = clean(&with("  demo/*: {}\n"));
        for asked in ["demo/*", "demo/gpt-*", "*"] {
            assert!(resolve(&registry, asked).is_none(), "matched `{asked}`");
        }
    }

    /// A namespace is merge material. It is not in the table, and it must not
    /// reach the wildcard beside it either.
    #[test]
    fn a_namespace_key_is_not_routable_even_beside_a_wildcard() {
        let registry = clean(&with("  demo/*: {}\n"));
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
        assert_eq!(
            keys(&load::load(yaml).unwrap()),
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
        let spec = model_spec("openai/gpt-5.6").unwrap();
        assert_eq!(spec.provider(), "openai");
        assert_eq!(spec.model(), "gpt-5.6");
    }

    #[test]
    fn resolves_deepseek() {
        let spec = model_spec("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(spec.provider(), "deepseek");
        assert_eq!(spec.model(), "deepseek-v4-flash");
        assert_eq!(spec.base_url(), "https://api.deepseek.com");
        assert_eq!(spec.env_api_key(), Some("DEEPSEEK_API_KEY"));
        assert_eq!(spec.protocol(), Protocol::OpenAiCompat);
    }

    /// The whole GPT-5.6 family is routable and takes its namespace's fields;
    /// the entries exist only to register the names.
    #[test]
    fn the_gpt_5_6_family_is_registered() {
        let base = model_spec("openai/gpt-5.6").unwrap();
        for model in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"] {
            let spec = model_spec(&format!("openai/{model}")).unwrap();
            assert_eq!(spec.model(), model);
            assert_eq!(spec.base_url(), base.base_url());
            assert_eq!(spec.env_api_key(), base.env_api_key());
            assert_eq!(
                spec.capabilities().supported_apis(),
                base.capabilities().supported_apis()
            );
        }
    }

    #[test]
    fn capabilities_resolve_for_the_addressed_model() {
        let openai = model_spec("openai/gpt-5.6").unwrap().capabilities();
        assert!(openai.supports_api(Api::ChatCompletions));
        assert!(openai.supports_api(Api::Responses));

        let deepseek = model_spec("deepseek/deepseek-v4-flash")
            .unwrap()
            .capabilities();
        assert!(deepseek.supports_api(Api::ChatCompletions));
        assert!(!deepseek.supports_api(Api::Responses));
    }

    /// Every registered model resolves to its own entry. Iterates the registry
    /// so a new `specs.yaml` entry is covered automatically.
    #[test]
    fn registered_models_resolve_to_their_own_entry() {
        assert!(!REGISTRY.specs.is_empty(), "the registry is empty");
        for (model_str, spec) in &REGISTRY.specs {
            let resolved = model_spec(model_str).unwrap();
            assert_eq!(resolved.model(), spec.model());
            assert_eq!(resolved.base_url(), spec.base_url());
            assert_eq!(
                resolved.capabilities().max_concurrent_requests(),
                spec.capabilities().max_concurrent_requests()
            );
        }
    }

    /// The leaf-level inheritance model entries rely on: the V4 pair restates
    /// one capability and keeps everything else from the DeepSeek namespace.
    #[test]
    fn model_entries_inherit_every_field_they_do_not_restate() {
        let flash = model_spec("deepseek/deepseek-v4-flash").unwrap();
        let pro = model_spec("deepseek/deepseek-v4-pro").unwrap();
        assert_eq!(flash.base_url(), pro.base_url());
        assert_eq!(flash.env_api_key(), pro.env_api_key());
        assert_eq!(flash.protocol(), pro.protocol());
        assert_eq!(
            flash.capabilities().supported_apis(),
            pro.capabilities().supported_apis()
        );
        // The divergence that motivated per-model entries: 5x the concurrency.
        assert_eq!(flash.capabilities().max_concurrent_requests(), Some(2500));
        assert_eq!(pro.capabilities().max_concurrent_requests(), Some(500));
    }

    /// The shipped roster declares no `*`, so every provider in it is closed:
    /// an unlisted name is an error, not a fallback. `openai/*` is in the list
    /// because a request may never name a wildcard directly.
    #[test]
    fn unregistered_models_are_rejected() {
        for model_str in ["openai/gpt-4o", "deepseek/deepseek-v5", "openai/*"] {
            let msg = model_spec(model_str).unwrap_err().to_string();
            assert!(msg.contains(model_str), "{msg}");
            assert!(msg.contains("no registered model spec"), "{msg}");
        }
    }

    /// A slash-containing name needs an entry or a `*` in its own namespace,
    /// and the registry has neither for `openai/org/`.
    #[test]
    fn unregistered_namespaces_are_rejected() {
        assert!(model_spec("openai/org/custom-model").is_err());
    }

    #[test]
    fn rejects_bare_model() {
        let msg = model_spec("gpt-5.6").unwrap_err().to_string();
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    #[test]
    fn rejects_unknown_provider() {
        let msg = model_spec("mistral/large").unwrap_err().to_string();
        assert!(msg.contains("mistral/large"), "{msg}");
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    /// A namespace key is merge material in the YAML and never reaches the
    /// table, so it cannot be routed to either.
    #[test]
    fn rejects_a_namespace_key() {
        assert!(model_spec("openai/").is_err());
    }

    /// Nothing but routable entries reaches the table: no namespace, every key
    /// carrying a provider segment, any `*` a whole final segment, and every
    /// row agreeing with the key it is filed under.
    ///
    /// The fixtures elsewhere assert the same over hand-written rosters. This
    /// asserts it over [`REGISTRY`], which is the table a caller actually
    /// routes against — a resolver that passed every other test and still
    /// built a bad table would fail here and only here.
    #[test]
    fn the_table_holds_only_routable_entries() {
        for (model_str, spec) in &REGISTRY.specs {
            assert!(
                !model_str.ends_with('/'),
                "namespace `{model_str}` is routable"
            );
            let (provider, name) = model_str
                .split_once('/')
                .unwrap_or_else(|| panic!("`{model_str}` names no provider"));
            assert_eq!(spec.provider(), provider, "`{model_str}`");
            assert!(!name.is_empty(), "`{model_str}` names no model");
            assert_eq!(
                spec.is_wildcard(),
                name.rsplit('/').next() == Some("*"),
                "`{model_str}`: wildcard flag disagrees with the key"
            );
            // A `*` is only ever the whole final segment.
            for segment in name.split('/') {
                assert!(
                    !segment.contains('*') || segment == "*",
                    "`{model_str}`: `{segment}` is neither a name nor a wildcard"
                );
            }
        }
    }
}
