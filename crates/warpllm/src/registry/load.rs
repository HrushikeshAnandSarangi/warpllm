//! Reading `specs.yaml` and turning it into the two tables the lookup answers
//! from.
//!
//! Everything here fails on what leaves no usable spec: syntax, an unknown
//! field, a key that disagrees with where it sits, a required field nobody
//! set. Whether a roster that loads cleanly is any GOOD is `super::lint`'s
//! question, and it is a separate gate for that reason.
//!
//! The YAML schema lives here rather than on the types next door, which are
//! read surfaces. Keeping the two apart is what lets a `ProviderSpec` hold a
//! settled `base_url: String` while the file it came from is free to be
//! missing one — and be told so, by serde, with a line and a column.

use std::collections::HashMap;

use serde::Deserialize;

use super::types::{Capabilities, ModelSpec, ProviderSpec, Registry};
use crate::types::{Api, Protocol};

/// The whole roster: providers, each holding the models routable under it.
///
/// Hashed, like the tables it becomes. Nothing is looked up in these and
/// nothing reads them in order — they are built once and drained straight
/// into the registry — so a sorted map would buy a string comparison per
/// level on every insert and nothing else. What keeps the FILE readable is
/// `lint`'s ordering check, which reads the text rather than any map.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    providers: HashMap<String, ProviderEntry>,
}

/// One provider as written. Everything but `env_api_key` and `models` is
/// required: there is no inheritance and so nowhere else a value could come
/// from, which is what lets serde report a missing one against the line it is
/// missing from.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEntry {
    base_url: String,
    env_api_key: Option<String>,
    protocol: Protocol,
    supported_apis: Vec<Api>,
    /// `Option`, and defaulted, so that both ways of writing "no models yet" —
    /// omitting the key and leaving it empty — reach the lint, which says what
    /// is wrong with that in its own words. Neither is a load failure, because
    /// both leave a perfectly buildable pair of tables.
    #[serde(default)]
    models: Option<HashMap<String, ModelEntry>>,
}

/// One model as written: what it ships upstream if that differs from its key,
/// and whatever limits are published for it. Both optional, which is why the
/// overwhelmingly common entry is `{}`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    model: Option<String>,
    /// Names `blank` rather than relying on `#[serde(default)]`, which would
    /// need `Capabilities: Default` — the public constructor that type does
    /// not want.
    #[serde(default = "Capabilities::blank")]
    capabilities: Capabilities,
}

/// Reads the roster into the provider and model tables.
/// `Err` carries the message a contributor sees.
pub(super) fn load(yaml: &str) -> Result<Registry, String> {
    let file = parse(yaml)?;
    let mut registry = Registry::default();
    for (name, entry) in file.providers {
        validate_provider(&name)?;
        let ProviderEntry {
            base_url,
            env_api_key,
            protocol,
            supported_apis,
            models,
        } = entry;
        for (key, model) in models.unwrap_or_default() {
            let spec = build(&key, &name, model).map_err(|e| format!("`{key}`: {e}"))?;
            registry.models.insert(key, spec);
        }
        registry.providers.insert(
            name.clone(),
            ProviderSpec {
                name,
                base_url,
                env_api_key,
                protocol,
                supported_apis,
            },
        );
    }
    Ok(registry)
}

/// Two passes over a few kilobytes. Only `Value`'s own deserializer rejects
/// duplicate map keys — a `HashMap` silently keeps the last — and only
/// `Value` preserves the order keys appear in the file, which is what the
/// sort check reads. The typed pass is what attaches line and column to type
/// and unknown-field errors. Each owns what it reports best.
fn parse(yaml: &str) -> Result<RegistryFile, String> {
    let _duplicate_key_check: yaml_serde::Value =
        yaml_serde::from_str(yaml).map_err(|e| e.to_string())?;
    yaml_serde::from_str(yaml).map_err(|e| e.to_string())
}

/// One model entry, checked against the key it sits under. The key settles
/// the one thing the entry itself may not state: the name that ships upstream.
fn build(key: &str, provider: &str, entry: ModelEntry) -> Result<ModelSpec, String> {
    validate_model(key, provider)?;
    let name = key
        .rsplit_once('/')
        .expect("the prefix check found a `/`")
        .1;
    Ok(ModelSpec {
        provider: provider.to_string(),
        model: entry.model.unwrap_or_else(|| name.to_string()),
        capabilities: entry.capabilities,
    })
}

// -------------------------------------------------------------------- keys

/// A provider is one segment: it is the whole first part of a `model_str` and
/// holds nothing above or below it.
fn validate_provider(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("a provider name is empty".into());
    }
    if name.contains('/') {
        return Err(format!(
            "`{name}`: a provider name is one segment and carries no `/` — write \
             `{}`, and file its models under its own `models:` map",
            name.trim_end_matches('/')
        ));
    }
    Ok(())
}

/// Everything checkable about a model key: that it agrees with the provider
/// holding it, and that every segment of it is a name.
fn validate_model(key: &str, provider: &str) -> Result<(), String> {
    let Some(name) = key
        .strip_prefix(provider)
        .and_then(|rest| rest.strip_prefix('/'))
    else {
        return Err(format!(
            "a model key is the whole string a caller routes with, so one under \
             provider `{provider}` has to start with `{provider}/`"
        ));
    };
    if name.is_empty() {
        return Err(format!(
            "nothing follows the `{provider}/` prefix, so this key names no model"
        ));
    }
    // A key is read literally, every character of it: there are no patterns
    // to interpret, so the only thing a segment can be wrong about is being
    // absent.
    if name.split('/').any(str::is_empty) {
        return Err("an empty path segment".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testing::{OTHER_PROVIDER, PROVIDER, clean, keys, providers, with};
    use super::*;

    /// `PROVIDER` with one line dropped, for the cases that ask what happens
    /// when a required field is not there at all.
    fn without(field: &str) -> String {
        PROVIDER
            .lines()
            .filter(|line| !line.trim_start().starts_with(&format!("{field}:")))
            .fold(String::new(), |acc, line| acc + line + "\n")
    }

    // ------------------------------------------------------ the two tables

    /// The split itself: a provider lands in one table, its models in the
    /// other, and neither holds the other's rows.
    #[test]
    fn providers_and_models_land_in_their_own_tables() {
        let registry = clean(&with("      demo/one: {}\n      demo/two: {}\n"));
        assert_eq!(providers(&registry), vec!["demo"]);
        assert_eq!(keys(&registry), vec!["demo/one", "demo/two"]);
    }

    /// Transport is stated once for the provider and read from there, so two
    /// models of one provider are answered by the same row rather than by two
    /// copies of it.
    #[test]
    fn a_providers_transport_is_stored_once() {
        let registry = clean(&with("      demo/one: {}\n      demo/two: {}\n"));
        let provider = registry.providers.get("demo").unwrap();
        assert_eq!(provider.name(), "demo");
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
        assert_eq!(provider.env_api_key(), Some("DEMO_API_KEY"));
        assert_eq!(provider.protocol(), Protocol::OpenAiCompat);
        assert_eq!(
            provider.supported_apis(),
            [Api::ChatCompletions, Api::Responses]
        );
        for key in ["demo/one", "demo/two"] {
            assert_eq!(registry.models.get(key).unwrap().provider, "demo");
        }
    }

    /// A provider naming no environment variable is a valid roster, not an
    /// incomplete one: such a provider authenticates only with a key the
    /// caller supplies.
    #[test]
    fn a_provider_may_name_no_env_api_key() {
        let registry = clean(&format!(
            "{}      demo/plain: {{}}\n",
            without("env_api_key")
        ));
        let provider = registry.providers.get("demo").unwrap();
        assert_eq!(provider.env_api_key(), None);
        // Everything else still resolved, so this is an absence and not a
        // half-loaded entry.
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
    }

    // ---------------------------------------------------------- model rows

    /// An entry with nothing to say exists purely to make a name routable —
    /// the shape every GPT-5.6 entry uses.
    #[test]
    fn an_empty_entry_is_routable_and_records_no_limits() {
        let registry = clean(&with("      demo/plain: {}\n"));
        let spec = registry.models.get("demo/plain").unwrap();
        assert_eq!(spec.model(), "plain");
        assert_eq!(spec.capabilities().max_input_tokens(), None);
        assert_eq!(spec.capabilities().max_concurrent_requests(), None);
    }

    /// Limits are per model and nothing else is: the shipped V4 pair is two
    /// entries for exactly this reason.
    #[test]
    fn capabilities_are_read_per_model() {
        let registry = clean(&with(concat!(
            "      demo/fast:\n",
            "        capabilities:\n",
            "          max_concurrent_requests: 2500\n",
            "      demo/slow:\n",
            "        capabilities:\n",
            "          max_concurrent_requests: 500\n",
        )));
        let caps = |key| registry.models.get(key).unwrap().capabilities();
        assert_eq!(caps("demo/fast").max_concurrent_requests(), Some(2500));
        assert_eq!(caps("demo/slow").max_concurrent_requests(), Some(500));
        assert_eq!(caps("demo/fast").max_input_tokens(), None);
    }

    #[test]
    fn the_wire_name_defaults_to_the_keys_last_segment() {
        let registry = clean(&with("      demo/plain: {}\n"));
        assert_eq!(registry.models.get("demo/plain").unwrap().model(), "plain");
    }

    #[test]
    fn an_explicit_model_beats_the_keys_last_segment() {
        let registry = clean(&with(
            "      demo/chat:\n        model: demo-chat-20240101\n",
        ));
        assert_eq!(
            registry.models.get("demo/chat").unwrap().model(),
            "demo-chat-20240101"
        );
    }

    /// A slash-containing name is one name, not a grouping: nothing inherits
    /// through it, and the wire name is still the last segment.
    #[test]
    fn a_slash_containing_name_is_one_name() {
        let registry = clean(&with("      demo/org/custom: {}\n"));
        assert_eq!(
            registry.models.get("demo/org/custom").unwrap().model(),
            "custom"
        );
    }

    /// Every model row names a provider that is actually in the other table.
    /// The key check is what guarantees it, and this is the guarantee stated
    /// over a resolved roster.
    #[test]
    fn every_model_names_a_provider_that_exists() {
        let registry = clean(&format!(
            "{}{OTHER_PROVIDER}",
            with("      demo/plain: {}\n")
        ));
        for (key, spec) in &registry.models {
            let provider = registry
                .providers
                .get(&spec.provider)
                .unwrap_or_else(|| panic!("`{key}` names provider `{}`", spec.provider));
            assert_eq!(key.split('/').next(), Some(provider.name()));
        }
    }

    // ------------------------------------------------------- load failures
    //
    // Nothing below leaves a correct pair of tables to build, so each is
    // rejected.

    #[test]
    fn unknown_protocols_are_rejected() {
        let err = load(&with("").replace("openai_compat", "anthropic")).unwrap_err();
        assert!(err.contains("unknown variant"), "{err}");
        assert!(err.contains("openai_compat"), "{err}");
    }

    #[test]
    fn misspelled_fields_are_rejected() {
        let err = load(&with("").replace("base_url:", "base_urls:")).unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("base_url"), "{err}");
    }

    /// A model-level field is checked just as closely as a provider-level one.
    #[test]
    fn a_misspelled_capability_is_rejected() {
        let err = load(&with(
            "      demo/plain:\n        capabilities:\n          max_input_token: 128\n",
        ))
        .unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("max_input_tokens"), "{err}");
    }

    /// A `HashMap` would keep the last silently; the `Value` pass is what
    /// makes this an error.
    #[test]
    fn duplicate_keys_are_rejected() {
        let err = load(&with("      demo/plain: {}\n      demo/plain: {}\n")).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        assert!(err.contains("demo/plain"), "{err}");
    }

    /// The accessors read these without a fallback, and with no inheritance
    /// left there is nowhere else they could come from — so a provider missing
    /// one has no spec to build.
    #[test]
    fn a_missing_provider_field_is_rejected() {
        for field in ["base_url", "protocol", "supported_apis"] {
            let mut yaml = without(field);
            if field == "supported_apis" {
                // Its two list items outlive the key they belonged to.
                yaml = yaml
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("- "))
                    .fold(String::new(), |acc, line| acc + line + "\n");
            }
            let err = load(&format!("{yaml}      demo/plain: {{}}\n")).unwrap_err();
            assert!(
                err.contains("missing field") && err.contains(field),
                "removing {field} gave: {err}"
            );
        }
    }

    /// The key is the whole string a caller routes with. A bare name under a
    /// provider would resolve to nothing, and the message says what to write.
    #[test]
    fn a_model_key_missing_its_provider_prefix_is_rejected() {
        let err = load(&with("      plain: {}\n")).unwrap_err();
        assert!(err.contains("has to start with `demo/`"), "{err}");
    }

    /// Filed under the wrong provider it would be routable under a transport
    /// that is not its own.
    #[test]
    fn a_model_key_under_the_wrong_provider_is_rejected() {
        let err = load(&with("      other/plain: {}\n")).unwrap_err();
        assert!(err.contains("has to start with `demo/`"), "{err}");
    }

    #[test]
    fn a_key_that_names_no_model_is_rejected() {
        let err = load(&with("      demo/: {}\n")).unwrap_err();
        assert!(err.contains("names no model"), "{err}");
    }

    /// An empty segment is the one thing a key's insides can be wrong about,
    /// since every other character is read literally as part of a name.
    #[test]
    fn an_empty_path_segment_is_rejected() {
        let err = load(&with("      demo//custom: {}\n")).unwrap_err();
        assert!(err.contains("an empty path segment"), "{err}");
    }

    /// The old shape's `demo/:` key, and the message that migrates it.
    #[test]
    fn a_provider_name_carrying_a_slash_is_rejected() {
        let err = load(&PROVIDER.replace("  demo:", "  demo/:")).unwrap_err();
        assert!(err.contains("a provider name is one segment"), "{err}");
        assert!(err.contains("write `demo`"), "{err}");
    }
}
