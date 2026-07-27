//! Reading `specs.yaml` and resolving it into the table the lookup answers
//! from.
//!
//! Everything here fails on what leaves no usable spec: syntax, unknown
//! fields, key structure, a required field no ancestor ever answered. Whether
//! a roster that resolves cleanly is any GOOD is `super::lint`'s question,
//! and it is a separate gate for that reason.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::types::{Capabilities, ModelSpec, Registry};

/// The whole roster. `specs` is a MAP keyed by `model_str` path, so a key is
/// the string a caller routes with and nothing inside an entry can disagree
/// with where that entry sits.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RegistryFile {
    pub(super) specs: BTreeMap<String, ModelSpec>,
}

/// The segment that makes a key serve every name under its namespace.
pub(super) const WILDCARD: &str = "*";

impl ModelSpec {
    /// Merges `over` on top of `self`, leaf by leaf: `over` wins wherever it
    /// speaks and what is accumulated holds everywhere else.
    fn apply(&mut self, over: &ModelSpec) {
        // Destructured with no `..` on purpose. A field added to `ModelSpec`
        // fails to compile here (E0027) rather than silently never inheriting
        // from its namespace, which is the one way this merge could go wrong
        // without anything complaining.
        let ModelSpec {
            // The key settles both of these, not the merge — see `finish`.
            provider: _,
            wildcard: _,
            model,
            base_url,
            env_api_key,
            protocol,
            capabilities,
        } = over;
        overlay(&mut self.model, model);
        overlay(&mut self.base_url, base_url);
        overlay(&mut self.env_api_key, env_api_key);
        overlay(&mut self.protocol, protocol);
        self.capabilities.apply(capabilities);
    }

    /// Settles the three things the key decides, then refuses to hand back a
    /// spec with an open question left in it.
    ///
    /// This is the whole reason the accessors can read these `Option`s
    /// without a fallback: a roster that would produce a spec missing any of
    /// these fails to load here, so no such spec reaches the table. Whether
    /// the values are any GOOD is `super::lint`'s question.
    fn finish(mut self, provider: &str, name: &str) -> Result<ModelSpec, String> {
        self.provider = Some(provider.to_string());
        self.wildcard = name == WILDCARD;
        if self.model.is_none() {
            // For a wildcard this stores `*`, which is not a name anything
            // ships: `wire_model` reads the request instead.
            self.model = Some(name.to_string());
        }
        // `env_api_key` is deliberately absent from this list: a model may
        // authenticate only with a key the caller supplies, so naming no
        // variable is a valid roster, not an incomplete one.
        for (field, present) in [
            ("base_url", self.base_url.is_some()),
            ("protocol", self.protocol.is_some()),
            (
                "capabilities.supported_endpoints",
                self.capabilities.supported_endpoints.is_some(),
            ),
        ] {
            if !present {
                return Err(format!("{field} is missing"));
            }
        }
        Ok(self)
    }
}

impl Capabilities {
    fn apply(&mut self, over: &Capabilities) {
        // Exhaustive for the same reason as `ModelSpec::apply`.
        let Capabilities {
            supported_endpoints,
            max_input_tokens,
            max_output_tokens,
            max_concurrent_requests,
        } = over;
        // Lists REPLACE rather than extend: a model must be able to serve
        // FEWER endpoints than its namespace, which appending could never say.
        overlay(&mut self.supported_endpoints, supported_endpoints);
        overlay(&mut self.max_input_tokens, max_input_tokens);
        overlay(&mut self.max_output_tokens, max_output_tokens);
        overlay(&mut self.max_concurrent_requests, max_concurrent_requests);
    }
}

/// One field of a merge: the overlay wins if it speaks, and silence leaves
/// what was already accumulated alone. Silence is never a reset.
fn overlay<T: Clone>(into: &mut Option<T>, over: &Option<T>) {
    if over.is_some() {
        *into = over.clone();
    }
}

/// Reads the roster and resolves every key against its ancestors.
/// `Err` carries the message a contributor sees.
pub(super) fn load(yaml: &str) -> Result<Registry, String> {
    let file = parse(yaml)?;
    // Validate every key before resolving any, so walking an ancestry can
    // never reach a namespace that was never declared.
    for (key, entry) in &file.specs {
        validate(key, entry, &file.specs).map_err(|e| format!("`{key}`: {e}"))?;
    }

    let mut registry = Registry::default();
    for (key, entry) in &file.specs {
        if key.ends_with('/') {
            continue; // a namespace is merge material, never routable itself
        }
        let mut merged = ModelSpec::blank();
        for ancestor in ancestry(key) {
            merged.apply(&file.specs[ancestor]);
        }
        merged.apply(entry);

        let name = key.rsplit_once('/').expect("validated to contain `/`").1;
        let provider = key.split('/').next().expect("split yields one item");
        let spec = merged
            .finish(provider, name)
            .map_err(|e| format!("`{key}`: {e}"))?;
        registry.specs.insert(key.clone(), spec);
    }
    Ok(registry)
}

/// Two passes over a few kilobytes. Only `Value`'s own deserializer rejects
/// duplicate map keys — a `BTreeMap` silently keeps the last — and only
/// `Value` preserves the order keys appear in the file, which is what the
/// sort check reads. The typed pass is what attaches line and column to type
/// and unknown-field errors. Each owns what it reports best.
pub(super) fn parse(yaml: &str) -> Result<RegistryFile, String> {
    let _duplicate_key_check: yaml_serde::Value =
        yaml_serde::from_str(yaml).map_err(|e| e.to_string())?;
    yaml_serde::from_str(yaml).map_err(|e| e.to_string())
}

// -------------------------------------------------------------------- keys

/// Everything checkable about one key in isolation, plus that its namespace
/// exists. Run over every key before any resolution.
fn validate(
    key: &str,
    entry: &ModelSpec,
    specs: &BTreeMap<String, ModelSpec>,
) -> Result<(), String> {
    if key.is_empty() {
        return Err("the key is empty".into());
    }
    if !key.contains('/') {
        return Err(format!(
            "no `/` in the key; a namespace ends with one, so write `{key}/`"
        ));
    }
    let body = key.strip_suffix('/').unwrap_or(key);
    let segments: Vec<&str> = body.split('/').collect();
    let last = segments.len() - 1;
    for (position, segment) in segments.iter().enumerate() {
        if segment.is_empty() {
            return Err("an empty path segment".into());
        }
        // A `*` is a whole final segment of a routable key, or nothing. There
        // is no pattern MATCHING — no `gpt-*` prefixes, no `*` standing in for
        // a grouping level, no wildcard namespace — so anything else carrying
        // one is a typo, and saying so beats registering a model literally
        // named `fast*`.
        let wildcard_slot = position == last && !key.ends_with('/');
        if segment.contains(WILDCARD) && !(wildcard_slot && *segment == WILDCARD) {
            return Err(format!(
                "`{segment}` uses `*` somewhere it means nothing. A `*` is only ever \
                 a whole final segment of a routable key, as in `{}/*`, where it \
                 serves every name directly under that namespace.",
                segments[0]
            ));
        }
    }
    if let Some(namespace) = parent(key)
        && !specs.contains_key(namespace)
    {
        return Err(format!("its namespace `{namespace}` has no entry"));
    }
    if entry.model.is_some() && (key.ends_with('/') || segments[last] == WILDCARD) {
        return Err(
            "`model` names one thing that ships upstream, so it belongs on an entry \
             that is one model — not on a namespace, and not on a `*`, which takes \
             its upstream name from whatever the caller asked for"
                .into(),
        );
    }
    Ok(())
}

/// The namespace one level up, or `None` at the top. Stripping a trailing
/// `/` first is what makes a namespace look for ITS parent, not itself.
pub(super) fn parent(key: &str) -> Option<&str> {
    let body = key.strip_suffix('/').unwrap_or(key);
    body.rfind('/').map(|i| &body[..=i])
}

/// Every namespace above `key`, outermost first — the order they merge in.
pub(super) fn ancestry(key: &str) -> Vec<&str> {
    let mut chain = Vec::new();
    let mut current = key;
    while let Some(namespace) = parent(current) {
        chain.push(namespace);
        current = namespace;
    }
    chain.reverse();
    chain
}

#[cfg(test)]
mod tests {
    use super::super::testing::{NAMESPACE, NESTED, clean, keys, with};
    use super::*;
    use crate::protocol::Protocol;

    // ---------------------------------------------------------- path algebra

    #[test]
    fn parent_walks_one_level_up() {
        assert_eq!(parent("openai/gpt-5.6"), Some("openai/"));
        assert_eq!(parent("deepseek/v1/model-abc"), Some("deepseek/v1/"));
    }

    /// A namespace has to find the one ABOVE it rather than matching itself,
    /// which is what stripping the trailing `/` first buys — and why walking
    /// an ancestry terminates instead of spinning.
    #[test]
    fn parent_of_a_namespace_is_the_one_above_it() {
        assert_eq!(parent("deepseek/v1/"), Some("deepseek/"));
        assert_eq!(parent("openai/"), None);
    }

    /// The ORDER is the contract. `load` merges in exactly this sequence, so
    /// reversing it would let an outer namespace overwrite the inner one that
    /// is meant to refine it.
    #[test]
    fn ancestry_is_outermost_first() {
        assert_eq!(ancestry("a/b/c/model"), vec!["a/", "a/b/", "a/b/c/"]);
        assert_eq!(ancestry("openai/gpt-5.6"), vec!["openai/"]);
    }

    /// A key never appears in its own ancestry: `load` applies the entry
    /// separately, after the chain, so listing it here would merge it twice.
    #[test]
    fn ancestry_excludes_the_key_itself() {
        assert_eq!(ancestry("deepseek/v1/"), vec!["deepseek/"]);
        assert!(ancestry("openai/").is_empty());
    }

    // ----------------------------------------------------------------- merge

    #[test]
    fn apply_lets_the_override_win_where_it_speaks() {
        let mut merged = ModelSpec {
            base_url: Some("https://base.test".into()),
            env_api_key: Some("BASE_KEY".into()),
            protocol: Some(Protocol::OpenAiCompat),
            ..ModelSpec::blank()
        };
        merged.apply(&ModelSpec {
            env_api_key: Some("OWN_KEY".into()),
            ..ModelSpec::blank()
        });
        assert_eq!(merged.env_api_key.as_deref(), Some("OWN_KEY"));
        // Silence is not a reset: what it did not mention still stands.
        assert_eq!(merged.base_url.as_deref(), Some("https://base.test"));
        assert_eq!(merged.protocol, Some(Protocol::OpenAiCompat));
    }

    /// What every `{}` entry in the registry rests on — those exist only to
    /// make a name routable, so an override with nothing to say must change
    /// nothing at all.
    #[test]
    fn an_empty_override_changes_nothing() {
        let base = ModelSpec {
            base_url: Some("https://base.test".into()),
            capabilities: Capabilities {
                max_input_tokens: Some(128),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        };
        let mut merged = base.clone();
        merged.apply(&ModelSpec::blank());
        assert_eq!(merged.base_url, base.base_url);
        assert_eq!(merged.capabilities.max_input_tokens, Some(128));
    }

    /// Lists REPLACE. A model has to be able to serve FEWER endpoints than
    /// its namespace, which appending could never express.
    #[test]
    fn endpoint_lists_replace_rather_than_extend() {
        let mut merged = ModelSpec {
            capabilities: Capabilities {
                supported_endpoints: Some(vec![
                    "/chat/completions".to_string(),
                    "/responses".to_string(),
                ]),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        };
        merged.apply(&ModelSpec {
            capabilities: Capabilities {
                supported_endpoints: Some(vec!["/responses".to_string()]),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        });
        assert_eq!(merged.capabilities.supported_endpoints(), ["/responses"]);
    }

    /// Capabilities merge per field, never wholesale. The shipped V4 pair
    /// depends on it: each restates only `max_concurrent_requests` and keeps
    /// its namespace's endpoints.
    #[test]
    fn capabilities_merge_field_by_field() {
        let mut merged = ModelSpec {
            capabilities: Capabilities {
                supported_endpoints: Some(vec!["/chat/completions".to_string()]),
                max_input_tokens: Some(128),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        };
        merged.apply(&ModelSpec {
            capabilities: Capabilities {
                max_concurrent_requests: Some(2500),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        });
        let caps = &merged.capabilities;
        assert_eq!(caps.supported_endpoints(), ["/chat/completions"]);
        assert_eq!(caps.max_input_tokens, Some(128));
        assert_eq!(caps.max_concurrent_requests, Some(2500));
    }

    /// Applying the chain in `ancestry` order means the nearest ancestor wins
    /// and a farther one still fills what nobody nearer answered.
    #[test]
    fn the_nearest_ancestor_wins() {
        let mut merged = ModelSpec::blank();
        for step in [
            ModelSpec {
                env_api_key: Some("OUTER".into()),
                base_url: Some("https://outer.test".into()),
                ..ModelSpec::blank()
            },
            ModelSpec {
                env_api_key: Some("INNER".into()),
                ..ModelSpec::blank()
            },
        ] {
            merged.apply(&step);
        }
        assert_eq!(merged.env_api_key.as_deref(), Some("INNER"));
        assert_eq!(merged.base_url.as_deref(), Some("https://outer.test"));
    }

    // ---------------------------------------------------------------- finish

    /// Everything `finish` demands, so each case below can remove exactly one
    /// field and nothing else.
    fn complete() -> ModelSpec {
        ModelSpec {
            base_url: Some("https://api.demo.test/v1".into()),
            env_api_key: Some("DEMO_API_KEY".into()),
            protocol: Some(Protocol::OpenAiCompat),
            capabilities: Capabilities {
                supported_endpoints: Some(vec!["/chat/completions".to_string()]),
                ..Capabilities::blank()
            },
            ..ModelSpec::blank()
        }
    }

    /// The key decides all three, which is what stops an entry from
    /// disagreeing with where it sits.
    #[test]
    fn finish_settles_provider_and_model_from_the_key() {
        let spec = complete().finish("demo", "widget").unwrap();
        assert_eq!(spec.provider(), "demo");
        assert_eq!(spec.model(), "widget");
        assert!(!spec.is_wildcard());
    }

    #[test]
    fn an_explicit_model_beats_the_keys_last_segment() {
        let spec = ModelSpec {
            model: Some("upstream-name".into()),
            ..complete()
        }
        .finish("demo", "alias")
        .unwrap();
        assert_eq!(spec.model(), "upstream-name");
    }

    /// A `*` last segment is what makes an entry a wildcard, and nothing else
    /// does — the flag is never written in the YAML.
    #[test]
    fn a_star_segment_is_what_makes_an_entry_a_wildcard() {
        let spec = complete().finish("demo", "*").unwrap();
        assert!(spec.is_wildcard());
        assert_eq!(spec.model(), "*", "a wildcard pins no upstream name");
        assert_eq!(
            spec.wire_model("demo/anything"),
            "anything",
            "the request supplies the name a wildcard cannot"
        );
    }

    /// Every field the accessors later read without a fallback is demanded
    /// here, by name. This is what makes `required` in `types.rs` unreachable
    /// rather than merely optimistic.
    #[test]
    fn finish_names_the_missing_field() {
        for (field, spec) in [
            (
                "base_url",
                ModelSpec {
                    base_url: None,
                    ..complete()
                },
            ),
            (
                "protocol",
                ModelSpec {
                    protocol: None,
                    ..complete()
                },
            ),
            (
                "capabilities.supported_endpoints",
                ModelSpec {
                    capabilities: Capabilities::blank(),
                    ..complete()
                },
            ),
        ] {
            assert_eq!(
                spec.finish("demo", "widget").unwrap_err(),
                format!("{field} is missing")
            );
        }
    }

    // -------------------------------------------------- resolving a roster

    /// Namespaces are merge material. They carry specs but no caller can
    /// route to one, so none of them may reach the table.
    #[test]
    fn only_routable_models_reach_the_table() {
        assert_eq!(
            keys(&load(NESTED).unwrap()),
            vec!["demo/plain", "demo/v1/model-abc"]
        );
    }

    /// Resolution runs through EVERY level, which is what `ancestry`'s order
    /// exists for: the deep model takes the outer namespace's transport and
    /// the inner one's capability, while its sibling outside gets neither.
    #[test]
    fn a_nested_model_inherits_from_every_level() {
        let registry = load(NESTED).unwrap();
        let deep = registry.get("demo/v1/model-abc").unwrap();
        assert_eq!(deep.base_url(), "https://api.demo.test/v1");
        assert_eq!(deep.env_api_key(), Some("DEMO_API_KEY"));
        assert_eq!(
            deep.capabilities().supported_endpoints(),
            ["/chat/completions"]
        );
        assert_eq!(deep.capabilities().max_input_tokens(), Some(128));
        assert_eq!(
            registry
                .get("demo/plain")
                .unwrap()
                .capabilities()
                .max_input_tokens(),
            None,
            "an inner namespace leaked to a sibling outside it"
        );
    }

    #[test]
    fn a_model_inherits_every_field_it_does_not_restate() {
        let registry = clean(&with(
            "  demo/fast:\n    capabilities:\n      max_concurrent_requests: 2500\n",
        ));
        let spec = registry.get("demo/fast").unwrap();
        assert_eq!(spec.base_url(), "https://api.demo.test/v1");
        assert_eq!(spec.env_api_key(), Some("DEMO_API_KEY"));
        assert_eq!(
            spec.capabilities().supported_endpoints(),
            ["/chat/completions", "/responses"]
        );
        assert_eq!(spec.capabilities().max_concurrent_requests(), Some(2500));
    }

    /// An entry with no differences at all exists purely to make a name
    /// routable — the shape every GPT-5.6 entry uses.
    #[test]
    fn an_empty_entry_is_routable_and_inherits_everything() {
        let registry = clean(&with("  demo/plain: {}\n"));
        let spec = registry.get("demo/plain").unwrap();
        assert_eq!(spec.base_url(), "https://api.demo.test/v1");
        assert_eq!(spec.model(), "plain");
        assert_eq!(spec.capabilities().max_concurrent_requests(), None);
    }

    /// Lists replace: a model serving fewer endpoints than its namespace is the
    /// case appending could never express.
    #[test]
    fn a_models_endpoint_list_replaces_rather_than_extends() {
        let registry = clean(&with(
            "  demo/chat-only:\n    capabilities:\n      supported_endpoints:\n        - /chat/completions\n",
        ));
        let caps = registry.get("demo/chat-only").unwrap().capabilities();
        assert_eq!(caps.supported_endpoints(), ["/chat/completions"]);
        assert!(!caps.supports_endpoint("/responses"));
    }

    /// Three levels: the middle namespace wins over the outer, and the leaf
    /// over both, while an untouched field falls all the way through.
    #[test]
    fn nested_namespaces_merge_outermost_first() {
        let registry = clean(&with(concat!(
            "  demo/v1/:\n",
            "    env_api_key: DEMO_V1_KEY\n",
            "    capabilities:\n",
            "      max_input_tokens: 1000\n",
            "  demo/v1/model-abc:\n",
            "    capabilities:\n",
            "      max_input_tokens: 2000\n",
        )));
        let spec = registry.get("demo/v1/model-abc").unwrap();
        assert_eq!(spec.base_url(), "https://api.demo.test/v1");
        assert_eq!(spec.env_api_key(), Some("DEMO_V1_KEY"));
        assert_eq!(spec.capabilities().max_input_tokens(), Some(2000));
    }

    /// Intermediate segments are grouping levels; only the last is the name.
    #[test]
    fn the_wire_name_defaults_to_the_last_segment() {
        let registry = clean(&with("  demo/v1/: {}\n  demo/v1/model-abc: {}\n"));
        assert_eq!(
            registry.get("demo/v1/model-abc").unwrap().model(),
            "model-abc"
        );
    }

    #[test]
    fn an_explicit_model_field_overrides_the_last_segment() {
        let registry = clean(&with("  demo/chat:\n    model: demo-chat-20240101\n"));
        assert_eq!(
            registry.get("demo/chat").unwrap().model(),
            "demo-chat-20240101"
        );
    }

    /// A slash-containing name is expressible, but only by registering the
    /// namespace it sits under — which also makes the wire name the last
    /// segment.
    #[test]
    fn slash_containing_names_resolve_through_a_registered_namespace() {
        let registry = clean(&with("  demo/org/: {}\n  demo/org/custom: {}\n"));
        assert_eq!(registry.get("demo/org/custom").unwrap().model(), "custom");
    }

    /// A namespace is merge material, never something that reaches the table.
    #[test]
    fn namespaces_are_not_routable() {
        assert_eq!(
            keys(&clean(&with("  demo/plain: {}\n"))),
            vec!["demo/plain"]
        );
    }

    /// `env_api_key` is the one transport field a roster may leave out
    /// entirely: such a model authenticates only with a key the caller
    /// supplies, so it must LOAD rather than be rejected as incomplete.
    #[test]
    fn an_entry_may_name_no_env_api_key() {
        let bare = NAMESPACE
            .lines()
            .filter(|line| !line.trim_start().starts_with("env_api_key:"))
            .fold(String::new(), |acc, line| acc + line + "\n");
        let registry = clean(&format!("{bare}  demo/plain: {{}}\n"));
        let spec = registry.get("demo/plain").unwrap();
        assert_eq!(spec.env_api_key(), None);
        // Everything else still resolved, so this is an absence and not a
        // half-loaded entry.
        assert_eq!(spec.base_url(), "https://api.demo.test/v1");
        assert_eq!(spec.model(), "plain");
    }

    /// It inherits like any other field, so a model under a namespace that
    /// names one keeps it — omitting it on the model is not opting out.
    #[test]
    fn an_env_api_key_is_inherited_like_any_other_field() {
        let registry = clean(&with("  demo/plain: {}\n"));
        assert_eq!(
            registry.get("demo/plain").unwrap().env_api_key(),
            Some("DEMO_API_KEY")
        );
    }

    /// A wildcard resolves like any other routable entry, inheriting its
    /// namespace — it is the MATCHING that is special, not the resolution.
    #[test]
    fn a_wildcard_entry_inherits_its_namespace() {
        let registry = load(&with("  demo/*: {}\n")).unwrap();
        let spec = registry.get("demo/*").unwrap();
        assert!(spec.is_wildcard());
        assert_eq!(spec.provider(), "demo");
        assert_eq!(spec.base_url(), "https://api.demo.test/v1");
        assert_eq!(spec.env_api_key(), Some("DEMO_API_KEY"));
        assert_eq!(
            spec.capabilities().supported_endpoints(),
            ["/chat/completions", "/responses"]
        );
    }

    /// No resolved row can contradict the key it is filed under, and none can
    /// reach a caller with a field the accessors read unconditionally still
    /// unanswered.
    #[test]
    fn every_spec_agrees_with_its_key() {
        for (key, spec) in &load(NESTED).unwrap().specs {
            assert_eq!(spec.provider(), key.split('/').next().unwrap(), "`{key}`");
            assert_eq!(spec.model(), key.rsplit_once('/').unwrap().1, "`{key}`");
            // Reading these at all is the assertion: each panics if unanswered.
            assert!(!spec.base_url().is_empty(), "`{key}`");
            assert!(spec.env_api_key() != Some(""), "`{key}`");
            assert!(
                !spec.capabilities().supported_endpoints().is_empty(),
                "`{key}`"
            );
            let _ = spec.protocol();
        }
    }

    // ------------------------------------------------------- load failures
    //
    // Nothing below leaves a correct table to build, so each is rejected.

    #[test]
    fn unknown_protocols_are_rejected() {
        let err = load(&NAMESPACE.replace("openai_compat", "anthropic")).unwrap_err();
        assert!(err.contains("unknown variant"), "{err}");
        assert!(err.contains("openai_compat"), "{err}");
    }

    #[test]
    fn misspelled_fields_are_rejected() {
        let err = load(&NAMESPACE.replace("base_url:", "base_urls:")).unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("base_url"), "{err}");
    }

    /// A `BTreeMap` would keep the last silently; the `Value` pass is what
    /// makes this an error.
    #[test]
    fn duplicate_keys_are_rejected() {
        let err = load(&with("  demo/plain: {}\n  demo/plain: {}\n")).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        assert!(err.contains("demo/plain"), "{err}");
    }

    /// An orphan key is what would force a lookup to walk a chain instead of
    /// answering directly.
    #[test]
    fn an_orphan_key_is_rejected() {
        let err = load(&with("  demo/v1/model-abc: {}\n")).unwrap_err();
        assert!(err.contains("namespace `demo/v1/` has no entry"), "{err}");
    }

    /// `*` is a whole final segment or nothing. A prefix pattern, a `*`
    /// standing in for a grouping level, and a wildcard NAMESPACE are all
    /// typos rather than syntax, because nothing would ever read them as
    /// patterns.
    #[test]
    fn a_star_anywhere_but_a_final_segment_is_rejected() {
        for key in ["demo/*/turbo", "demo/fast*", "demo/*x", "demo/*/"] {
            let err = load(&with(&format!("  {key}: {{}}\n"))).unwrap_err();
            assert!(
                err.contains("uses `*` somewhere it means nothing"),
                "{key}: {err}"
            );
        }
    }

    /// A bare `*` with no provider in front of it would be a roster-wide
    /// catch-all, which is not a thing: a wildcard is always opted into by
    /// one namespace.
    #[test]
    fn a_wildcard_needs_a_namespace_in_front_of_it() {
        let err = load(&with("  \"*\": {}\n")).unwrap_err();
        assert!(err.contains("no `/` in the key"), "{err}");
    }

    #[test]
    fn a_model_field_on_a_namespace_is_rejected() {
        let err =
            load(&NAMESPACE.replace("  demo/:\n", "  demo/:\n    model: pinned\n")).unwrap_err();
        assert!(err.contains("not on a namespace"), "{err}");
    }

    /// A wildcard takes its upstream name from the request, so pinning one
    /// would quietly route every name under the namespace to a single model.
    #[test]
    fn a_model_field_on_a_wildcard_is_rejected() {
        let err = load(&with("  demo/*:\n    model: pinned\n")).unwrap_err();
        assert!(err.contains("not on a `*`"), "{err}");
    }

    /// The accessors read these unconditionally, so an entry that inherits
    /// none of them has no spec to hand back.
    #[test]
    fn a_missing_required_field_is_rejected() {
        for (missing, field) in [
            ("base_url", "base_url is missing"),
            ("protocol", "protocol is missing"),
        ] {
            let stripped: String = NAMESPACE
                .lines()
                .filter(|line| !line.trim_start().starts_with(&format!("{missing}:")))
                .fold(String::new(), |acc, line| acc + line + "\n");
            let err = load(&format!("{stripped}  demo/plain: {{}}\n")).unwrap_err();
            assert!(err.contains(field), "removing {missing} gave: {err}");
        }

        // The list is nested, so drop the whole `capabilities:` block.
        let bare = NAMESPACE
            .split("    capabilities:")
            .next()
            .expect("split yields one item");
        let err = load(&format!("{bare}  demo/plain: {{}}\n")).unwrap_err();
        assert!(
            err.contains("capabilities.supported_endpoints is missing"),
            "{err}"
        );
    }
}
