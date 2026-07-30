//! Roster hygiene: everything true of a GOOD roster that resolving it does
//! not need in order to succeed.
//!
//! `cfg(test)`, which is the whole point. A roster that is merely untidy still
//! produces a correct table, so none of this belongs in the code a caller
//! links: the table is right whether or not two providers share an
//! environment variable. The tests below run [`check`] over the shipped file,
//! so CI is the gate and a PR is where it bites.

use std::collections::{BTreeMap, BTreeSet};

use super::load::{ancestry, load, parse};
use super::types::{ModelSpec, Registry};

/// Every policy below, first failure reported.
pub(super) fn check(yaml: &str) -> Result<(), String> {
    // Structure first: a real syntax or key error should be reported as
    // itself, not as whatever hygiene complaint it happens to also trip.
    let registry = load(yaml)?;
    let file = parse(yaml)?;
    if file.specs.is_empty() {
        return Err("specs: the registry names no models".into());
    }
    // Only `Value` keeps the order keys appear in the file; the typed pass has
    // already sorted them into a `BTreeMap` by the time it returns.
    let value: yaml_serde::Value = yaml_serde::from_str(yaml).map_err(|e| e.to_string())?;
    sorted(yaml, &value)?;
    reachable(&file.specs)?;
    let resolved = by_key(&registry);
    for (key, spec) in &resolved {
        routable(spec).map_err(|e| format!("`{key}`: {e}"))?;
    }
    env_api_keys(&resolved)
}

/// A `Registry` is a `HashMap`, so iterating it directly would make the
/// messages below name whichever member of a collision came up first. Every
/// check that can fail on one of two entries reads this instead, so the error
/// a contributor sees is the same one CI saw.
fn by_key(registry: &Registry) -> BTreeMap<&str, &ModelSpec> {
    registry
        .specs
        .iter()
        .map(|(key, spec)| (key.as_str(), spec))
        .collect()
}

/// `specs:` is kept in ascending key order. Byte order puts a namespace
/// immediately above the models that inherit from it, so a contributor adding
/// an entry has exactly one place to put it and two PRs adding different
/// providers do not collide.
///
/// Reads the raw text as well as the parsed value, only to put line numbers in
/// the message. The whole point of this check is that a contributor can fix it
/// without having to work out what "unsorted" means, so it names the key to
/// move, where to move it, and the ordering it is being judged against.
fn sorted(yaml: &str, value: &yaml_serde::Value) -> Result<(), String> {
    // A missing or malformed `specs:` is the typed pass's error to report.
    let Some(specs) = value.get("specs").and_then(|v| v.as_mapping()) else {
        return Ok(());
    };
    let keys: Vec<&str> = specs.keys().filter_map(|k| k.as_str()).collect();
    for pair in keys.windows(2) {
        // Equal keys never reach this: `parse`'s `Value` pass rejects a
        // duplicate before the order of two of them could be in question.
        let (above, below) = (pair[0], pair[1]);
        if above <= below {
            continue;
        }
        return Err(format!(
            "specs: `{below}`{} is out of order — move it above `{above}`{}.\n\
             Entries are kept in ascending BYTE order, which is what \
             `LC_ALL=C sort` gives and may differ from your editor's sort. \
             That ordering is what puts a namespace directly above the \
             models that inherit from it.",
            line_of(yaml, below),
            line_of(yaml, above),
        ));
    }
    Ok(())
}

/// ` (line N)` for a `specs:` key, or nothing when the text does not show it
/// plainly.
///
/// A scan rather than a span because `yaml_serde::Value` carries no positions.
/// Requiring the `:` immediately after the key is what keeps a search for
/// `openai/gpt-5.6` off the `openai/gpt-5.6-luna` line. It only ever decorates
/// a message, so a miss costs nothing but the number.
fn line_of(yaml: &str, key: &str) -> String {
    yaml.lines()
        .position(|line| {
            let line = line.trim_start();
            line.strip_prefix(key)
                .or_else(|| line.strip_prefix(&format!("{key:?}")))
                .is_some_and(|rest| rest.starts_with(':'))
        })
        .map_or_else(String::new, |i| format!(" (line {})", i + 1))
}

/// A namespace nothing inherits from is an unreachable provider: it holds a
/// spec no caller can ever route to.
fn reachable(specs: &BTreeMap<String, ModelSpec>) -> Result<(), String> {
    let mut reached = BTreeSet::new();
    for key in specs.keys().filter(|k| !k.ends_with('/')) {
        reached.extend(ancestry(key));
    }
    for key in specs.keys().filter(|k| k.ends_with('/')) {
        if !reached.contains(key.as_str()) {
            return Err(format!(
                "`{key}`: nothing routes through this namespace. Give it a \
                 model entry, or delete it."
            ));
        }
    }
    Ok(())
}

/// Whether a resolved entry could actually serve a request. Every one of these
/// resolves fine and then misbehaves at runtime, which is exactly why they are
/// worth a test.
fn routable(resolved: &ModelSpec) -> Result<(), String> {
    // Every `expect` here is discharged by `finish`, which is the only way a
    // `ModelSpec` reaches this function.
    let base_url = resolved
        .base_url
        .as_deref()
        .expect("finish requires base_url");
    if base_url.is_empty() {
        return Err("base_url is empty".into());
    }
    if base_url.ends_with('/') {
        return Err(format!(
            "base_url `{base_url}` ends with `/`; endpoints append their own \
             path, which would produce a doubled slash"
        ));
    }
    // Optional — an entry may name no variable at all — but an empty string is
    // never what anyone meant by that. Say so rather than resolve it to a
    // lookup of `""` at request time.
    if resolved.env_api_key.as_deref().is_some_and(str::is_empty) {
        return Err(
            "env_api_key is empty; omit the field entirely if this model takes \
             its key from the caller"
                .into(),
        );
    }
    let apis = resolved
        .capabilities
        .supported_apis
        .as_deref()
        .expect("finish requires supported_apis");
    if apis.is_empty() {
        return Err(
            "capabilities.supported_apis names no API, so nothing could ever \
             route here"
                .into(),
        );
    }
    // A model can serve fewer APIs than its protocol defines, never one the
    // protocol has never heard of. Naming an API that does not exist at all is
    // caught earlier still, by `Api`'s own deserialization.
    let protocol = resolved.protocol.expect("finish requires protocol");
    for api in apis {
        if !protocol.apis().contains(api) {
            return Err(format!(
                "`{api:?}` is not an API of this protocol, which defines {:?}",
                protocol.apis()
            ));
        }
    }
    Ok(())
}

/// Two providers sharing one environment variable would silently authenticate
/// one with the other's credentials, so a collision has to be resolved
/// deliberately rather than inherited by accident.
fn env_api_keys(resolved: &BTreeMap<&str, &ModelSpec>) -> Result<(), String> {
    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    for spec in resolved.values() {
        // Nothing to collide over when an entry names no variable.
        let Some(env_api_key) = spec.env_api_key.as_deref() else {
            continue;
        };
        let provider = spec.provider.as_deref().expect("finish sets provider");
        let claimed = owner.insert(env_api_key, provider);
        if let Some(other) = claimed
            && other != provider
        {
            return Err(format!(
                "`{env_api_key}` is claimed by both `{other}` and `{provider}`; \
                 every provider needs its own environment variable"
            ));
        }
    }
    Ok(())
}

/// Every case here RESOLVES fine — the table it produces is correct. What is
/// wrong with it is the roster, which is why this is the gate and `load` is
/// not.
#[cfg(test)]
mod tests {
    use super::super::testing::{NAMESPACE, clean, with};
    use super::*;

    /// Two providers sharing an environment variable would silently
    /// authenticate one with the other's credentials.
    #[test]
    fn colliding_env_api_keys_are_rejected() {
        let yaml = with(concat!(
            "  demo/plain: {}\n",
            "  other/:\n",
            "    base_url: \"https://api.other.test\"\n",
            "    env_api_key: DEMO_API_KEY\n",
            "    protocol: openai_compat\n",
            "    capabilities:\n",
            "      supported_apis:\n",
            "        - chat_completions\n",
            "  other/plain: {}\n",
        ));
        assert!(
            load(&yaml).is_ok(),
            "the roster is what is wrong, not the table"
        );
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("DEMO_API_KEY"), "{err}");
        assert!(err.contains("its own environment variable"), "{err}");
    }

    /// One provider using one variable across all its models is the normal
    /// case and must keep working.
    #[test]
    fn one_provider_may_reuse_its_own_env_api_key() {
        let registry = clean(&with("  demo/a: {}\n  demo/b: {}\n"));
        assert_eq!(
            registry.get("demo/a").unwrap().env_api_key(),
            Some("DEMO_API_KEY")
        );
        assert_eq!(
            registry.get("demo/b").unwrap().env_api_key(),
            Some("DEMO_API_KEY")
        );
    }

    /// Naming no variable is not a collision, so two providers may both do it.
    /// The check has to skip absent keys rather than treat `None` as a shared
    /// owner, which would reject a perfectly good roster.
    #[test]
    fn two_providers_may_both_name_no_env_api_key() {
        let bare = NAMESPACE
            .lines()
            .filter(|line| !line.trim_start().starts_with("env_api_key:"))
            .fold(String::new(), |acc, line| acc + line + "\n");
        clean(&format!(
            "{bare}{}",
            concat!(
                "  demo/plain: {}\n",
                "  other/:\n",
                "    base_url: \"https://api.other.test\"\n",
                "    protocol: openai_compat\n",
                "    capabilities:\n",
                "      supported_apis:\n",
                "        - chat_completions\n",
                "  other/plain: {}\n",
            )
        ));
    }

    /// An empty string is not the same as omitting the field, and resolving it
    /// would send a request-time lookup of `""`.
    #[test]
    fn an_empty_env_api_key_points_at_omitting_it() {
        let err = check(&with("  demo/plain:\n    env_api_key: \"\"\n")).unwrap_err();
        assert!(err.contains("env_api_key is empty"), "{err}");
        assert!(err.contains("omit the field entirely"), "{err}");
    }

    /// A misspelled API is a typo, and a closed vocabulary catches it before
    /// the roster even resolves — so this one does not reach the lint at all,
    /// let alone a live provider. Held here anyway: what matters is that the
    /// typo is rejected, not which gate does it.
    ///
    /// The message names the whole vocabulary, which is what lets a
    /// contributor fix the line without opening `protocol/types.rs`.
    #[test]
    fn an_api_outside_the_vocabulary_fails_to_load() {
        let yaml = with(
            "  demo/typo:\n    capabilities:\n      supported_apis:\n        - chat_completion\n",
        );
        let err = load(&yaml).unwrap_err();
        assert!(err.contains("unknown variant `chat_completion`"), "{err}");
        for known in ["chat_completions", "chat_completions_stream", "responses"] {
            assert!(err.contains(known), "vocabulary missing {known}: {err}");
        }
        // `check` loads first, so it reports the same thing rather than some
        // downstream hygiene complaint the typo happens to also trip.
        assert!(
            check(&yaml).unwrap_err().contains("unknown variant"),
            "{err}"
        );
    }

    /// `specs:` is kept in ascending key order so a new entry has one place to
    /// go and two PRs adding different providers do not collide. Purely a
    /// convention, which is exactly why it must not stop a table being built.
    #[test]
    fn out_of_order_keys_are_rejected() {
        let yaml = with("  demo/zeta: {}\n  demo/alpha: {}\n");
        assert!(load(&yaml).is_ok(), "key order cannot break resolution");
        let err = check(&yaml).unwrap_err();
        // Names the key to move, where to move it, and the ordering it is
        // judged against, so nobody has to work out what "unsorted" meant.
        assert!(err.contains("`demo/alpha` (line 11)"), "{err}");
        assert!(err.contains("move it above `demo/zeta` (line 10)"), "{err}");
        assert!(err.contains("ascending BYTE order"), "{err}");
    }

    /// The line number points at the misplaced key's OWN line even when a
    /// neighbour starts with the same bytes — the case a naive substring
    /// search gets wrong.
    #[test]
    fn the_order_error_points_at_the_right_line() {
        let err = check(&with("  demo/model-x: {}\n  demo/model: {}\n")).unwrap_err();
        assert!(err.contains("`demo/model` (line 11)"), "{err}");
        assert!(err.contains("above `demo/model-x` (line 10)"), "{err}");
    }

    #[test]
    fn an_unreachable_namespace_is_rejected() {
        let err = check(NAMESPACE).unwrap_err();
        assert!(
            err.contains("nothing routes through this namespace"),
            "{err}"
        );
    }

    /// A wildcard is a routable entry, so it makes its namespace reachable on
    /// its own — a provider can be served entirely by one `*`.
    #[test]
    fn a_wildcard_alone_makes_its_namespace_reachable() {
        clean(&with("  demo/*: {}\n"));
    }

    /// Each of these resolves to a spec that is well-formed and then cannot
    /// serve a request: no host, no credential, or a URL that would double its
    /// slash.
    #[test]
    fn unroutable_specs_are_rejected() {
        let err = check(&with("  demo/plain:\n    env_api_key: \"\"\n")).unwrap_err();
        assert!(err.contains("env_api_key is empty"), "{err}");

        let trailing = NAMESPACE.replace("api.demo.test/v1", "api.demo.test/v1/");
        let err = check(&format!("{trailing}  demo/plain: {{}}\n")).unwrap_err();
        assert!(err.contains("doubled slash"), "{err}");

        // An EMPTY list is a lint failure; omitting the field entirely fails
        // to load, since the merge then has nothing to settle it with.
        let empty = NAMESPACE.replace(
            "      supported_apis:\n        - chat_completions\n        - responses\n",
            "      supported_apis: []\n",
        );
        let err = check(&format!("{empty}  demo/plain: {{}}\n")).unwrap_err();
        assert!(err.contains("names no API"), "{err}");

        let err = check("specs: {}\n").unwrap_err();
        assert!(err.contains("names no models"), "{err}");
    }
}
