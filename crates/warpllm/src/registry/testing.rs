//! Roster fixtures shared by the three modules' tests.
//!
//! Here rather than duplicated because the line numbers matter: [`PROVIDER`]
//! is nine lines and ends on its `models:` key, so a case that appends model
//! entries knows its first one lands on line 10 — which is what the sort
//! check's error message is asserted against.

use super::load::load;
use super::types::Registry;

/// One provider serving two APIs, with no models yet — the base most cases
/// diverge from. Cases append their own model entries, already indented six
/// spaces, so appended keys start at line 10.
pub(super) const PROVIDER: &str = "\
providers:
  demo:
    base_url: \"https://api.demo.test/v1\"
    env_api_key: DEMO_API_KEY
    protocol: openai_compat
    supported_apis:
      - chat_completions
      - responses
    models:
";

/// A second provider, complete with a model, sorted after `demo` so it can be
/// appended to a [`with`] roster without tripping the order check.
///
/// Written without the `\` line continuation the fixture above uses: that
/// escape swallows the following line's indentation, and this block's two
/// leading spaces are what make it a sibling of `demo` rather than a
/// top-level key.
pub(super) const OTHER_PROVIDER: &str = "  other:
    base_url: \"https://api.other.test\"
    env_api_key: OTHER_API_KEY
    protocol: openai_compat
    supported_apis:
      - chat_completions
    models:
      other/plain: {}
";

pub(super) fn with(models: &str) -> String {
    format!("{PROVIDER}{models}")
}

/// A roster expected to be valid AND clean, resolved. Anything the shipped file
/// has to satisfy, a fixture is held to as well — so a fixture cannot quietly
/// rely on something the roster itself would be rejected for.
pub(super) fn clean(yaml: &str) -> Registry {
    super::lint::check(yaml).unwrap_or_else(|e| panic!("{e}"));
    load(yaml).unwrap_or_else(|e| panic!("{e}"))
}

/// Sorted, because the tables are `HashMap`s and an assertion on their
/// contents must not depend on iteration order.
pub(super) fn keys(registry: &Registry) -> Vec<&str> {
    sorted(registry.models.keys())
}

pub(super) fn providers(registry: &Registry) -> Vec<&str> {
    sorted(registry.providers.keys())
}

fn sorted<'a>(keys: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut keys: Vec<&str> = keys.map(String::as_str).collect();
    keys.sort_unstable();
    keys
}
