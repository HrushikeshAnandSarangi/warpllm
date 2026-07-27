//! Roster fixtures shared by the three modules' tests.
//!
//! Here rather than duplicated because the line numbers matter: `NAMESPACE` is
//! nine lines, so a case that appends entries knows its first one lands on line
//! 10 — which is what the sort check's error message is asserted against.

use super::load::load;
use super::types::Registry;

/// A namespace with two endpoints and no models — the base most cases diverge
/// from. Cases append their own entries, already indented two spaces and sorted
/// after `demo/`, so appended keys start at line 10.
pub(super) const NAMESPACE: &str = "\
specs:
  demo/:
    base_url: \"https://api.demo.test/v1\"
    env_api_key: DEMO_API_KEY
    protocol: openai_compat
    capabilities:
      supported_endpoints:
        - /chat/completions
        - /responses
";

/// Nesting at two depths, a namespace at each level, and a sibling outside the
/// inner one so inheritance can be shown NOT to leak sideways.
pub(super) const NESTED: &str = "\
specs:
  demo/:
    base_url: \"https://api.demo.test/v1\"
    env_api_key: DEMO_API_KEY
    protocol: openai_compat
    capabilities:
      supported_endpoints:
        - /chat/completions
  demo/plain: {}
  demo/v1/:
    capabilities:
      max_input_tokens: 128
  demo/v1/model-abc: {}
";

pub(super) fn with(entries: &str) -> String {
    format!("{NAMESPACE}{entries}")
}

/// A roster expected to be valid AND clean, resolved. Anything the shipped file
/// has to satisfy, a fixture is held to as well — so a fixture cannot quietly
/// rely on something the roster itself would be rejected for.
pub(super) fn clean(yaml: &str) -> Registry {
    super::lint::check(yaml).unwrap_or_else(|e| panic!("{e}"));
    load(yaml).unwrap_or_else(|e| panic!("{e}"))
}

/// Sorted, because a `Registry` is a `HashMap` and an assertion on its contents
/// must not depend on iteration order.
pub(super) fn keys(registry: &Registry) -> Vec<&str> {
    let mut keys: Vec<&str> = registry.specs.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}
