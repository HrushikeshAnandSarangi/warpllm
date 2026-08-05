use serde_json::json;

use crate::gateway::types::ProviderError;

/// Every way a warpllm call can fail, in ONE flat enum.
///
/// Flat on purpose. A provider failure is not a special kind of error that
/// needs unwrapping before it can be read — it is an error, and it sits
/// beside warpllm's own. That means one `match` reaches every case
/// (`Error::RateLimited(e)`), rather than a match on `Error` followed by a
/// second match on a nested kind. This enum is Rust's alone — the bindings
/// raise one class carrying OpenAI's error object — so a nested kind would
/// have been a shape only Rust callers ever saw.
///
/// The trade the flattening makes is that WHO failed is no longer visible in
/// the type — so [`Error::origin`] states it explicitly, and
/// [`Error::provider_error`] hands back the upstream's evidence for the
/// variants that have any. Both are total functions over the enum, which is
/// what stops the grouping from rotting as variants are added.
///
/// `non_exhaustive` because this exists to grow: providers invent failure
/// modes, and adding one must not break every downstream `match`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    // ---------------------------------------------- warpllm's own failures
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid model string '{given}': no registered model spec")]
    InvalidModel { given: String },
    #[error("{}", missing_api_key_message(provider, *env_var))]
    MissingApiKey {
        provider: &'static str,
        /// The variable to set, when the registry names one for this model.
        ///
        /// `None` means the entry declares no `env_api_key`, and since the
        /// environment is the only key source, such an entry cannot be
        /// authenticated at all — the remedy is a roster edit, not a shell one.
        /// One `missing_api_key` code covers both: the failure is the same,
        /// only the remedy differs, and the remedy rides in the message —
        /// which is the only place it can, since the OpenAI envelope the
        /// bindings receive has no field to put a variable name in.
        env_var: Option<&'static str>,
    },
    /// The provider was never reached, so it never had a chance to answer.
    #[error("network error talking to {provider}: {source}")]
    Network {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },
    /// The provider answered with a success status and a body warpllm could
    /// not read — a gap in warpllm's shapes, not a failure the provider
    /// reported.
    #[error("could not decode {provider} response: {message}")]
    Decode {
        provider: &'static str,
        message: String,
    },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// warpllm could not set itself up — building the HTTP client failed, for
    /// instance, which happens when the platform's TLS backend will not
    /// initialize.
    ///
    /// Nothing the caller passed is wrong, so this is deliberately NOT
    /// [`InvalidInput`](Self::InvalidInput): that spelling would tell someone
    /// to go fix a payload that was fine, and the server would answer 400 for
    /// what is squarely a 500.
    #[error("internal error: {0}")]
    Internal(String),

    // ----------------------------------------- the provider's own failures
    //
    // One variant per normalized failure, each carrying the same evidence.
    // Boxed so that eleven wide variants do not widen every `Result` in the
    // crate. What a caller should DO about one is deliberately absent:
    // warpllm has no retry loop and no fallback chain yet, so an opinion
    // about either would be an untested guess shipped as a public contract.
    /// Requests arrived faster than the account may send them.
    #[error("{0}")]
    RateLimited(Box<ProviderError>),
    /// The account is out of credit or past a spend cap. Distinct from
    /// [`RateLimited`](Self::RateLimited) because waiting does not fix it —
    /// somebody has to pay — and providers report both under a 429.
    #[error("{0}")]
    QuotaExceeded(Box<ProviderError>),
    /// The provider is up but shedding load (503, Anthropic-style 529).
    #[error("{0}")]
    Overloaded(Box<ProviderError>),
    /// An unattributed 5xx from the provider.
    #[error("{0}")]
    ServerError(Box<ProviderError>),
    /// The prompt (or prompt + completion) exceeds the model's window.
    #[error("{0}")]
    ContextLengthExceeded(Box<ProviderError>),
    /// The provider refused on safety grounds.
    #[error("{0}")]
    ContentFilter(Box<ProviderError>),
    /// The credential is missing or invalid. Distinct from
    /// [`MissingApiKey`](Self::MissingApiKey), which fires before any
    /// request goes out.
    #[error("{0}")]
    Authentication(Box<ProviderError>),
    /// The credential is valid but not entitled to what was asked for.
    /// Split from [`Authentication`](Self::Authentication) because the
    /// remedy differs: a wrong key is replaced, an unentitled one is
    /// granted access.
    #[error("{0}")]
    PermissionDenied(Box<ProviderError>),
    /// The provider does not serve the model that was requested. Distinct
    /// from [`InvalidModel`](Self::InvalidModel), which fires before any
    /// request goes out.
    #[error("{0}")]
    ModelNotFound(Box<ProviderError>),
    /// The provider rejected the request as malformed, or rejected a
    /// parameter it does not accept.
    #[error("{0}")]
    InvalidRequest(Box<ProviderError>),
    /// The provider failed and nothing in the envelope identified how.
    /// Carries no claim either way — read the [`ProviderError`] and decide.
    #[error("{0}")]
    Unknown(Box<ProviderError>),
}

/// Who a failure came from.
///
/// The distinction the flat enum gives up in its shape, handed back as a
/// value. It is the first question worth asking about any failure: an
/// upstream answered and said no, or the call never got that far.
///
/// Deliberately NOT `non_exhaustive`, unlike [`Error`] itself. Two arms is
/// the whole design — every failure is one side or the other — so a third
/// would be a change every consumer must see, not one to wave through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// warpllm, its configuration, or the call into it. The provider either
    /// was never reached or was never asked. Retrying the identical call
    /// changes nothing.
    Gateway,
    /// The provider answered, and its answer was a failure. A
    /// [`ProviderError`] is attached with everything it said.
    Provider,
}

impl Origin {
    /// The wire spelling, written once so every surface that reports an
    /// origin agrees. Today that is the gateway's HTTP envelope; the FFI
    /// form deliberately omits it, origin being warpllm's own vocabulary.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gateway => "gateway",
            Self::Provider => "provider",
        }
    }
}

/// A function rather than two `#[error]` strings because the REMEDY differs
/// while the failure does not. Naming a variable that no spec declares would
/// send someone off to set an environment variable nothing ever reads.
fn missing_api_key_message(provider: &str, env_var: Option<&str>) -> String {
    match env_var {
        Some(var) => format!("missing API key for {provider}: set the {var} environment variable"),
        None => format!(
            "missing API key for {provider}: its registry entry names no environment variable \
             to read one from"
        ),
    }
}

impl Error {
    /// Whether the provider said no, or the call failed before it could.
    ///
    /// Exhaustive by construction: a new variant does not compile until it
    /// is placed on one side or the other, which is what keeps this honest
    /// as the enum grows.
    pub fn origin(&self) -> Origin {
        match self {
            Error::InvalidInput(_)
            | Error::InvalidModel { .. }
            | Error::MissingApiKey { .. }
            // Network and Decode name a provider but are NOT its failures:
            // one never reached it, the other means it answered with a
            // success warpllm could not read. Neither is something the
            // upstream reported.
            | Error::Network { .. }
            | Error::Decode { .. }
            | Error::NotImplemented(_)
            | Error::Internal(_) => Origin::Gateway,
            Error::RateLimited(_)
            | Error::QuotaExceeded(_)
            | Error::Overloaded(_)
            | Error::ServerError(_)
            | Error::ContextLengthExceeded(_)
            | Error::ContentFilter(_)
            | Error::Authentication(_)
            | Error::PermissionDenied(_)
            | Error::ModelNotFound(_)
            | Error::InvalidRequest(_)
            | Error::Unknown(_) => Origin::Provider,
        }
    }

    /// Everything the upstream said, for the variants an upstream produced.
    ///
    /// `Some` exactly when [`origin`](Self::origin) is
    /// [`Origin::Provider`] — the two answer the same question, one as a
    /// tag and one as the evidence.
    pub fn provider_error(&self) -> Option<&ProviderError> {
        match self {
            Error::RateLimited(e)
            | Error::QuotaExceeded(e)
            | Error::Overloaded(e)
            | Error::ServerError(e)
            | Error::ContextLengthExceeded(e)
            | Error::ContentFilter(e)
            | Error::Authentication(e)
            | Error::PermissionDenied(e)
            | Error::ModelNotFound(e)
            | Error::InvalidRequest(e)
            | Error::Unknown(e) => Some(e),
            _ => None,
        }
    }

    /// The stable machine-readable slug for this variant, and a public
    /// contract: these strings cross the FFI boundary as `code` and the
    /// bindings dispatch on them.
    ///
    /// Flat, like the enum — one slug per variant, no second field to
    /// disambiguate. `invalid_request` is warpllm's own rejection and
    /// `provider_invalid_request` is the upstream's, because a caller
    /// fixing the first edits their call while the second may just need a
    /// different model.
    pub fn code(&self) -> &'static str {
        match self {
            Error::InvalidInput(_) | Error::InvalidModel { .. } => "invalid_request",
            Error::MissingApiKey { .. } => "missing_api_key",
            Error::Network { .. } => "connection_error",
            Error::Decode { .. } => "decode_error",
            Error::NotImplemented(_) => "not_implemented",
            Error::Internal(_) => "internal_error",
            Error::RateLimited(_) => "rate_limited",
            Error::QuotaExceeded(_) => "quota_exceeded",
            Error::Overloaded(_) => "overloaded",
            Error::ServerError(_) => "provider_server_error",
            Error::ContextLengthExceeded(_) => "context_length_exceeded",
            Error::ContentFilter(_) => "content_filter",
            Error::Authentication(_) => "authentication",
            Error::PermissionDenied(_) => "permission_denied",
            Error::ModelNotFound(_) => "model_not_found",
            Error::InvalidRequest(_) => "provider_invalid_request",
            Error::Unknown(_) => "provider_unknown",
        }
    }

    /// The HTTP status and OpenAI `type` an OpenAI-compatible surface answers
    /// with, so the in-process FFI form and the gateway's HTTP response
    /// cannot drift apart.
    ///
    /// A provider-driven failure is answered with the upstream's own status
    /// and family, so a genuinely unknown model surfaces as the provider's
    /// 404 rather than as a warpllm opinion about it. Matched first: [`Error`]
    /// is `non_exhaustive`, so a new provider variant must land here without
    /// a change over here.
    pub fn openai_status_and_type(&self) -> (u16, &str) {
        match (self.provider_error(), self) {
            (Some(upstream), _) => (
                upstream.status,
                upstream.error_type.as_deref().unwrap_or("api_error"),
            ),
            (None, Error::InvalidInput(_) | Error::InvalidModel { .. }) => {
                (400, "invalid_request_error")
            }
            (None, Error::MissingApiKey { .. }) => (401, "invalid_request_error"),
            (None, Error::Network { .. } | Error::Decode { .. }) => (502, "api_error"),
            (None, Error::NotImplemented(_)) => (501, "api_error"),
            // Unreachable today; a new gateway-side variant lands here rather
            // than failing to compile a caller that must keep serving.
            (None, _) => (500, "api_error"),
        }
    }

    /// The OpenAI error envelope, for the FFI boundary.
    ///
    /// Shaped like what the official SDKs build an error from: the `error`
    /// object exactly as OpenAI spells it, beside the two things an SDK would
    /// otherwise read off the HTTP response — which is the rule for what
    /// belongs here at all. `status` and `request_id` are on `APIError`;
    /// `retry_after` is not, so it is not here either.
    ///
    /// NOTHING here is warpllm's own vocabulary. The bindings are a surface
    /// callers program against directly, so a field named here is a field
    /// warpllm owes compatibility on — and the taxonomy is not settled enough
    /// to promise. [`code`](Self::code) and [`origin`](Self::origin) stay
    /// Rust-side, where changing them costs a recompile rather than a major
    /// version.
    ///
    /// `code` carries the PROVIDER's own slug when there is one, so a quota
    /// exhaustion stays `insufficient_quota` and cannot be mistaken for a
    /// plain rate limit — the distinction survives in OpenAI's vocabulary
    /// rather than needing warpllm's. Absent an upstream there is no
    /// provider slug to carry, so warpllm's own stands in; `code` is
    /// free-form in OpenAI's schema, and naming the failure is what it is for.
    pub fn to_openai_json(&self) -> String {
        let (status, error_type) = self.openai_status_and_type();
        let upstream = self.provider_error();
        json!({
            "status": status,
            "request_id": upstream.and_then(|e| e.request_id.as_deref()),
            "error": {
                "message": self.to_string(),
                "type": error_type,
                // warpllm does not model which parameter was at fault.
                "param": serde_json::Value::Null,
                "code": upstream
                    .and_then(|e| e.provider_code.as_deref())
                    .unwrap_or(self.code()),
            },
        })
        .to_string()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn wire(err: &Error) -> serde_json::Value {
        serde_json::from_str(&err.to_openai_json()).unwrap()
    }

    fn rate_limit() -> Error {
        Error::RateLimited(Box::new(ProviderError {
            provider: "openai",
            status: 429,
            message: "slow down".into(),
            error_type: Some("rate_limit_error".into()),
            provider_code: Some("rate_limit_exceeded".into()),
            retry_after: Some(Duration::from_secs(30)),
            request_id: Some("req-1".into()),
            raw_body: r#"{"error":{"message":"slow down"}}"#.into(),
        }))
    }

    #[test]
    fn a_provider_failure_renders_as_an_openai_error() {
        let v = wire(&rate_limit());
        assert_eq!(v["status"], 429);
        assert_eq!(v["request_id"], "req-1");
        assert_eq!(v["error"]["type"], "rate_limit_error");
        assert!(v["error"]["param"].is_null());
        assert!(v["error"]["message"].as_str().unwrap().contains("HTTP 429"));
        // The PROVIDER's slug, not warpllm's `rate_limited`.
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
    }

    /// The reason `code` carries the provider's slug rather than warpllm's.
    ///
    /// A quota exhaustion and a rate limit are both 429s and read alike, but
    /// no amount of backing off buys credit. OpenAI already spells the
    /// difference — `insufficient_quota` — so the distinction survives a
    /// pure-OpenAI envelope without warpllm's taxonomy crossing the boundary
    /// to carry it.
    #[test]
    fn quota_exhaustion_stays_distinguishable_from_a_rate_limit() {
        let quota = Error::QuotaExceeded(Box::new(ProviderError {
            provider: "openai",
            status: 429,
            message: "quota".into(),
            error_type: Some("insufficient_quota".into()),
            provider_code: Some("insufficient_quota".into()),
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }));
        let (quota, limit) = (wire(&quota), wire(&rate_limit()));

        assert_eq!(quota["status"], limit["status"], "both are 429s");
        assert_ne!(quota["error"]["code"], limit["error"]["code"]);
        assert_eq!(quota["error"]["code"], "insufficient_quota");
    }

    /// An unclassifiable failure still produces a well-formed envelope:
    /// `status`, `type` and `code` are always present, never absent or null.
    #[test]
    fn an_unknown_provider_failure_still_serializes() {
        let v = wire(&Error::Unknown(Box::new(ProviderError {
            provider: "demo",
            status: 402,
            message: "balance".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: "Insufficient Balance".into(),
        })));
        assert_eq!(v["status"], 402);
        assert_eq!(v["error"]["type"], "api_error");
        // Nothing upstream named it, so warpllm's slug stands in.
        assert_eq!(v["error"]["code"], "provider_unknown");
        assert!(v["request_id"].is_null());
    }

    /// The envelope is OpenAI's and only OpenAI's. warpllm's taxonomy is
    /// reachable in Rust and deliberately absent from the wire, so shipping
    /// a binding does not promise compatibility on a name still in flux.
    #[test]
    fn warpllms_own_vocabulary_never_crosses_the_boundary() {
        for error in [rate_limit(), Error::NotImplemented("streaming")] {
            let v = wire(&error);
            let envelope = v["error"].as_object().unwrap();
            let mut keys: Vec<_> = envelope.keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, ["code", "message", "param", "type"], "{v}");
            for warpllm_only in ["origin", "provider", "provider_code", "provider_raw"] {
                assert!(v.get(warpllm_only).is_none(), "{warpllm_only} in {v}");
            }
        }
    }

    /// A gateway-side failure has no upstream to borrow a status from, so
    /// the mapping has to supply one an OpenAI client can act on.
    #[test]
    fn gateway_failures_map_onto_openai_statuses() {
        let cases = [
            (
                Error::InvalidInput("x".into()),
                400,
                "invalid_request_error",
            ),
            (
                Error::MissingApiKey {
                    provider: "openai",
                    env_var: Some("OPENAI_API_KEY"),
                },
                401,
                "invalid_request_error",
            ),
            (Error::NotImplemented("streaming"), 501, "api_error"),
            (Error::Internal("tls".into()), 500, "api_error"),
        ];
        for (error, status, error_type) in cases {
            assert_eq!(error.openai_status_and_type(), (status, error_type));
        }
    }

    /// The whole point of the flat enum: a caller matches one level.
    #[test]
    fn a_provider_failure_matches_without_unwrapping_a_kind() {
        assert!(matches!(rate_limit(), Error::RateLimited(_)));
    }

    /// `origin` and `provider_error` answer the same question two ways, so
    /// they must never disagree — an evidence-free provider error, or an
    /// upstream payload on a gateway failure, would both be bugs.
    #[test]
    fn origin_and_provider_error_agree_on_every_variant() {
        let all = [
            Error::InvalidInput("x".into()),
            Error::InvalidModel { given: "x".into() },
            Error::MissingApiKey {
                provider: "openai",
                env_var: None,
            },
            Error::Decode {
                provider: "openai",
                message: "x".into(),
            },
            Error::NotImplemented("x"),
            Error::Internal("x".into()),
            rate_limit(),
            Error::QuotaExceeded(Box::new(provider_error())),
            Error::Overloaded(Box::new(provider_error())),
            Error::ServerError(Box::new(provider_error())),
            Error::ContextLengthExceeded(Box::new(provider_error())),
            Error::ContentFilter(Box::new(provider_error())),
            Error::Authentication(Box::new(provider_error())),
            Error::PermissionDenied(Box::new(provider_error())),
            Error::ModelNotFound(Box::new(provider_error())),
            Error::InvalidRequest(Box::new(provider_error())),
            Error::Unknown(Box::new(provider_error())),
        ];
        for error in &all {
            assert_eq!(
                error.origin() == Origin::Provider,
                error.provider_error().is_some(),
                "{error:?}"
            );
        }
    }

    fn provider_error() -> ProviderError {
        ProviderError {
            provider: "demo",
            status: 500,
            message: "m".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }
    }

    /// The slugs are a public contract the bindings dispatch on, and they
    /// must be UNIQUE now that one flat space holds warpllm's failures and
    /// the provider's: `invalid_request` is warpllm rejecting the call,
    /// `provider_invalid_request` is the upstream rejecting it.
    #[test]
    fn every_code_is_distinct() {
        let codes = [
            Error::InvalidInput("x".into()).code(),
            Error::MissingApiKey {
                provider: "openai",
                env_var: None,
            }
            .code(),
            Error::Decode {
                provider: "openai",
                message: "x".into(),
            }
            .code(),
            Error::NotImplemented("x").code(),
            Error::Internal("x".into()).code(),
            Error::RateLimited(Box::new(provider_error())).code(),
            Error::QuotaExceeded(Box::new(provider_error())).code(),
            Error::Overloaded(Box::new(provider_error())).code(),
            Error::ServerError(Box::new(provider_error())).code(),
            Error::ContextLengthExceeded(Box::new(provider_error())).code(),
            Error::ContentFilter(Box::new(provider_error())).code(),
            Error::Authentication(Box::new(provider_error())).code(),
            Error::PermissionDenied(Box::new(provider_error())).code(),
            Error::ModelNotFound(Box::new(provider_error())).code(),
            Error::InvalidRequest(Box::new(provider_error())).code(),
            Error::Unknown(Box::new(provider_error())).code(),
        ];
        let unique: std::collections::BTreeSet<_> = codes.iter().collect();
        assert_eq!(unique.len(), codes.len(), "{codes:?}");
        // The pair that forced the naming: both mean "bad request", and a
        // caller fixes them in different places.
        assert_eq!(Error::InvalidInput("x".into()).code(), "invalid_request");
        assert_eq!(
            Error::InvalidRequest(Box::new(provider_error())).code(),
            "provider_invalid_request"
        );
    }

    /// A quota exhaustion is not a rate limit. Both arrive as a 429 and
    /// read alike, but no amount of backing off buys credit — so they are
    /// separate variants with separate slugs.
    #[test]
    fn quota_exhaustion_is_a_distinct_variant_from_a_rate_limit() {
        let quota = Error::QuotaExceeded(Box::new(provider_error()));
        assert!(!matches!(quota, Error::RateLimited(_)));
        assert_ne!(quota.code(), rate_limit().code());
    }

    /// The variable to set is the whole remedy, and the envelope has no
    /// field for it — so it has to survive in the message instead.
    #[test]
    fn missing_key_wire_format() {
        let v = wire(&Error::MissingApiKey {
            provider: "openai",
            env_var: Some("OPENAI_API_KEY"),
        });
        assert_eq!(v["status"], 401);
        assert_eq!(v["error"]["code"], "missing_api_key");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("OPENAI_API_KEY")
        );
    }

    /// Network names a provider but is warpllm-side: the upstream was never
    /// reached, so there is nothing it said to report — no status of its
    /// own, and no `request_id`.
    #[test]
    fn a_network_failure_is_a_gateway_failure() {
        let error = Error::Decode {
            provider: "openai",
            message: "bad json".into(),
        };
        assert_eq!(error.origin(), Origin::Gateway);
        assert!(error.provider_error().is_none());
        let v = wire(&error);
        assert_eq!(v["status"], 502);
        assert_eq!(v["error"]["type"], "api_error");
        assert!(v["request_id"].is_null());
    }

    /// A setup failure is warpllm's, not the caller's. Spelled out because
    /// the tempting variant is `InvalidInput` — which reads as "your request
    /// was wrong" and lands a 500-class failure on a 400.
    #[test]
    fn a_setup_failure_blames_the_gateway_and_not_the_caller() {
        let error = Error::Internal("tls backend unavailable".into());
        assert_eq!(error.origin(), Origin::Gateway);
        assert!(error.provider_error().is_none());
        assert_ne!(error.code(), Error::InvalidInput("x".into()).code());
        let v = wire(&error);
        assert_eq!(v["error"]["code"], "internal_error");
        assert_eq!(v["status"], 500, "a setup failure is not a 400");
    }

    #[test]
    fn not_implemented_wire_format() {
        let v = wire(&Error::NotImplemented("streaming"));
        assert_eq!(v["status"], 501);
        assert_eq!(v["error"]["code"], "not_implemented");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("streaming")
        );
    }
}
