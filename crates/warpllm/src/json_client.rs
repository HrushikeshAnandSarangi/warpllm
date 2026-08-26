//! The shared JSON boundary used by foreign-language bindings.
//!
//! PyO3 and napi-rs should expose Rust, not each maintain the same serde
//! adapter. Keeping that adapter here makes both native modules mechanical.

use crate::{
    ChatCompletionStream, Client, ClientConfig, CreateChatCompletionRequest, Error, Result,
    balancer::Balancer,
};

/// A [`Client`] whose inputs and outputs are JSON strings.
///
/// This is intentionally small: ownership and async-runtime integration stay
/// language-specific, while parsing, validation, dispatch, and serialization
/// happen once in the core.
pub struct JsonClient {
    inner: Client,
}

fn parse_request(request_json: &str) -> Result<CreateChatCompletionRequest> {
    serde_json::from_str(request_json).map_err(|error| Error::InvalidInput(error.to_string()))
}

impl JsonClient {
    pub fn new(config_json: &str) -> Result<Self> {
        let config: ClientConfig = serde_json::from_str(config_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        Ok(Self {
            inner: Client::new(config)?,
        })
    }

    pub async fn chat_completions(&self, request_json: &str) -> Result<String> {
        let request = parse_request(request_json)?;
        // A whole reply is the only thing this method's `String` can carry, so
        // a request for chunks is refused rather than served the wrong shape —
        // the same judgement [`Client::chat_completions`] makes, and now with
        // the same advice, since this boundary gained the method to point at.
        if request.stream == Some(true) {
            return Err(Error::InvalidInput(
                "stream: true asks for chunks; call chat_completions_stream".into(),
            ));
        }
        let response = self.inner.chat_completions(request).await?;
        serde_json::to_string(&response).map_err(|error| Error::Internal(error.to_string()))
    }

    /// The streaming counterpart: chunks as JSON strings, one per call to
    /// [`JsonChatStream::next`].
    ///
    /// `stream` is set on the caller's behalf, as the Rust entrypoint does —
    /// the method name states the intent, and a request contradicting it would
    /// have nothing useful to mean.
    ///
    /// The [`Result`] covers only what can fail before the first chunk. Once the
    /// stream is open, failures arrive as items on it.
    pub async fn chat_completions_stream(&self, request_json: &str) -> Result<JsonChatStream> {
        Ok(JsonChatStream {
            inner: self
                .inner
                .chat_completions_stream(parse_request(request_json)?)
                .await?,
        })
    }
}

/// A [`ChatCompletionStream`] whose chunks come out as JSON strings.
///
/// Owned outright — it borrows nothing from the [`JsonClient`] that opened it —
/// so a binding can hand it to its own language without lifetime plumbing.
///
/// Deliberately NOT an iterator of any kind. Rust has three (`Iterator`,
/// `Stream`, and whatever a caller hand-rolls), and the languages on the far
/// side have their own; committing this boundary to one of Rust's would make
/// every binding adapt away from it. An inherent `next` is what all of them can
/// build their own protocol on: `Symbol.asyncIterator` in Node, `__next__` and
/// `__anext__` in Python.
#[derive(Debug)]
pub struct JsonChatStream {
    inner: ChatCompletionStream,
}

impl JsonChatStream {
    /// The next chunk as serialized JSON, or `None` once the stream ends.
    ///
    /// An error item is terminal, inherited from the transport: whatever
    /// produced it also ended the stream, so the next call returns `None`.
    pub async fn next(&mut self) -> Option<Result<String>> {
        Some(self.inner.next().await?.and_then(|chunk| {
            serde_json::to_string(&chunk).map_err(|error| Error::Internal(error.to_string()))
        }))
    }
}

/// A load-balanced [`Client`] whose inputs and outputs are JSON strings.
///
/// Mirrors [`JsonClient`] but wraps a [`Balancer`](crate::balancer::Balancer)
/// that selects the next candidate on each request. The candidates JSON is a
/// list of `{"model": "provider/model", "weight": N}` objects.
pub struct JsonBalancedClient {
    client: Client,
    balancer: Balancer,
}

impl std::fmt::Debug for JsonBalancedClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonBalancedClient")
            .field("balancer", &self.balancer)
            .finish()
    }
}

impl JsonBalancedClient {
    pub fn new(config_json: &str, candidates_json: &str) -> Result<Self> {
        let config: ClientConfig = serde_json::from_str(config_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let raw: Vec<CandidateJson> = serde_json::from_str(candidates_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        if raw.is_empty() {
            return Err(Error::InvalidInput(
                "balanced client requires at least one candidate".into(),
            ));
        }
        let client = Client::new(config)?;
        let mut candidates = Vec::with_capacity(raw.len());
        for entry in raw {
            // Validated here rather than stored: the selected candidate's
            // model string is written back onto the request and the inner
            // client resolves it again through its own gates.
            client.fetch_model(&entry.model)?;
            candidates.push(crate::balancer::Candidate {
                model_str: entry.model,
                weight: entry.weight,
            });
        }
        Ok(Self {
            client,
            balancer: Balancer::new(candidates),
        })
    }

    pub async fn chat_completions(&self, request_json: &str) -> Result<String> {
        let mut request: CreateChatCompletionRequest = parse_request(request_json)?;
        if request.stream == Some(true) {
            return Err(Error::InvalidInput(
                "stream: true asks for chunks; call chat_completions_stream".into(),
            ));
        }
        request.model.clone_from(&self.balancer.select().model_str);
        let response = self.client.chat_completions(request).await?;
        serde_json::to_string(&response).map_err(|error| Error::Internal(error.to_string()))
    }

    pub async fn chat_completions_stream(&self, request_json: &str) -> Result<JsonChatStream> {
        let mut request: CreateChatCompletionRequest = parse_request(request_json)?;
        request.model.clone_from(&self.balancer.select().model_str);
        Ok(JsonChatStream {
            inner: self.client.chat_completions_stream(request).await?,
        })
    }
}

#[derive(serde::Deserialize)]
struct CandidateJson {
    model: String,
    weight: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every construction here reads the environment, so it holds temp-env's
    /// lock — see [`crate::credentials::with_env`].
    fn json_client(config_json: &str) -> Result<JsonClient> {
        crate::credentials::with_env(&[], || JsonClient::new(config_json))
    }

    #[test]
    fn invalid_config_is_a_core_error() {
        let error = json_client(r#"{"unknown": true}"#)
            .err()
            .expect("invalid configuration should fail");
        assert!(matches!(error, Error::InvalidInput(_)));
    }

    /// A declaration narrows the JSON boundary exactly as it narrows Rust's,
    /// which is what lets both bindings inherit the behaviour from here rather
    /// than each reimplementing it.
    #[tokio::test]
    async fn a_declaration_narrows_the_json_boundary_too() {
        let client = json_client(r#"{"providers":{"openai":{"api_key":"sk-x"}}}"#).unwrap();
        let error = client
            .chat_completions(r#"{"model":"deepseek/deepseek-v4-flash","messages":[]}"#)
            .await
            .expect_err("this client does not serve deepseek");
        assert!(
            matches!(&error, Error::ProviderNotDeclared { provider, .. } if *provider == "deepseek"),
            "{error:?}"
        );
        assert_eq!(error.to_openai().status, Some(400));
    }

    /// A misspelled provider fails where it was written, at construction —
    /// the bindings raise on the constructor rather than on a later request.
    #[test]
    fn an_unknown_declared_provider_is_a_core_error() {
        let error = json_client(r#"{"providers":{"openia":{}}}"#)
            .err()
            .expect("the roster has no `openia`");
        assert!(
            matches!(&error, Error::InvalidInput(m) if m.contains("openia")),
            "{error:?}"
        );
    }

    /// `chat_completions` returns one whole reply, which is a shape chunks do
    /// not fit in — so a request asking for them is refused and pointed at the
    /// method on this same boundary that serves them.
    ///
    /// Both language suites assert this from the other side. It is an
    /// `InvalidInput` rather than an unimplemented surface because the advice is
    /// now actionable here: a binding built on this boundary can reach
    /// `chat_completions_stream`, which it could not before.
    #[tokio::test]
    async fn stream_true_is_refused_and_names_the_streaming_method() {
        let client = json_client("{}").unwrap();
        let error = client
            .chat_completions(r#"{"model":"openai/gpt-5.6","messages":[],"stream":true}"#)
            .await
            .expect_err("a whole reply cannot carry chunks");
        match &error {
            Error::InvalidInput(message) => assert!(
                message.contains("chat_completions_stream"),
                "the refusal must name where to go instead: {message}"
            ),
            other => panic!("{other:?}"),
        }
        assert_eq!(error.to_openai().status, Some(400));
    }

    /// Both entrypoints reject a body that is not a request at all, before
    /// anything reaches the roster or the network.
    #[tokio::test]
    async fn a_malformed_request_body_is_invalid_input_on_both_entrypoints() {
        let client = json_client("{}").unwrap();
        assert!(matches!(
            client.chat_completions("not json").await,
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            client.chat_completions_stream("not json").await.err(),
            Some(Error::InvalidInput(_))
        ));
    }

    /// The roster is consulted before the network, on the streaming entrypoint
    /// too — an unregistered name never opens a socket.
    #[tokio::test]
    async fn an_unregistered_model_never_opens_a_stream() {
        let client = json_client("{}").unwrap();
        let error = client
            .chat_completions_stream(r#"{"model":"openai/not-a-model","messages":[]}"#)
            .await
            .expect_err("the roster is closed");
        assert!(
            matches!(&error, Error::InvalidModel { given } if given == "openai/not-a-model"),
            "{error:?}"
        );
    }

    #[test]
    fn json_balanced_empty_candidates_rejected() {
        let error = JsonBalancedClient::new("{}", "[]").unwrap_err();
        assert!(error.to_string().contains("at least one candidate"));
    }

    #[test]
    fn json_balanced_unknown_model_rejected() {
        let error =
            JsonBalancedClient::new("{}", r#"[{"model":"nope/nope","weight":1}]"#).unwrap_err();
        assert!(error.to_string().contains("no registered model"));
    }
}
