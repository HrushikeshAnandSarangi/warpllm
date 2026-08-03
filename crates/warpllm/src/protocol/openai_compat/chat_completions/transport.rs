//! The HTTP binding for `POST /chat/completions`: the one place the
//! [`Api::ChatCompletions`](crate::Api) → URL path mapping physically lives,
//! since the module path spells the API's name rather than its route.

use std::time::Duration;

use crate::error::{Error, Result};
use crate::http::{network_error, read_response};
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};

/// What the upstream said. A non-2xx is NOT an [`Err`] here: which [`Error`] a
/// given status and body becomes is the caller's to decide: a provider may
/// envelope its errors differently from the protocol default, and deciding here
/// would mean `protocol` reaching into the conversion layer to find out, which
/// is the dependency this module exists without. `Err` is reserved for failures
/// nothing could reinterpret — the request never completing, or a 2xx body that
/// will not decode.
///
/// `large_enum_variant` is allowed rather than fixed: boxing the success
/// variant would add an allocation to every successful request to shrink a
/// value that is constructed once, moved once, and destructured immediately.
/// The lint's premise — many of these held at once — never happens.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Outcome {
    Ok(CreateChatCompletionResponse),
    /// The status, raw body, and header evidence, verbatim, for the caller
    /// to map.
    ///
    /// The headers travel with the body because this is the LAST place they
    /// exist: `Retry-After` and the upstream request id appear nowhere in
    /// any error envelope, so a caller handed only a status and a body
    /// cannot answer how long to wait or what to quote to the provider.
    Status {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
        request_id: Option<String>,
    },
}

pub(crate) async fn post(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateChatCompletionRequest,
) -> Result<Outcome> {
    let response = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(body)
        .send()
        .await
        .map_err(|e| network_error(provider, e))?;

    let parts = read_response(provider, response).await?;
    if !(200..300).contains(&parts.status) {
        return Ok(Outcome::Status {
            status: parts.status,
            body: parts.body,
            retry_after: parts.retry_after,
            request_id: parts.request_id,
        });
    }

    serde_json::from_str(&parts.body)
        .map(Outcome::Ok)
        .map_err(|e| Error::Decode {
            provider,
            message: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn post_to(server: &MockServer) -> Result<Outcome> {
        post(
            &reqwest::Client::new(),
            "demo",
            &server.uri(),
            "sk-demo",
            &CreateChatCompletionRequest::default(),
        )
        .await
    }

    /// The contract this module exists to state: a non-2xx is DATA, not an
    /// `Err`. Mapping it to an [`Error`] belongs to the caller, since a
    /// provider may envelope its errors differently from the protocol default.
    #[tokio::test]
    async fn a_non_2xx_comes_back_as_status_with_the_body_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .mount(&server)
            .await;

        match post_to(&server).await.unwrap() {
            Outcome::Status { status, body, .. } => {
                assert_eq!(status, 429);
                assert_eq!(body, "slow down", "the body must reach the caller unread");
            }
            Outcome::Ok(_) => panic!("a 429 decoded as success"),
        }
    }

    /// ...whereas a 2xx that will not decode is nobody's to reinterpret.
    #[tokio::test]
    async fn a_2xx_that_will_not_decode_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        assert!(matches!(
            post_to(&server).await,
            Err(Error::Decode {
                provider: "demo",
                ..
            })
        ));
    }

    /// The URL suffix and bearer scheme live only here, so they are only
    /// asserted here: the mock matches nothing else and would 404.
    #[tokio::test]
    async fn posts_to_chat_completions_with_a_bearer_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-demo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "demo",
                "choices": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        assert!(matches!(post_to(&server).await.unwrap(), Outcome::Ok(_)));
    }

    /// A trailing slash on the base URL must not double up in the path.
    #[tokio::test]
    async fn a_trailing_slash_on_the_base_url_is_trimmed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .expect(1)
            .mount(&server)
            .await;

        // Decode is the expected outcome; reaching the mock at all is the point.
        let err = post(
            &reqwest::Client::new(),
            "demo",
            &format!("{}/", server.uri()),
            "sk-demo",
            &CreateChatCompletionRequest::default(),
        )
        .await;
        assert!(matches!(err, Err(Error::Decode { .. })), "{err:?}");
    }
}
