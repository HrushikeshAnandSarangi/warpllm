use std::sync::Arc;

use napi_derive::napi;

#[napi]
pub fn version() -> &'static str {
    warpllm::version()
}

/// Errors cross to JS as `Error` whose message is the wire-format JSON;
/// the TypeScript wrapper parses it into `WarpLLMError`.
fn wire_err(e: warpllm::Error) -> napi::Error {
    napi::Error::from_reason(e.to_openai_json())
}

#[napi]
pub struct Client {
    inner: Arc<warpllm::JsonClient>,
}

#[napi]
impl Client {
    #[napi(constructor)]
    pub fn new(config_json: String) -> napi::Result<Self> {
        let inner = warpllm::JsonClient::new(&config_json).map_err(wire_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[napi]
    pub async fn chat_completion(&self, request_json: String) -> napi::Result<String> {
        let client = self.inner.clone();
        client
            .chat_completion(&request_json)
            .await
            .map_err(wire_err)
    }
}
