use crate::openai_common::{openai_completion_body, request};
use warpllm::{Client, ClientConfig, Error};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these scenarios run inside one test
/// body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn deepseek_key_resolves_per_provider() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("DEEPSEEK_API_KEY", Some("sk-deepseek-env"), || {
        runtime.block_on(async {
            // 1. DeepSeek requests use DEEPSEEK_API_KEY as bearer.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-deepseek-env"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                base_url: Some(server.uri()),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completion(request("deepseek/deepseek-v4-flash"))
                .await
                .unwrap();
        });
    });

    temp_env::with_vars(
        [
            ("OPENAI_API_KEY", Some("sk-openai-env")),
            ("DEEPSEEK_API_KEY", None),
        ],
        || {
            runtime.block_on(async {
                // 2. An OpenAI key must not satisfy DeepSeek: the missing
                //    key errors at request time, naming DeepSeek's env var.
                let client = Client::new(ClientConfig::default()).unwrap();
                let err = client
                    .chat_completion(request("deepseek/deepseek-v4-flash"))
                    .await
                    .unwrap_err();
                match err {
                    Error::MissingApiKey { provider, env_var } => {
                        assert_eq!(provider, "deepseek");
                        assert_eq!(env_var, "DEEPSEEK_API_KEY");
                    }
                    other => panic!("expected MissingApiKey, got {other:?}"),
                }
            });
        },
    );
}
