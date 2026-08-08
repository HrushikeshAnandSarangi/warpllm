use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;

create_exception!(
    _warpllm,
    WarpLLMNativeError,
    pyo3::exceptions::PyException,
    "Raised by the native layer with a wire-format JSON message; \
     the Python wrapper translates it into typed exceptions."
);

#[pyfunction]
fn version() -> &'static str {
    warpllm::version()
}

async fn run_chat(
    client: Arc<warpllm::JsonClient>,
    request_json: String,
) -> Result<String, String> {
    client
        .chat_completions(&request_json)
        .await
        .map_err(|error| error.to_openai_json())
}

#[pyclass]
struct Client {
    inner: Arc<warpllm::JsonClient>,
}

#[pymethods]
impl Client {
    #[new]
    fn new(config_json: String) -> PyResult<Self> {
        let inner = warpllm::JsonClient::new(&config_json)
            .map_err(|error| WarpLLMNativeError::new_err(error.to_openai_json()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Blocks on the shared tokio runtime with the GIL released — no
    /// `asyncio.run` involved, so this works inside notebooks and scripts
    /// alike and reuses pooled connections across calls.
    fn chat_completions(&self, py: Python<'_>, request_json: String) -> PyResult<String> {
        let client = self.inner.clone();
        py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(run_chat(client, request_json))
                .map_err(WarpLLMNativeError::new_err)
        })
    }

    fn async_chat_completions<'p>(
        &self,
        py: Python<'p>,
        request_json: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            run_chat(client, request_json)
                .await
                .map_err(WarpLLMNativeError::new_err)
        })
    }
}

#[pymodule]
fn _warpllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Client>()?;
    m.add(
        "WarpLLMNativeError",
        m.py().get_type::<WarpLLMNativeError>(),
    )?;
    Ok(())
}
