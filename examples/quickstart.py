"""One chat completion.

    OPENAI_API_KEY=sk-... python examples/quickstart.py

Model strings are `provider/model`. The prefix is required: the roster matches
the whole string, so a bare `gpt-5-nano` is rejected rather than guessed at.
"""

from warpllm import WarpLLM

client = WarpLLM()

completion = client.chat_completions(
    {
        "model": "openai/gpt-5-nano",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    }
)

print(completion["choices"][0]["message"]["content"])
