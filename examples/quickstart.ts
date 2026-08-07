/**
 * One chat completion.
 *
 *   OPENAI_API_KEY=sk-... node --experimental-strip-types examples/quickstart.ts
 *
 * Model strings are `provider/model`. The prefix is required: the roster
 * matches the whole string, so a bare `gpt-5-nano` is rejected rather than
 * guessed at.
 */

import { WarpLLM } from '@warpllm/warpllm'

const client = new WarpLLM()

const completion = await client.chatCompletions({
  model: 'openai/gpt-5-nano',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' },
  ],
})

console.log(completion.choices[0].message.content)
