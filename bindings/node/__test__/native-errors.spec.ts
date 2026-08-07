import { expect, test } from 'vitest'

import { BadRequestError, WarpLLM } from '../dist/index.js'

test('Rust chooses the JavaScript exception class', async () => {
  const client = new WarpLLM()

  const error = await client
    .chatCompletions({ model: 'not-a-routable-model', messages: [] })
    .catch((caught: unknown) => caught)

  expect(error).toBeInstanceOf(BadRequestError)
  expect((error as BadRequestError).status).toBe(400)
})
