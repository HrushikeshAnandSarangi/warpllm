import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { version } from '../index.js'

// vitest runs with cwd = bindings/node (npm scripts run from the package dir)
const workspaceVersion = readFileSync(join(process.cwd(), '../../Cargo.toml'), 'utf8')
  .match(/\[workspace\.package\][^[]*?version\s*=\s*"([^"]+)"/)![1]

test('version matches the workspace Cargo.toml (single source of truth)', () => {
  expect(version()).toBe(workspaceVersion)
})

test('package.json version matches the workspace Cargo.toml', () => {
  const pkg = JSON.parse(readFileSync(join(process.cwd(), 'package.json'), 'utf8'))
  expect(pkg.version).toBe(workspaceVersion)
})

// The async runtime bridge used to be covered here, through `echo`. It is now
// covered by `chat.spec.ts`, which awaits a real `chatCompletion` against a
// mock server — the same suspension point, exercised by the call that matters
// rather than by a placeholder that shipped in the published package.
