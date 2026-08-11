// Compile-only drift alarm: warpllm's reply shapes measured against OpenAI's
// own declarations, for both the whole completion and the streamed chunk.
//
// `openai` is a devDependency and an ORACLE, never a contract. Nothing here is
// re-exported, and the package is absent from the published dependency tree —
// warpllm's wire types are its own, and this file only asserts they still fit
// what the vendor says. The version is pinned exactly, so an upstream field
// arrives as a failure here rather than as a surprise in someone's stream.
//
// Two questions, asked separately:
//
//  1. Does everything OpenAI can emit FIT warpllm's shape? Assignability
//     answers that, and it is the one users feel: a field we typed too
//     narrowly makes their real traffic a type error. It is also what makes
//     "permissive superset" a checked claim rather than a description —
//     including the nulls, which is why every optional-and-nullable field is
//     `Option<Option<T>>` in Rust.
//  2. Does warpllm MODEL every field OpenAI models? Assignability cannot
//     answer that — an object with extra properties is still assignable — so
//     the key sets are compared directly. A field we never modelled still
//     reaches callers through `unknown_fields`, but it reaches them untyped.
//
// There are no exceptions to either. If one becomes necessary, it belongs
// here, spelled as a type: a deviation nobody can read is a deviation nobody
// weighed.
import type { ChatCompletion, ChatCompletionChunk, ChatCompletionMessage } from 'openai/resources/chat/completions'

import type {
  ChatCompletionMessageToolCallChunk,
  ChatCompletionResponseMessage,
  ChatCompletionStreamResponseDelta,
  Choice,
  CreateChatCompletionResponse,
  CreateChatCompletionStreamResponse,
  StreamChoice,
} from '../src-ts/generated/types.js'

// Fails to compile unless `T` is `never` — an empty array would satisfy
// `Missing[]` no matter what `Missing` held, which is why this is a constraint
// and not a value.
type Nothing<T extends never> = T

type Missing<Upstream, Ours> = Exclude<keyof Upstream, keyof Ours>

// ---------------------------------------------------------------------------
// 1. Anything OpenAI emits, warpllm holds
// ---------------------------------------------------------------------------

export const acceptsAnOpenAICompletion: CreateChatCompletionResponse = {} as ChatCompletion
export const acceptsAnOpenAIChunk: CreateChatCompletionStreamResponse = {} as ChatCompletionChunk

// ---------------------------------------------------------------------------
// 2. Anything OpenAI models, warpllm names
// ---------------------------------------------------------------------------

export type CompletionFieldsAreModelled = Nothing<
  Missing<ChatCompletion, CreateChatCompletionResponse>
>
export type CompletionChoiceFieldsAreModelled = Nothing<Missing<ChatCompletion.Choice, Choice>>
export type MessageFieldsAreModelled = Nothing<
  Missing<ChatCompletionMessage, ChatCompletionResponseMessage>
>

export type ChunkFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk, CreateChatCompletionStreamResponse>
>
export type ChunkChoiceFieldsAreModelled = Nothing<Missing<ChatCompletionChunk.Choice, StreamChoice>>
export type DeltaFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk.Choice.Delta, ChatCompletionStreamResponseDelta>
>
export type ToolCallFieldsAreModelled = Nothing<
  Missing<ChatCompletionChunk.Choice.Delta.ToolCall, ChatCompletionMessageToolCallChunk>
>
