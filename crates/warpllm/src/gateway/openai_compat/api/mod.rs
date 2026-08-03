//! One module per API surface this protocol serves, named for its
//! [`Api`](crate::Api) variant.
//!
//! Surfaces sit under `api/` and everything protocol-WIDE stays a sibling of
//! it — [`error`](super::error) most of all, since one error envelope is
//! shared by every endpoint here. A thing that belongs to one endpoint goes
//! in that endpoint's module; a thing true of the whole protocol never does.

pub(crate) mod chat_completions;
