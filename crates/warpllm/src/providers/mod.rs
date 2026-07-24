//! Provider-specific data. A provider module contributes one thing: its
//! model roster ([`crate::model::ModelSpec`] entries) under
//! `<provider>/models/`, which `crate::model::MODELS` stitches together.
//! The protocol-generic conversions and transport live in
//! `crate::protocol`.

pub(crate) mod deepseek;
pub(crate) mod openai;
