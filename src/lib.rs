mod error;
mod messages;
pub mod types;
mod model;

pub use error::EimError;
pub use model::EimModel;
pub use types::ModelParameters;
pub use messages::{InferenceResult, InferenceResponse};

#[cfg(test)]
mod tests;