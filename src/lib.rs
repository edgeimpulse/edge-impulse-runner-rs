mod error;
mod messages;
mod model;
pub mod types;

pub use error::EimError;
pub use messages::{InferenceResponse, InferenceResult};
pub use model::EimModel;
pub use types::ModelParameters;

#[cfg(test)]
mod tests;
