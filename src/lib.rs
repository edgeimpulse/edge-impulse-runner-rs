mod error;
mod messages;
mod model;
mod types;

pub use error::EimError;
pub use messages::{HelloMessage, ClassifyMessage, ModelInfo, InferenceResponse, InferenceResult};
pub use model::EimModel;
pub use types::{ModelParameters, ProjectInfo, TimingInfo, BoundingBox};

#[cfg(test)]
mod tests;