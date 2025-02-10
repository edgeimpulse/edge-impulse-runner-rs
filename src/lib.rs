//! # Edge Impulse Runner
//!
//! A Rust library for running Edge Impulse Linux models. This crate provides a safe
//! and ergonomic interface for loading and running machine learning models exported
//! from Edge Impulse.
//!
//! ## Features
//!
//! - Load and run Edge Impulse models (.eim files)
//! - Support for multiple sensor types (audio, accelerometer, camera, etc.)
//! - Handle both classification and object detection models
//! - Manage model lifecycle and IPC communication
//! - Continuous and one-shot inference modes
//!
//! ## Example
//!
//! ```no_run
//! use edge_impulse_runner::{EimModel, InferenceResult};
//!
//! // Create a new model instance
//! let mut model = EimModel::new("path/to/model.eim").unwrap();
//!
//! // Prepare your input features
//! let features = vec![0.1, 0.2, 0.3];
//!
//! // Run inference
//! let result = model.classify(features, None).unwrap();
//!
//! // Handle the results
//! match result.result {
//!     InferenceResult::Classification { classification } => {
//!         for (class, probability) in classification {
//!             println!("{}: {:.2}%", class, probability * 100.0);
//!         }
//!     },
//!     InferenceResult::ObjectDetection { bounding_boxes, .. } => {
//!         for bbox in bounding_boxes {
//!             println!("Found {} at ({}, {}) with confidence {:.2}%",
//!                 bbox.label, bbox.x, bbox.y, bbox.value * 100.0);
//!         }
//!     }
//! }
//! ```
//!
//! ## Architecture
//!
//! The library uses a client-server architecture where:
//! - The Edge Impulse model runs as a separate process
//! - Communication happens over Unix sockets using JSON messages
//! - The library handles process lifecycle and message serialization/deserialization
//!
//! ## Error Handling
//!
//! All operations that can fail return a `Result<T, EimError>`. The `EimError` enum
//! provides detailed error information for:
//! - Model loading failures
//! - Communication errors
//! - Invalid input data
//! - Model runtime errors
//!
//! ## Modules
//!
//! - `error`: Error types and handling
//! - `messages`: Communication protocol message definitions
//! - `model`: Core model management functionality
//! - `types`: Common types and parameters

mod error;
mod messages;
pub mod model;
pub mod types;

pub use error::EimError;
pub use messages::{InferenceResponse, InferenceResult};
pub use model::EimModel;
pub use types::ModelParameters;
pub use model::SensorType;

#[cfg(test)]
mod tests;
