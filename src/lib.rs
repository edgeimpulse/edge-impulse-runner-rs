//! # Edge Impulse Runner
//!
//! A Rust library for running Edge Impulse Linux models. This crate provides a safe
//! and ergonomic interface for loading and running machine learning models exported
//! from Edge Impulse.
//!
//! ## Features
//!
//! - Load and run Edge Impulse models (.eim files)
//! - Support for multiple sensor types (audio, camera, etc.)
//! - Handle both classification and object detection models
//! - Manage model lifecycle and IPC communication
//! - Continuous and one-shot inference modes
//!
//! ## Example
//!
//! ```no_run
//! use edge_impulse_runner::{EimModel, SensorType, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance
//!     let mut model = EimModel::new("path/to/model.eim")?;
//!
//!     // Check the sensor type
//!     match model.sensor_type()? {
//!         SensorType::Camera => println!("Camera model"),
//!         SensorType::Microphone => println!("Audio model"),
//!         SensorType::Accelerometer => println!("Motion model"),
//!         SensorType::Positional => println!("Position model"),
//!         SensorType::Unknown => println!("Unknown sensor type"),
//!     }
//!
//!     // Prepare your input data as a vector of f32 values
//!     let features: Vec<f32> = vec![/* your input data */];
//!
//!     // Run inference
//!     let result = model.classify(features, None)?;
//!
//!     // Process the results
//!     match result.result {
//!         InferenceResult::Classification { classification } => {
//!             for (label, probability) in classification {
//!                 println!("{}: {:.2}%", label, probability * 100.0);
//!             }
//!         }
//!         InferenceResult::ObjectDetection { bounding_boxes, classification } => {
//!             println!("Detected objects:");
//!             for bbox in bounding_boxes {
//!                 println!("{} ({:.2}%) at ({}, {}, {}, {})",
//!                     bbox.label, bbox.value * 100.0,
//!                     bbox.x, bbox.y, bbox.width, bbox.height);
//!             }
//!         }
//!     }
//!
//!     Ok(())
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
//! - `inference`: Core model management and inference functionality
//!   - `messages`: Communication protocol message definitions
//!   - `model`: Model implementation and runtime management
//! - `ingestion`: Data ingestion and preprocessing (coming soon)
//! - `types`: Common types and parameters

pub mod error;
pub mod inference;
pub mod ingestion;
pub mod types;

pub use inference::messages::{InferenceResponse, InferenceResult};
pub use inference::EimModel;

pub use error::EimError;
pub use types::BoundingBox;
pub use types::ModelParameters;
pub use types::ProjectInfo;
pub use types::SensorType;
pub use types::TimingInfo;

#[cfg(test)]
mod tests;
