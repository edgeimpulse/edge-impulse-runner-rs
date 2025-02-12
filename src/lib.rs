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
//! use edge_impulse_runner::EimModel;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance
//!     let mut model = EimModel::new("path/to/model.eim")?;
//!
//!     // Get model information
//!     let params = model.parameters()?;
//!     println!("Model type: {}", params.model_type);
//!
//!     // Check sensor type
//!     match model.sensor_type()? {
//!         SensorType::Camera => println!("Camera model"),
//!         SensorType::Microphone => println!("Audio model"),
//!         SensorType::Accelerometer => println!("Motion model"),
//!         SensorType::Positional => println!("Position model"),
//!         SensorType::Other => println!("Other sensor type"),
//!     }
//!
//!     // Run inference with normalized features
//!     let raw_features = vec![128, 128, 128];  // Example raw values
//!     let features: Vec<f32> = raw_features.into_iter().map(|x| x as f32 / 255.0).collect();
//!     let result = model.classify(features, None)?;
//!
//!     // Handle the results based on model type
//!     match result.result {
//!         InferenceResult::Classification { classification } => {
//!             for (label, probability) in classification {
//!                 println!("{}: {:.2}", label, probability);
//!             }
//!         }
//!         InferenceResult::ObjectDetection { bounding_boxes, classification } => {
//!             for bbox in bounding_boxes {
//!                 println!("Found {} at ({}, {}) with confidence {:.2}",
//!                     bbox.label, bbox.x, bbox.y, bbox.value);
//!             }
//!             if !classification.is_empty() {
//!                 println!("\nOverall classification:");
//!                 for (label, prob) in classification {
//!                     println!("{}: {:.2}", label, prob);
//!                 }
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
pub use model::SensorType;
pub use types::ModelParameters;

#[cfg(test)]
mod tests;
