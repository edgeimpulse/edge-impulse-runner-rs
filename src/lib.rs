//! # Edge Impulse
//!
//! A Rust library for running inference with Edge Impulse Linux models (EIM) and uploading data to
//! Edge Impulse. This crate provides safe and easy-to-use interfaces for:
//! - Running machine learning models on Linux and MacOS
//! - Uploading training, testing and anomaly data to Edge Impulse projects
//!
//! ## Features
//!
//! ### Inference
//! - Run Edge Impulse models (.eim files) on Linux and MacOS
//! - Support for different model types:
//!   - Classification models
//!   - Object detection models
//! - Support for different sensor types:
//!   - Camera
//!   - Microphone
//!   - Accelerometer
//!   - Positional sensors
//! - Continuous classification mode support
//! - Debug output option
//!
//! ### Data Ingestion
//! - Upload data to Edge Impulse projects
//! - Support for multiple data categories:
//!   - Training data
//!   - Testing data
//!   - Anomaly data
//! - Handle various file formats:
//!   - Images (JPG, PNG)
//!   - Audio (WAV)
//!   - Video (MP4, AVI)
//!   - Sensor data (CBOR, JSON, CSV)
//!
//! ## Quick Start Examples
//!
//! ### Basic Classification
//! ```no_run
//! use edge_impulse_runner::{EimModel, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance
//!     let mut model = EimModel::new("path/to/model.eim")?;
//!
//!     // Prepare normalized features (e.g., image pixels, audio samples)
//!     let features: Vec<f32> = vec![0.1, 0.2, 0.3];
//!
//!     // Run inference
//!     let result = model.infer(features, None)?;
//!
//!     // Process results
//!     match result.result {
//!         InferenceResult::Classification { classification } => {
//!             println!("Classification: {:?}", classification);
//!         }
//!         InferenceResult::ObjectDetection {
//!             bounding_boxes,
//!             classification,
//!         } => {
//!             println!("Detected objects: {:?}", bounding_boxes);
//!             if !classification.is_empty() {
//!                 println!("Classification: {:?}", classification);
//!             }
//!         }
//!         InferenceResult::VisualAnomaly {
//!             visual_anomaly_grid,
//!             visual_anomaly_max,
//!             visual_anomaly_mean,
//!             anomaly,
//!         } => {
//!             let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) =
//!                 model.normalize_visual_anomaly(
//!                     anomaly,
//!                     visual_anomaly_max,
//!                     visual_anomaly_mean,
//!                     &visual_anomaly_grid.iter()
//!                         .map(|bbox| (bbox.value, bbox.x as u32, bbox.y as u32, bbox.width as u32, bbox.height as u32))
//!                         .collect::<Vec<_>>()
//!                 );
//!             println!("Anomaly score: {:.2}%", normalized_anomaly * 100.0);
//!             println!("Maximum score: {:.2}%", normalized_max * 100.0);
//!             println!("Mean score: {:.2}%", normalized_mean * 100.0);
//!             for (value, x, y, w, h) in normalized_regions {
//!                 println!("Region: score={:.2}%, x={}, y={}, width={}, height={}",
//!                     value * 100.0, x, y, w, h);
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Data Upload
//! ```no_run
//! use edge_impulse_runner::ingestion::{Category, Ingestion, UploadOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Create client with API key
//! let ingestion = Ingestion::new("your-api-key".to_string());
//!
//! // Upload a file
//! let result = ingestion
//!     .upload_file(
//!         "data.jpg",
//!         Category::Training,
//!         Some("label".to_string()),
//!         Some(UploadOptions {
//!             disallow_duplicates: true,
//!             add_date_id: true,
//!         }),
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ### Inference Protocol
//! The Edge Impulse Inference Runner uses a Unix socket-based IPC mechanism to communicate
//! with the model process. The protocol is JSON-based and follows a request-response pattern
//! for model initialization, classification requests, and error handling.
//!
//! ### Ingestion API
//! The ingestion module interfaces with the Edge Impulse Ingestion API over HTTPS, supporting
//! both data and file endpoints for uploading samples to Edge Impulse projects.
//!
//! ## Prerequisites
//!
//! Some functionality (particularly video capture) requires GStreamer to be installed:
//! - **macOS**: Install both runtime and development packages from gstreamer.freedesktop.org
//! - **Linux**: Install required packages (libgstreamer1.0-dev and related packages)
//!
//! ## Error Handling
//!
//! The crate uses the `EimError` type to provide detailed error information:
//! ```no_run
//! use edge_impulse_runner::{EimModel, EimError};
//!
//! // Match on model creation
//! match EimModel::new("model.eim") {
//!     Ok(mut model) => {
//!         // Match on classification
//!         match model.infer(vec![0.1, 0.2, 0.3], None) {
//!             Ok(result) => println!("Success!"),
//!             Err(EimError::InvalidInput(msg)) => println!("Invalid input: {}", msg),
//!             Err(e) => println!("Other error: {}", e),
//!         }
//!     },
//!     Err(e) => println!("Failed to load model: {}", e),
//! }
//! ```
//!
//! ## Modules
//!
//! - `error`: Error types and handling
//! - `inference`: Model management and inference functionality
//! - `ingestion`: Data upload and project management
//! - `types`: Common types and parameters

pub mod error;
pub mod inference;
pub mod ingestion;
pub mod types;
pub mod backends;

pub use inference::messages::{InferenceResponse, InferenceResult};

pub use error::EimError;
pub use inference::model::EdgeImpulseModel;
pub use types::BoundingBox;
pub use types::ModelParameters;
pub use types::ProjectInfo;
pub use types::SensorType;
pub use types::TimingInfo;
