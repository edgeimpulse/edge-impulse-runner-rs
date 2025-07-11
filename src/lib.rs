//! # Edge Impulse Runner
//!
//! A Rust library for running inference with Edge Impulse models and uploading data to
//! Edge Impulse projects. This crate provides safe and easy-to-use interfaces for:
//! - Running machine learning models on Linux and macOS
//! - Uploading training, testing and anomaly data to Edge Impulse projects
//!
//! ## Inference Modes
//!
//! The crate supports two inference modes:
//!
//! ### FFI Mode (Default - Recommended)
//! - Direct FFI calls to the Edge Impulse C++ SDK
//! - Models are compiled into the binary
//! - **Superior performance**: Faster startup and inference times
//! - **No external dependencies**: No need for model files on filesystem
//! - **Production ready**: This is the recommended mode for all new applications
//!
//! ### EIM Mode (Legacy - Backward Compatibility)
//! - Run Edge Impulse models (.eim files) using binary communication
//! - Requires model files to be present on the filesystem
//! - **Performance penalty**: Slower due to IPC overhead
//! - **Legacy support**: Maintained for backward compatibility
//! - **Development friendly**: Useful for rapid prototyping and testing
//!
//! ## Model Support
//!
//! - **Classification models**: Multi-class and binary classification
//! - **Object detection models**: Bounding box detection with labels
//! - **Anomaly detection**: Visual and sensor-based anomaly detection
//! - **Sensor types**: Camera, microphone, accelerometer, positional sensors
//! - **Continuous mode**: Real-time streaming inference support
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
//! ### FFI Mode (Default - Recommended)
//! ```no_run
//! use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance with FFI mode (default)
//!     let mut model = EdgeImpulseModel::new()?;
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
//! ### EIM Mode (Legacy - Backward Compatibility)
//! ```no_run
//! use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance with EIM file (legacy mode)
//!     let mut model = EdgeImpulseModel::new_eim("path/to/model.eim")?;
//!
//!     // Prepare normalized features (e.g., image pixels, audio samples)
//!     let features: Vec<f32> = vec![0.1, 0.2, 0.3];
//!
//!     // Run inference
//!     let result = model.infer(features, None)?;
//!
//!     // Process results (same as FFI mode)
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
//!         InferenceResult::VisualAnomaly { .. } => {
//!             println!("Anomaly detection result");
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
//! ### Backend Abstraction
//! The crate uses a trait-based backend system that allows switching between different
//! inference modes:
//!
//! - **EIM Backend**: Uses Unix socket-based IPC to communicate with Edge Impulse model
//!   processes. The protocol is JSON-based and follows a request-response pattern for
//!   model initialization, classification requests, and error handling.
//!
//! - **FFI Backend**: Direct FFI calls to the Edge Impulse C++ SDK, providing faster
//!   startup times and lower latency by eliminating IPC overhead.
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
//! The crate uses the `EdgeImpulseError` type to provide detailed error information:
//! ```no_run
//! use edge_impulse_runner::{EdgeImpulseModel, EdgeImpulseError};
//!
//! // Match on model creation
//! match EdgeImpulseModel::new("model.eim") {
//!     Ok(mut model) => {
//!         // Match on classification
//!         match model.infer(vec![0.1, 0.2, 0.3], None) {
//!             Ok(result) => println!("Success!"),
//!             Err(EdgeImpulseError::InvalidInput(msg)) => println!("Invalid input: {}", msg),
//!             Err(e) => println!("Other error: {}", e),
//!         }
//!     },
//!     Err(e) => println!("Failed to load model: {}", e),
//! }
//! ```
//!
//! ## Modules
//!
//! - `backends`: Backend abstraction and implementations (EIM, FFI)
//! - `error`: Error types and handling
//! - `ffi`: Safe Rust bindings for Edge Impulse C++ SDK (when `ffi` feature is enabled)
//! - `inference`: Model management and inference functionality
//! - `ingestion`: Data upload and project management
//! - `types`: Common types and parameters
//!
//! ## Cargo Features
//!
//! - **`eim`** (default): Enable EIM binary communication mode
//! - **`ffi`**: Enable FFI direct mode (requires `edge-impulse-ffi-rs` dependency)
//!
//! Only one backend should be enabled at a time. Enabling both features simultaneously
//! is not supported and may cause conflicts.

pub mod backends;
pub mod error;
pub mod ffi;
pub mod inference;
pub mod ingestion;
pub mod types;

pub use inference::messages::{InferenceResponse, InferenceResult};

pub use error::EdgeImpulseError;
pub use inference::model::EdgeImpulseModel;
pub use types::BoundingBox;
pub use types::ModelParameters;
pub use types::ProjectInfo;
pub use types::SensorType;
pub use types::TimingInfo;
