//! # Edge Impulse Runner
//!
//! A Rust library for running inference with Edge Impulse models and uploading data to
//! Edge Impulse projects. This crate provides safe and easy-to-use interfaces for:
//! - Running machine learning models on Linux and macOS
//! - Uploading training, testing and anomaly data to Edge Impulse projects
//!

//!
//! ## Overview
//!
//! This Rust library provides two modes for running Edge Impulse models:
//!
//! - **FFI Mode (Default & Recommended)**: Uses direct C++ SDK integration via FFI bindings. This is the default and recommended mode for all new applications.
//! - **EIM Mode (Legacy/Compatibility)**: Uses Edge Impulse's EIM (Edge Impulse Model) format for inference. This mode is only for backward compatibility and is not recommended for new projects due to performance penalties.
//!
//! ## Features
//!
//! ### Inference Capabilities
//! - Run Edge Impulse models on Linux and MacOS
//! - Support for different model types:
//!   - Classification models
//!   - Object detection models
//!   - Visual anomaly detection models
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
//! ## Quick Start
//!
//! ### FFI Mode (Default & Recommended)
//! Set your Edge Impulse project credentials and build:
//!
//! ```sh
//! export EI_PROJECT_ID=12345
//! export EI_API_KEY=ei_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
//! cargo clean
//! cargo build
//! ```
//!
//! This will automatically download your model from Edge Impulse and build the C++ SDK bindings.
//!
//! **Note:** The download process may take several minutes on the first build. Never commit your API key to version control.
//!
//! ### EIM Mode (Legacy/Compatibility)
//! > **Not recommended except for legacy/dev use.**
//!
//! ```sh
//! cargo build --no-default-features --features eim
//! cargo run --example basic_infer -- --model path/to/model.eim --features "0.1,0.2,0.3"
//! ```
//!
//! ## Examples
//!
//! ### Basic Inference (FFI mode - default)
//! ```sh
//! cargo run --example basic_infer -- --features "0.1,0.2,0.3"
//! ```
//!
//! ### Image Inference (FFI mode - default)
//! ```sh
//! cargo run --example image_infer -- --image /path/to/image.png
//! ```
//!
//! ### Audio Inference (FFI mode - default)
//! ```sh
//! cargo run --example audio_infer -- --audio /path/to/audio.wav
//! ```
//!
//! ### EIM Mode (Legacy)
//! ```sh
//! # For EIM mode, use the --eim flag and provide a model path
//! cargo run --example basic_infer -- --eim --model path/to/model.eim --features "0.1,0.2,0.3"
//! ```
//!
//! ## Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! edge-impulse-runner = "2.0.0"
//! ```
//!
//! FFI mode is enabled by default. Set the `EI_PROJECT_ID` and `EI_API_KEY` environment variables.
//!
//! ## Usage
//!
//! ### FFI Mode (Default & Recommended)
//!
//! ```no_run
//! use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance using FFI (default)
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
//!         InferenceResult::ObjectDetection { bounding_boxes, classification } => {
//!             println!("Detected objects: {:?}", bounding_boxes);
//!             if !classification.is_empty() {
//!                 println!("Classification: {:?}", classification);
//!             }
//!         }
//!         InferenceResult::VisualAnomaly { visual_anomaly_grid, visual_anomaly_max, visual_anomaly_mean, anomaly } => {
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
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! #### FFI Mode with Debug
//! ```no_run
//! use edge_impulse_runner::EdgeImpulseModel;
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut model = EdgeImpulseModel::new_with_debug(true)?;
//!     Ok(())
//! }
//! ```
//!
//! ### EIM Mode (Legacy/Compatibility)
//! > **Not recommended except for legacy/dev use.**
//!
//! ```no_run
//! use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a new model instance using EIM (legacy)
//!     let mut model = EdgeImpulseModel::new_eim("path/to/model.eim")?;
//!     // ...
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
//! ## Cargo Features
//!
//! The crate supports different backends through Cargo features:
//!
//! ```toml
//! [dependencies]
//! edge-impulse-runner = { version = "2.0.0" }
//! ```
//!
//! ### Available Features
//!
//! - **`ffi`** (default): Enable FFI direct mode (requires `edge-impulse-ffi-rs` dependency)
//! - **`eim`**: Enable EIM binary communication mode (legacy)
//!
//! ## Migration from 1.0.0 to 2.0.0
//!
//! Version 2.0.0 introduces significant improvements and new features while maintaining backward compatibility for EIM mode.
//!
//! ### What's New in 2.0.0
//!
//! #### 🚀 **FFI Mode is now Default & Recommended**
//! - **Direct FFI calls**: New FFI backend for improved performance
//! - **No inter-process communication**: Eliminates socket overhead
//! - **Lower latency**: Direct calls to Edge Impulse C++ SDK
//! - **Reduced memory usage**: No separate model process
//!
//! #### 🔄 **Unified API**
//! - **Consistent naming**: `EimModel` renamed to `EdgeImpulseModel` for clarity
//! - **Dual backend support**: Same API works with both FFI and EIM modes
//! - **Feature-based selection**: Choose backend via Cargo features
//! - **Constructor change**: `EdgeImpulseModel::new()` is now FFI, `new_eim()` is for legacy EIM
//!
//! #### 🛠 **Enhanced Architecture**
//! - **Trait-based backends**: Clean abstraction for different inference engines
//!
//! #### ⚠️ **EIM Mode is Legacy**
//! - EIM mode is only for backward compatibility and is not recommended for new projects.
//!
//! ## Building
//!
//! ### FFI Build Modes
//!
//! - **Default (TensorFlow Lite Micro - for microcontrollers):**
//!   ```sh
//!   cargo build
//!   ```
//! - **Full TensorFlow Lite (for desktop/server):**
//!   ```sh
//!   USE_FULL_TFLITE=1 cargo build
//!   ```
//!
//! ### Platform-Specific Builds
//!
//! You can specify the target platform explicitly using these environment variables:
//!
//! #### macOS
//! ```sh
//! # Apple Silicon (M1/M2/M3)
//! TARGET_MAC_ARM64=1 USE_FULL_TFLITE=1 cargo build
//! # Intel Mac
//! TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 cargo build
//! ```
//! #### Linux
//! ```sh
//! # Linux x86_64
//! TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 cargo build
//! # Linux ARM64
//! TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build
//! # Linux ARMv7
//! TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1 cargo build
//! ```
//! #### NVIDIA Jetson
//! ```sh
//! # Jetson Nano
//! TARGET_JETSON_NANO=1 USE_FULL_TFLITE=1 cargo build
//! # Jetson Orin
//! TARGET_JETSON_ORIN=1 USE_FULL_TFLITE=1 cargo build
//! ```
//!
//! ### Platform Support
//!
//! Full TensorFlow Lite uses prebuilt binaries from the `tflite/` directory:
//!
//! | Platform           | Directory                |
//! |--------------------|-------------------------|
//! | macOS ARM64        | tflite/mac-arm64/       |
//! | macOS x86_64       | tflite/mac-x86_64/      |
//! | Linux x86          | tflite/linux-x86/       |
//! | Linux ARM64        | tflite/linux-aarch64/   |
//! | Linux ARMv7        | tflite/linux-armv7/     |
//! | Jetson Nano        | tflite/linux-jetson-nano/|
//!
//! **Note:** If no platform is specified, the build system will auto-detect based on your current system architecture.
//!
//! ### Model Updates
//!
//! If you want to switch between models or make sure you always get the latest version of your model, you have to set the `CLEAN_MODEL=1` environment variable:
//!
//! ```sh
//! CLEAN_MODEL=1 cargo build
//! ```
//!
//! ## Advanced Build Flags
//!
//! This project supports a wide range of advanced build flags for hardware accelerators, backends, and cross-compilation, mirroring the Makefile from Edge Impulse's [example-standalone-inferencing-linux](https://github.com/edgeimpulse/example-standalone-inferencing-linux). You can combine these flags as needed:
//!
//! | Flag                          | Purpose / Effect                                                                                 |
//! |-------------------------------|-----------------------------------------------------------------------------------------------|
//! | `USE_TVM=1`                   | Enable Apache TVM backend (requires `TVM_HOME` env var)                                         |
//! | `USE_ONNX=1`                  | Enable ONNX Runtime backend                                                                    |
//! | `USE_QUALCOMM_QNN=1`          | Enable Qualcomm QNN delegate (requires `QNN_SDK_ROOT` env var)                                 |
//! | `USE_ETHOS=1`                 | Enable ARM Ethos-U delegate                                                                   |
//! | `USE_AKIDA=1`                 | Enable BrainChip Akida backend                                                                |
//! | `USE_MEMRYX=1`                | Enable MemryX backend                                                                         |
//! | `LINK_TFLITE_FLEX_LIBRARY=1`  | Link TensorFlow Lite Flex library                                                             |
//! | `EI_CLASSIFIER_USE_MEMRYX_SOFTWARE=1` | Use MemryX software mode (with Python bindings)                                    |
//! | `TENSORRT_VERSION=8.5.2`      | Set TensorRT version for Jetson platforms                                                     |
//! | `TVM_HOME=/path/to/tvm`       | Path to TVM installation (required for `USE_TVM=1`)                                           |
//! | `QNN_SDK_ROOT=/path/to/qnn`   | Path to Qualcomm QNN SDK (required for `USE_QUALCOMM_QNN=1`)                                  |
//! | `PYTHON_CROSS_PATH=...`       | Path prefix for cross-compiling Python bindings                                               |
//!
//! ### Example Advanced Builds
//!
//! ```sh
//! # Build with ONNX Runtime and full TensorFlow Lite for TI AM68A
//! USE_ONNX=1 TARGET_AM68A=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build with TVM backend (requires TVM_HOME)
//! USE_TVM=1 TVM_HOME=/opt/tvm TARGET_RENESAS_RZV2L=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build with Qualcomm QNN delegate (requires QNN_SDK_ROOT)
//! USE_QUALCOMM_QNN=1 QNN_SDK_ROOT=/opt/qnn TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build with ARM Ethos-U delegate
//! USE_ETHOS=1 TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build with MemryX backend in software mode
//! USE_MEMRYX=1 EI_CLASSIFIER_USE_MEMRYX_SOFTWARE=1 TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build with TensorFlow Lite Flex library
//! LINK_TFLITE_FLEX_LIBRARY=1 USE_FULL_TFLITE=1 cargo build
//!
//! # Build for Jetson Nano with specific TensorRT version
//! TARGET_JETSON_NANO=1 TENSORRT_VERSION=8.5.2 USE_FULL_TFLITE=1 cargo build
//! ```
//!
//! See the Makefile in Edge Impulse's [example-standalone-inferencing-linux](https://github.com/edgeimpulse/example-standalone-inferencing-linux) for more details on what each flag does. Not all combinations are valid for all models/platforms.
//!
//! ## Prerequisites
//!
//! Some examples (particularly video capture) requires GStreamer to be installed:
//! - **macOS**: Install both runtime and development packages from gstreamer.freedesktop.org
//! - **Linux**: Install required packages (libgstreamer1.0-dev and related packages)
//!
//! ## Error Handling
//!
//! The crate uses the `EdgeImpulseError` type to provide detailed error information:
//!
//! ```rust
//! use edge_impulse_runner::{EdgeImpulseModel, EdgeImpulseError};
//!
//! match EdgeImpulseModel::new() {
//!     Ok(mut model) => {
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
//! ## Architecture
//!
//! ### Backend Abstraction
//!
//! The library uses a trait-based backend abstraction that allows switching between different inference engines:
//!
//! ```rust
//! use edge_impulse_runner::{backends::BackendConfig, EdgeImpulseError, InferenceResponse, ModelParameters};
//!
//! pub trait InferenceBackend: Send + Sync {
//!     fn new(config: BackendConfig) -> Result<Self, EdgeImpulseError> where Self: Sized;
//!     fn infer(&mut self, features: Vec<f32>, debug: Option<bool>) -> Result<InferenceResponse, EdgeImpulseError>;
//!     fn parameters(&self) -> Result<&ModelParameters, EdgeImpulseError>;
//!     // ... other methods
//! }
//! ```
//!
//! ### EIM Backend
//! - Communicates with Edge Impulse binary files over Unix sockets
//! - Supports all features including continuous mode and threshold configuration
//! - Requires `.eim` files to be present
//!
//! ### FFI Backend
//! - Direct FFI calls to the Edge Impulse C++ SDK
//! - Improved performance with no inter-process communication overhead
//! - Requires the `edge-impulse-ffi-rs` crate as a dependency
//! - Model must be compiled into the binary
//!
//! ## Inference Communication Protocol (EIM Mode)
//!
//! The Edge Impulse Inference Runner uses a Unix socket-based IPC mechanism to communicate with the model process in EIM mode. The protocol is JSON-based and follows a request-response pattern.
//!
//! ### Protocol Messages
//!
//! #### 1. Initialization
//! When a new model is created, the following sequence occurs:
//!
//! ##### HelloMessage (Runner -> Model):
//! ```json
//! {
//!     "hello": 1,
//!     "id": 1
//! }
//! ```
//!
//! ##### ModelInfo Response (Model -> Runner):
//! ```json
//! {
//!     "success": true,
//!     "id": 1,
//!     "model_parameters": {
//!         "axis_count": 1,
//!         "frequency": 16000.0,
//!         "has_anomaly": 0,
//!         "image_channel_count": 3,
//!         "image_input_frames": 1,
//!         "image_input_height": 96,
//!         "image_input_width": 96,
//!         "image_resize_mode": "fit-shortest",
//!         "inferencing_engine": 4,
//!         "input_features_count": 9216,
//!         "interval_ms": 1.0,
//!         "label_count": 1,
//!         "labels": ["class1"],
//!         "model_type": "classification",
//!         "sensor": 3,
//!         "slice_size": 2304,
//!         "threshold": 0.5,
//!         "use_continuous_mode": false
//!     },
//!     "project": {
//!         "deploy_version": 1,
//!         "id": 12345,
//!         "name": "Project Name",
//!         "owner": "Owner Name"
//!     }
//! }
//! ```
//!
//! #### 2. Inference
//! For each inference request:
//!
//! ##### ClassifyMessage (Runner -> Model):
//! ```json
//! {
//!     "classify": [0.1, 0.2, 0.3],
//!     "id": 1,
//!     "debug": false
//! }
//! ```
//!
//! ##### InferenceResponse (Model -> Runner):
//! For classification models:
//! ```json
//! {
//!     "success": true,
//!     "id": 2,
//!     "result": {
//!         "classification": {
//!             "class1": 0.8,
//!             "class2": 0.2
//!         }
//!     }
//! }
//! ```
//!
//! For object detection models:
//! ```json
//! {
//!     "success": true,
//!     "id": 2,
//!     "result": {
//!         "bounding_boxes": [
//!             {
//!                 "label": "object1",
//!                 "value": 0.95,
//!                 "x": 100,
//!                 "y": 150,
//!                 "width": 50,
//!                 "height": 50
//!             }
//!         ],
//!         "classification": {
//!             "class1": 0.8,
//!             "class2": 0.2
//!         }
//!     }
//! }
//! ```
//!
//! For visual anomaly detection models:
//! ```json
//! {
//!     "success": true,
//!     "id": 2,
//!     "result": {
//!         "visual_anomaly": {
//!             "anomaly": 5.23,
//!             "visual_anomaly_max": 7.89,
//!             "visual_anomaly_mean": 4.12,
//!             "visual_anomaly_grid": [
//!                 {
//!                     "value": 0.955,
//!                     "x": 24,
//!                     "y": 40,
//!                     "width": 8,
//!                     "height": 16
//!                 }
//!             ]
//!         }
//!     }
//! }
//! ```
//!
//! #### 3. Error Response
//! When errors occur:
//!
//! ##### ErrorResponse (Model -> Runner):
//! ```json
//! {
//!     "success": false,
//!     "error": "Error message",
//!     "id": 2
//! }
//! ```
//!
//! ## Troubleshooting
//!
//! ### Common Issues
//!
//! **Model is not downloaded or you see warnings like `Could not find edge-impulse-ffi-rs build output directory`:**
//! - Make sure the environment variables are set:
//!   - `EI_PROJECT_ID`
//!   - `EI_API_KEY`
//! - Run:
//!   ```sh
//!   cargo clean
//!   cargo build --features ffi
//!   ```
//! - The FFI crate is always pulled from the official GitHub repository, not from a local path.
//!
//! **Download fails with authentication error:**
//! - Verify your API key is correct and has access to the project
//! - Check that the project ID exists and is accessible
//! - Ensure environment variables are set correctly: `EI_PROJECT_ID` and `EI_API_KEY`
//!
//! **Download times out:**
//! - The download process can take several minutes for large models
//! - Check your internet connection
//!
//! **Build job fails:**
//! - Ensure your Edge Impulse project has a valid model deployed
//! - Check the Edge Impulse Studio for any build errors
//! - Verify the project has the correct deployment target (Linux)
//!
//! ### Manual Override
//!
//! If automated download fails, you can:
//! 1. Manually download the model from Edge Impulse Studio
//! 2. Extract it to the `model/` directory
//! 3. Unset the environment variables: `unset EI_PROJECT_ID EI_API_KEY`
//! 4. Build normally with `cargo build`
//!
//! ## Modules
//!
//! - `backends`: Backend abstraction and implementations (EIM, FFI)
//! - `error`: Error types and handling
//! - `ffi`: Safe Rust bindings for Edge Impulse C++ SDK (when `ffi` feature is enabled)
//! - `inference`: Model management and inference functionality
//! - `ingestion`: Data upload and project management
//! - `types`: Common types and parameters

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
