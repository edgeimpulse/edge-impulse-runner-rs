//! FFI module for Edge Impulse C++ SDK bindings
//!
//! This module provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing direct FFI calls to run inference on trained models.
//!
//! ## Features
//!
//! - **Model Metadata**: Access to model configuration and parameters
//! - **Signal Processing**: Safe wrappers for audio and sensor data
//! - **Inference**: Direct model inference with error handling
//! - **Classification Results**: Structured access to model outputs
//!
//! ## Usage
//!
//! This module is only available when the `ffi` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! edge-impulse-runner = { version = "2.0.0", features = ["ffi"] }
//! ```

mod wrapper;

// Re-export the wrapper's public API
pub use wrapper::*;
