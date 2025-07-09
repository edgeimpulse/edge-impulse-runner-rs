//! FFI module for Edge Impulse C++ SDK bindings
//!
//! This module provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing direct FFI calls to run inference on trained models.

#[cfg(feature = "ffi")]
pub use edge_impulse_ffi_rs::*;

#[cfg(not(feature = "ffi"))]
mod bindings;
#[cfg(not(feature = "ffi"))]
mod classifier;
#[cfg(not(feature = "ffi"))]
mod error;
#[cfg(not(feature = "ffi"))]
mod metadata;
#[cfg(not(feature = "ffi"))]
mod signal;
#[cfg(not(feature = "ffi"))]
mod types;

#[cfg(not(feature = "ffi"))]
pub use classifier::EdgeImpulseClassifier;
#[cfg(not(feature = "ffi"))]
pub use error::{EdgeImpulseError, EdgeImpulseResult};
#[cfg(not(feature = "ffi"))]
pub use metadata::ModelMetadata;
#[cfg(not(feature = "ffi"))]
pub use signal::Signal;
#[cfg(not(feature = "ffi"))]
pub use types::{BoundingBox, ClassificationResult, InferenceResult, TimingResult};