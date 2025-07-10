//! Backend abstraction for Edge Impulse inference
//!
//! This module provides a trait-based abstraction that allows switching between
//! different inference backends:
//!
//! - **EIM Backend**: Binary communication with Edge Impulse model processes via Unix sockets
//! - **FFI Backend**: Direct FFI calls to the Edge Impulse C++ SDK
//!
//! The backend system is designed to be extensible, allowing new inference engines
//! to be added by implementing the `InferenceBackend` trait.

use crate::error::EimError;
use crate::inference::messages::InferenceResponse;
use crate::types::ModelParameters;
use std::path::PathBuf;

/// Configuration for different backend types
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// EIM binary communication mode
    Eim {
        /// Path to the .eim file
        path: PathBuf,
        /// Optional custom socket path
        socket_path: Option<PathBuf>,
    },
    /// FFI direct mode
    Ffi {
        /// Enable debug logging
        debug: bool,
    },
}

/// Trait for inference backends
pub trait InferenceBackend: Send + Sync {
    /// Create a new backend instance
    fn new(config: BackendConfig) -> Result<Self, EimError>
    where
        Self: Sized;

    /// Run inference on features
    fn infer(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError>;

    /// Get model parameters
    fn parameters(&self) -> Result<&ModelParameters, EimError>;

    /// Get sensor type
    fn sensor_type(&self) -> Result<crate::types::SensorType, EimError>;

    /// Get input size
    fn input_size(&self) -> Result<usize, EimError>;

    /// Set debug callback
    fn set_debug_callback(&mut self, callback: Box<dyn Fn(&str) + Send + Sync>);

    /// Normalize visual anomaly results
    fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> crate::types::VisualAnomalyResult;
}

#[cfg(feature = "eim")]
pub mod eim;

#[cfg(feature = "ffi")]
pub mod ffi;

/// Factory function to create the appropriate backend
pub fn create_backend(config: BackendConfig) -> Result<Box<dyn InferenceBackend>, EimError> {
    match config {
        #[cfg(feature = "eim")]
        BackendConfig::Eim { path, socket_path } => {
            // Validate file extension for EIM backend
            if let Some(ext) = path.extension() {
                if ext != "eim" {
                    return Err(EimError::InvalidPath);
                }
            } else {
                return Err(EimError::InvalidPath);
            }

            use eim::EimBackend;
            Ok(Box::new(EimBackend::new(BackendConfig::Eim {
                path,
                socket_path,
            })?))
        }
        #[cfg(feature = "ffi")]
        BackendConfig::Ffi { debug } => {
            use ffi::FfiBackend;
            Ok(Box::new(FfiBackend::new(BackendConfig::Ffi { debug })?))
        }
        #[cfg(not(feature = "eim"))]
        BackendConfig::Eim { .. } => Err(EimError::InvalidOperation(
            "EIM backend not enabled. Enable the 'eim' feature.".to_string(),
        )),
        #[cfg(not(feature = "ffi"))]
        BackendConfig::Ffi { .. } => Err(EimError::InvalidOperation(
            "FFI backend not enabled. Enable the 'ffi' feature.".to_string(),
        )),
    }
}
