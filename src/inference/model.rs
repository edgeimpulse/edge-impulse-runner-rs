use crate::backends::{create_backend, BackendConfig, InferenceBackend};
use crate::error::EimError;
use crate::inference::messages::InferenceResponse;
use crate::types::{ModelParameters, SensorType, VisualAnomalyResult};
use std::path::Path;

/// Main Edge Impulse model interface that abstracts over different backends
pub struct EdgeImpulseModel {
    backend: Box<dyn InferenceBackend>,
}

impl EdgeImpulseModel {
    /// Create a new model instance using EIM backend
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, EimError> {
        let config = BackendConfig::Eim {
            path: model_path.as_ref().to_path_buf(),
            socket_path: None,
        };
        let backend = create_backend(config)?;
        Ok(Self { backend })
    }

    /// Create a new model instance using EIM backend with custom socket path
    pub fn new_with_socket<P: AsRef<Path>>(
        model_path: P,
        socket_path: P,
    ) -> Result<Self, EimError> {
        let config = BackendConfig::Eim {
            path: model_path.as_ref().to_path_buf(),
            socket_path: Some(socket_path.as_ref().to_path_buf()),
        };
        let backend = create_backend(config)?;
        Ok(Self { backend })
    }

    /// Create a new model instance using EIM backend with debug output
    pub fn new_with_debug<P: AsRef<Path>>(model_path: P, debug: bool) -> Result<Self, EimError> {
        let config = BackendConfig::Eim {
            path: model_path.as_ref().to_path_buf(),
            socket_path: None,
        };
        let mut backend = create_backend(config)?;
        if debug {
            backend.set_debug_callback(Box::new(|msg| println!("[DEBUG] {msg}")));
        }
        Ok(Self { backend })
    }

    /// Create a new model instance using FFI backend
    pub fn new_ffi(debug: bool) -> Result<Self, EimError> {
        let config = BackendConfig::Ffi { debug };
        let mut backend = create_backend(config)?;
        if debug {
            backend.set_debug_callback(Box::new(|msg| println!("[DEBUG] {msg}")));
        }
        Ok(Self { backend })
    }

    /// Run inference on the provided features
    pub fn infer(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError> {
        self.backend.infer(features, debug)
    }

    /// Get model parameters
    pub fn parameters(&self) -> Result<&ModelParameters, EimError> {
        self.backend.parameters()
    }

    /// Get sensor type
    pub fn sensor_type(&self) -> Result<SensorType, EimError> {
        self.backend.sensor_type()
    }

    /// Get input size
    pub fn input_size(&self) -> Result<usize, EimError> {
        self.backend.input_size()
    }

    /// Normalize visual anomaly results
    pub fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> VisualAnomalyResult {
        self.backend
            .normalize_visual_anomaly(anomaly, max, mean, regions)
    }
}

impl std::fmt::Debug for EdgeImpulseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeImpulseModel")
            .field("backend", &"<backend>")
            .finish()
    }
}
