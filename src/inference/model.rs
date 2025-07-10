use crate::backends::{BackendConfig, InferenceBackend, create_backend};
use crate::error::EdgeImpulseError;
use crate::inference::messages::InferenceResponse;
use crate::types::{ModelParameters, SensorType, VisualAnomalyResult};
use std::path::Path;

/// Main Edge Impulse model interface that abstracts over different backends
///
/// This struct provides a unified interface for running inference on Edge Impulse models,
/// regardless of whether you're using EIM binary communication or FFI direct calls.
/// The backend is automatically selected based on the constructor used.
///
/// ## Examples
///
/// ```no_run
/// use edge_impulse_runner::EdgeImpulseModel;
///
/// // EIM mode (default)
/// let mut model = EdgeImpulseModel::new("model.eim")?;
///
/// // FFI mode
/// let mut model = EdgeImpulseModel::new_ffi(false)?;
///
/// // Run inference
/// let result = model.infer(vec![0.1, 0.2, 0.3], None)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EdgeImpulseModel {
    backend: Box<dyn InferenceBackend>,
}

impl EdgeImpulseModel {
    /// Create a new model instance using EIM backend
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, EdgeImpulseError> {
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
    ) -> Result<Self, EdgeImpulseError> {
        let config = BackendConfig::Eim {
            path: model_path.as_ref().to_path_buf(),
            socket_path: Some(socket_path.as_ref().to_path_buf()),
        };
        let backend = create_backend(config)?;
        Ok(Self { backend })
    }

    /// Create a new model instance using EIM backend with debug output
    pub fn new_with_debug<P: AsRef<Path>>(
        model_path: P,
        debug: bool,
    ) -> Result<Self, EdgeImpulseError> {
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
    pub fn new_ffi(debug: bool) -> Result<Self, EdgeImpulseError> {
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
    ) -> Result<InferenceResponse, EdgeImpulseError> {
        self.backend.infer(features, debug)
    }

    /// Get model parameters
    pub fn parameters(&self) -> Result<&ModelParameters, EdgeImpulseError> {
        self.backend.parameters()
    }

    /// Get sensor type
    pub fn sensor_type(&self) -> Result<SensorType, EdgeImpulseError> {
        self.backend.sensor_type()
    }

    /// Get input size
    pub fn input_size(&self) -> Result<usize, EdgeImpulseError> {
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
