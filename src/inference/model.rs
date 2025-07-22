use crate::backends::{BackendConfig, InferenceBackend, create_backend};
use crate::error::EdgeImpulseError;
use crate::inference::messages::InferenceResponse;
use crate::types::{ModelParameters, SensorType, VisualAnomalyResult};
#[cfg(feature = "eim")]
use std::path::Path;

/// Main Edge Impulse model interface that abstracts over different backends
///
/// This struct provides a unified interface for running inference on Edge Impulse models,
/// with automatic backend detection based on available features. The model automatically
/// chooses between FFI (recommended) and EIM (legacy) backends.
///
/// ## Examples
///
/// ```no_run
/// use edge_impulse_runner::EdgeImpulseModel;
///
/// // Automatic backend detection (FFI if available, EIM otherwise)
/// let mut model = EdgeImpulseModel::new()?;
///
/// // Run inference
/// let result = model.infer(vec![0.1, 0.2, 0.3], None)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ```ignore
/// // EIM mode (legacy - backward compatibility)
/// // This example requires the "eim" feature to be enabled
/// let mut model = EdgeImpulseModel::new_eim("model.eim")?;
///
/// // Run inference
/// let result = model.infer(vec![0.1, 0.2, 0.3], None)?;
/// ```
pub struct EdgeImpulseModel {
    backend: Box<dyn InferenceBackend>,
}

impl EdgeImpulseModel {
    /// Create a new model instance using FFI backend (default - recommended)
    ///
    /// This is the recommended constructor for all new applications. FFI mode provides
    /// superior performance with faster startup and inference times.
    pub fn new() -> Result<Self, EdgeImpulseError> {
        let config = BackendConfig::Ffi { debug: false };
        let backend = create_backend(config)?;
        Ok(Self { backend })
    }

    /// Create a new model instance using FFI backend with debug output
    pub fn new_with_debug(debug: bool) -> Result<Self, EdgeImpulseError> {
        let config = BackendConfig::Ffi { debug };
        let mut backend = create_backend(config)?;
        if debug {
            backend.set_debug_callback(Box::new(|msg| println!("[DEBUG] {msg}")));
        }
        Ok(Self { backend })
    }

    /// Create a new model instance using EIM backend (legacy - backward compatibility)
    ///
    /// This constructor is provided for backward compatibility. EIM mode has performance
    /// penalties due to IPC overhead. Use `new()` for better performance.
    #[cfg(feature = "eim")]
    pub fn new_eim<P: AsRef<Path>>(model_path: P) -> Result<Self, EdgeImpulseError> {
        let config = BackendConfig::Eim {
            path: model_path.as_ref().to_path_buf(),
            socket_path: None,
        };
        let backend = create_backend(config)?;
        Ok(Self { backend })
    }

    /// Create a new model instance using EIM backend with custom socket path
    #[cfg(feature = "eim")]
    pub fn new_eim_with_socket<P: AsRef<Path>>(
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
    #[cfg(feature = "eim")]
    pub fn new_eim_with_debug<P: AsRef<Path>>(
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

    /// Set a threshold for a specific model block
    pub fn set_threshold(
        &mut self,
        threshold: crate::types::ModelThreshold,
    ) -> Result<(), crate::error::EdgeImpulseError> {
        self.backend.set_threshold(threshold)
    }

    /// Get the path to the model file (EIM mode only)
    ///
    /// Returns `Some(path)` if the model was loaded from a file in EIM mode,
    /// or `None` if using FFI mode or if the path is not available.
    #[cfg(feature = "eim")]
    pub fn path(&self) -> Option<&std::path::Path> {
        self.backend.path()
    }
}

impl std::fmt::Debug for EdgeImpulseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeImpulseModel")
            .field("backend", &"<backend>")
            .finish()
    }
}
