//! EIM backend implementation
//!
//! This module provides the EIM backend that communicates with Edge Impulse
//! binary files over Unix sockets.

use super::{BackendConfig, InferenceBackend};
use crate::error::EdgeImpulseError;
use crate::inference::messages::InferenceResponse;
use crate::types::{ModelParameters, SensorType, VisualAnomalyResult};
use rand::{Rng, thread_rng};
// Removed unused import
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::Child;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tempfile::{TempDir, tempdir};

use crate::inference::messages::{ClassifyMessage, HelloMessage, InferenceResult, ModelInfo};

/// Debug callback type for receiving debug messages
pub type DebugCallback = Box<dyn Fn(&str) + Send + Sync>;

/// EIM backend implementation for socket-based communication
pub struct EimBackend {
    /// Path to the Edge Impulse model file (.eim)
    #[allow(dead_code)]
    path: std::path::PathBuf,
    /// Path to the Unix socket used for IPC
    #[allow(dead_code)]
    socket_path: std::path::PathBuf,
    /// Handle to the temporary directory for the socket (ensures cleanup)
    #[allow(dead_code)]
    tempdir: Option<TempDir>,
    /// Active Unix socket connection to the model process
    socket: UnixStream,
    /// Enable debug logging of socket communications
    #[allow(dead_code)]
    debug: bool,
    /// Optional debug callback for receiving debug messages
    debug_callback: Option<DebugCallback>,
    /// Handle to the model process (kept alive while model exists)
    _process: Child,
    /// Cached model information received during initialization
    model_info: Option<ModelInfo>,
    /// Atomic counter for generating unique message IDs
    message_id: AtomicU32,
    /// Model parameters extracted from model info
    model_parameters: ModelParameters,
}

impl EimBackend {
    /// Create a new EIM backend
    pub fn new(config: BackendConfig) -> Result<Self, EdgeImpulseError> {
        let BackendConfig::Eim { path, .. } = config else {
            return Err(EdgeImpulseError::InvalidOperation(
                "Invalid config type for EIM backend".to_string(),
            ));
        };

        // Always generate a temp socket path
        let tempdir = tempdir().map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to create tempdir: {e}"))
        })?;
        let mut rng = thread_rng();
        let socket_name = format!("eim_socket_{}", rng.r#gen::<u64>());
        let socket_path = tempdir.path().join(socket_name);

        // Ensure the model file has execution permissions
        Self::ensure_executable(&path)?;

        // Start the model process with the socket path as the first positional argument
        println!(
            "Starting EIM process: {} {}",
            path.display(),
            socket_path.display()
        );
        let process = std::process::Command::new(&path)
            .arg(&socket_path)
            .spawn()
            .map_err(|e| {
                EdgeImpulseError::ExecutionError(format!("Failed to start model process: {e}"))
            })?;

        // Wait for the socket to be created and connect
        let socket = Self::connect_with_retry(&socket_path, Duration::from_secs(10))?;

        let mut backend = Self {
            path,
            socket_path,
            tempdir: Some(tempdir),
            socket,
            debug: false,
            debug_callback: None,
            _process: process,
            model_info: None,
            message_id: AtomicU32::new(1),
            model_parameters: ModelParameters::default(),
        };

        // Send hello message to get model info
        backend.send_hello()?;

        Ok(backend)
    }

    /// Ensure the model file has execution permissions for the current user
    fn ensure_executable<P: AsRef<Path>>(path: P) -> Result<(), EdgeImpulseError> {
        use std::os::unix::fs::PermissionsExt;

        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to get file metadata: {e}"))
        })?;

        let perms = metadata.permissions();
        let current_mode = perms.mode();
        if current_mode & 0o100 == 0 {
            // File is not executable for user, try to make it executable
            let mut new_perms = perms;
            new_perms.set_mode(current_mode | 0o100); // Add executable bit for user only
            std::fs::set_permissions(path, new_perms).map_err(|e| {
                EdgeImpulseError::ExecutionError(format!(
                    "Failed to set executable permissions: {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// Connect to the socket with retry logic
    fn connect_with_retry(
        socket_path: &Path,
        timeout: Duration,
    ) -> Result<UnixStream, EdgeImpulseError> {
        println!("Attempting to connect to socket: {}", socket_path.display());
        let start = Instant::now();
        while start.elapsed() < timeout {
            match UnixStream::connect(socket_path) {
                Ok(socket) => {
                    println!("Successfully connected to socket");
                    return Ok(socket);
                }
                Err(_e) => {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
        Err(EdgeImpulseError::SocketError(format!(
            "Timeout waiting for socket {} to become available",
            socket_path.display()
        )))
    }

    /// Send hello message to get model information
    fn send_hello(&mut self) -> Result<(), EdgeImpulseError> {
        let hello = HelloMessage {
            id: self.next_message_id(),
            hello: 1,
        };

        let hello_json = serde_json::to_string(&hello).map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!("Failed to serialize hello: {e}"))
        })?;

        self.debug_message(&format!("Sending hello: {hello_json}"));

        // Send the message
        self.socket
            .write_all(hello_json.as_bytes())
            .map_err(|e| EdgeImpulseError::ExecutionError(format!("Failed to send hello: {e}")))?;
        self.socket.write_all(b"\n").map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to send newline: {e}"))
        })?;

        // Read the response
        let mut reader = BufReader::new(&self.socket);
        let mut response = String::new();
        reader.read_line(&mut response).map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to read hello response: {e}"))
        })?;

        self.debug_message(&format!("Received hello response: {}", response.trim()));

        // Parse the response
        let model_info: ModelInfo = serde_json::from_str(&response).map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!("Failed to parse hello response: {e}"))
        })?;

        self.model_info = Some(model_info.clone());

        // Extract model parameters from the model info
        self.model_parameters = model_info.model_parameters;

        Ok(())
    }

    /// Generate the next unique message ID
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Set the debug callback
    pub fn set_debug_callback(&mut self, callback: DebugCallback) {
        self.debug_callback = Some(callback);
    }

    /// Send a debug message if a callback is set
    fn debug_message(&self, msg: &str) {
        if let Some(callback) = &self.debug_callback {
            callback(msg);
        }
    }

    /// Update the cached model parameters with a new threshold
    fn update_cached_threshold(&mut self, new_threshold: crate::types::ModelThreshold) {
        // Find and update the existing threshold with the same ID, or add a new one
        let mut found = false;

        // First, try to find and update existing threshold
        for i in 0..self.model_parameters.thresholds.len() {
            let should_update = match (&self.model_parameters.thresholds[i], &new_threshold) {
                (
                    crate::types::ModelThreshold::ObjectDetection {
                        id: existing_id, ..
                    },
                    crate::types::ModelThreshold::ObjectDetection { id: new_id, .. },
                ) if *existing_id == *new_id => true,
                (
                    crate::types::ModelThreshold::AnomalyGMM {
                        id: existing_id, ..
                    },
                    crate::types::ModelThreshold::AnomalyGMM { id: new_id, .. },
                ) if *existing_id == *new_id => true,
                _ => false,
            };

            if should_update {
                self.model_parameters.thresholds[i] = new_threshold.clone();
                found = true;
                break;
            }
        }

        // If no existing threshold was found, add the new one
        if !found {
            self.model_parameters.thresholds.push(new_threshold);
        }
    }

    /// Classify a single input
    fn classify(&mut self, input: &[f32]) -> Result<InferenceResult, EdgeImpulseError> {
        let classify = ClassifyMessage {
            id: self.next_message_id(),
            classify: input.to_vec(),
            debug: None,
        };

        let classify_json = serde_json::to_string(&classify).map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!("Failed to serialize classify: {e}"))
        })?;

        self.socket
            .write_all(classify_json.as_bytes())
            .map_err(|e| {
                EdgeImpulseError::ExecutionError(format!("Failed to send classify: {e}"))
            })?;
        self.socket.write_all(b"\n").map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to send newline: {e}"))
        })?;

        let mut reader = BufReader::new(&self.socket);
        let mut response_json = String::new();
        reader.read_line(&mut response_json).map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to read classify response: {e}"))
        })?;

        let response: InferenceResponse = match serde_json::from_str(&response_json) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[EIM backend] Failed to parse classify response: {}\nRaw response: {}",
                    e,
                    response_json.trim()
                );
                return Err(EdgeImpulseError::InvalidOperation(format!(
                    "Failed to parse classify response: {e}"
                )));
            }
        };

        Ok(response.result)
    }
}

impl InferenceBackend for EimBackend {
    fn new(config: BackendConfig) -> Result<Self, EdgeImpulseError> {
        EimBackend::new(config)
    }

    fn infer(
        &mut self,
        features: Vec<f32>,
        _debug: Option<bool>,
    ) -> Result<InferenceResponse, EdgeImpulseError> {
        // Use classify and wrap in InferenceResponse
        let result = self.classify(&features)?;
        Ok(InferenceResponse {
            success: true,
            id: self.next_message_id(),
            result,
        })
    }

    fn parameters(&self) -> Result<&ModelParameters, EdgeImpulseError> {
        Ok(&self.model_parameters)
    }

    fn sensor_type(&self) -> Result<SensorType, EdgeImpulseError> {
        // Convert from i32 to SensorType
        Ok(SensorType::from(self.model_parameters.sensor))
    }

    fn input_size(&self) -> Result<usize, EdgeImpulseError> {
        Ok(self.model_parameters.input_features_count as usize)
    }

    fn set_debug_callback(&mut self, callback: Box<dyn Fn(&str) + Send + Sync>) {
        self.set_debug_callback(callback);
    }

    fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> VisualAnomalyResult {
        (anomaly, max, mean, regions.to_vec())
    }

    fn set_threshold(
        &mut self,
        threshold: crate::types::ModelThreshold,
    ) -> Result<(), EdgeImpulseError> {
        // Convert ModelThreshold to ThresholdConfig
        let threshold_config = match threshold {
            crate::types::ModelThreshold::ObjectDetection { id, min_score } => {
                crate::inference::messages::ThresholdConfig::ObjectDetection { id, min_score }
            }
            crate::types::ModelThreshold::AnomalyGMM {
                id,
                min_anomaly_score,
            } => crate::inference::messages::ThresholdConfig::AnomalyGMM {
                id,
                min_anomaly_score,
            },
            _ => {
                return Err(EdgeImpulseError::InvalidOperation(
                    "Unsupported threshold type for EIM backend".to_string(),
                ));
            }
        };

        let set_threshold_msg = crate::inference::messages::SetThresholdMessage {
            set_threshold: threshold_config,
            id: self.next_message_id(),
        };

        let set_threshold_json = serde_json::to_string(&set_threshold_msg).map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!("Failed to serialize set_threshold: {e}"))
        })?;

        self.socket
            .write_all(set_threshold_json.as_bytes())
            .map_err(|e| {
                EdgeImpulseError::ExecutionError(format!("Failed to send set_threshold: {e}"))
            })?;
        self.socket.write_all(b"\n").map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to send newline: {e}"))
        })?;

        let mut reader = BufReader::new(&self.socket);
        let mut response_json = String::new();
        reader.read_line(&mut response_json).map_err(|e| {
            EdgeImpulseError::ExecutionError(format!("Failed to read set_threshold response: {e}"))
        })?;

        let response: crate::inference::messages::SetThresholdResponse = match serde_json::from_str(
            &response_json,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[EIM backend] Failed to parse set_threshold response: {}\nRaw response: {}",
                    e,
                    response_json.trim()
                );
                return Err(EdgeImpulseError::InvalidOperation(format!(
                    "Failed to parse set_threshold response: {e}"
                )));
            }
        };

        if !response.success {
            return Err(EdgeImpulseError::InvalidOperation(
                "Failed to set threshold".to_string(),
            ));
        }

        // Update the cached model parameters with the new threshold
        self.update_cached_threshold(threshold);

        Ok(())
    }

    fn path(&self) -> Option<&std::path::Path> {
        Some(&self.path)
    }
}
