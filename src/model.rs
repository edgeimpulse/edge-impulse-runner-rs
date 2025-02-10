use std::io::{BufReader, Write, BufRead};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};
use std::process::{Child};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::EimError;
use crate::messages::*;
use crate::types::ModelParameters;

/// Supported sensor types for Edge Impulse models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorType {
    Unknown,
    Microphone,
    Accelerometer,
    Camera,
    Positional,
}

impl From<u32> for SensorType {
    fn from(value: u32) -> Self {
        match value {
            1 => SensorType::Microphone,
            2 => SensorType::Accelerometer,
            3 => SensorType::Camera,
            4 => SensorType::Positional,
            _ => SensorType::Unknown,
        }
    }
}

pub struct EimModel {
    path: std::path::PathBuf,
    socket_path: std::path::PathBuf,
    stream: UnixStream,
    debug: bool,
    _process: Child, // Keep the process alive while the model exists
    model_info: Option<ModelInfo>,
    message_id: AtomicU32,
}

impl EimModel {
    /// Create a new EimModel instance from a path to the .eim file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .eim file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use edge_impulse_runner::EimModel;
    ///
    /// let model = EimModel::new("path/to/model.eim").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, EimError> {
        Self::new_with_debug(path, false)
    }

    /// Create a new EimModel instance with debug output enabled
    pub fn new_with_debug<P: AsRef<Path>>(path: P, debug: bool) -> Result<Self, EimError> {
        let path = path.as_ref();

        // Verify the file exists and has .eim extension
        if !path.exists() {
            return Err(EimError::FileError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "EIM file not found",
            )));
        }

        if path.extension().and_then(|ext| ext.to_str()) != Some("eim") {
            return Err(EimError::InvalidPath);
        }

        // Create a temporary socket path in the system's temp directory
        let socket_path = std::env::temp_dir().join("eim_socket");

        // Remove any existing socket file to avoid "Address already in use" errors
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)
                .map_err(|e| EimError::SocketError(format!("Failed to remove existing socket: {}", e)))?;
        }

        // Start the EIM process, passing the socket path as an argument
        let process = std::process::Command::new(path)
            .arg(&socket_path)
            .spawn()
            .map_err(|e| EimError::ExecutionError(e.to_string()))?;

        // Attempt to connect to the socket with retries and timeout
        let stream = Self::connect_with_retry(&socket_path, Duration::from_secs(5))?;

        let mut model = Self {
            path: path.to_path_buf(),
            socket_path,
            stream,
            debug,
            _process: process,
            model_info: None,
            message_id: AtomicU32::new(1),
        };

        // Send initial hello message to establish communication
        model.send_hello()?;

        Ok(model)
    }

    /// Attempts to connect to the Unix socket with a retry mechanism
    ///
    /// This function will repeatedly try to connect to the socket until either:
    /// - A successful connection is established
    /// - An unexpected error occurs
    /// - The timeout duration is exceeded
    ///
    /// # Arguments
    ///
    /// * `socket_path` - Path to the Unix socket
    /// * `timeout` - Maximum time to wait for connection
    fn connect_with_retry(socket_path: &Path, timeout: Duration) -> Result<UnixStream, EimError> {
        let start = Instant::now();
        let retry_interval = Duration::from_millis(50);

        while start.elapsed() < timeout {
            match UnixStream::connect(socket_path) {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    // NotFound and ConnectionRefused are expected errors while the socket
                    // is being created, so we retry in these cases
                    if e.kind() != std::io::ErrorKind::NotFound &&
                       e.kind() != std::io::ErrorKind::ConnectionRefused {
                        return Err(EimError::SocketError(format!("Failed to connect to socket: {}", e)));
                    }
                }
            }
            std::thread::sleep(retry_interval);
        }

        Err(EimError::SocketError(format!(
            "Timeout waiting for socket {} to become available",
            socket_path.display()
        )))
    }

    /// Get the next message ID
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, Ordering::Relaxed)
    }

    fn send_hello(&mut self) -> Result<(), EimError> {
        let hello_msg = HelloMessage {
            hello: 1,
            id: self.next_message_id(),
        };

        let msg = serde_json::to_string(&hello_msg)?;
        if self.debug {
            println!("-> {}", msg);
        }

        writeln!(self.stream, "{}", msg)
            .map_err(|e| EimError::SocketError(format!("Failed to send hello message: {}", e)))?;

        let reader = BufReader::new(&self.stream);
        for line in reader.lines() {
            let line = line.map_err(|e|
                EimError::SocketError(format!("Failed to read response: {}", e)))?;

            if self.debug {
                println!("<- {}", line);
            }

            if let Ok(info) = serde_json::from_str::<ModelInfo>(&line) {
                if !info.success {
                    return Err(EimError::ExecutionError("Model initialization failed".to_string()));
                }
                self.model_info = Some(info);
                return Ok(());
            }

            if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                if !error.success {
                    return Err(EimError::ExecutionError(
                        error.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
            }
        }

        Err(EimError::SocketError("No valid response received".to_string()))
    }

    /// Get the path to the EIM file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the socket path that will be used for communication
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Get the sensor type for this model
    pub fn sensor_type(&self) -> Result<SensorType, EimError> {
        self.model_info.as_ref()
            .map(|info| SensorType::from(info.model_parameters.sensor))
            .ok_or_else(|| EimError::ExecutionError("Model info not available".to_string()))
    }

    /// Get the model parameters
    pub fn parameters(&self) -> Result<&ModelParameters, EimError> {
        self.model_info.as_ref()
            .map(|info| &info.model_parameters)
            .ok_or_else(|| EimError::ExecutionError("Model info not available".to_string()))
    }

    /// Classify raw features
    pub fn classify(&mut self, features: Vec<f32>, debug: Option<bool>) -> Result<InferenceResponse, EimError> {
        // Validate input features match model requirements
        let params = self.parameters()?;
        if features.len() != params.input_features_count as usize {
            return Err(EimError::InvalidInput(format!(
                "Expected {} features but got {}",
                params.input_features_count,
                features.len()
            )));
        }

        let msg = ClassifyMessage {
            classify: features,
            id: self.next_message_id(),
            debug,
        };

        let msg = serde_json::to_string(&msg)?;
        if self.debug {
            println!("-> {}", msg);
        }

        writeln!(self.stream, "{}", msg)
            .map_err(|e| EimError::SocketError(format!("Failed to send classify message: {}", e)))?;

        let reader = BufReader::new(&self.stream);
        for line in reader.lines() {
            let line = line.map_err(|e|
                EimError::SocketError(format!("Failed to read response: {}", e)))?;

            if self.debug {
                println!("<- {}", line);
            }

            // First try to parse as InferenceResponse
            if let Ok(response) = serde_json::from_str::<InferenceResponse>(&line) {
                if !response.success {
                    return Err(EimError::ExecutionError("Inference failed".to_string()));
                }
                return Ok(response);
            }

            // If that fails, check if it's an error response
            if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                if !error.success {
                    return Err(EimError::ExecutionError(
                        error.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
            }
            // Skip other message types
        }

        Err(EimError::SocketError("No inference response received".to_string()))
    }

    /// Classify continuous data (for models that support it)
    pub fn classify_continuous(&mut self, features: Vec<f32>) -> Result<InferenceResponse, EimError> {
        let params = self.parameters()?;

        // Validate model supports continuous mode
        if !params.use_continuous_mode {
            return Err(EimError::InvalidOperation(
                "Model does not support continuous mode".to_string()
            ));
        }

        // Validate slice size if specified
        if let Some(slice_size) = params.slice_size {
            if features.len() != slice_size as usize {
                return Err(EimError::InvalidInput(format!(
                    "Expected slice size of {} but got {}",
                    slice_size,
                    features.len()
                )));
            }
        }

        let msg = ClassifyMessage {
            classify: features,
            id: self.next_message_id(),
            debug: None,
        };

        let msg = serde_json::to_string(&msg)?;
        if self.debug {
            println!("-> {}", msg);
        }

        writeln!(self.stream, "{}", msg)
            .map_err(|e| EimError::SocketError(format!("Failed to send classify message: {}", e)))?;

        let reader = BufReader::new(&self.stream);
        for line in reader.lines() {
            let line = line.map_err(|e|
                EimError::SocketError(format!("Failed to read response: {}", e)))?;

            if self.debug {
                println!("<- {}", line);
            }

            // First try to parse as InferenceResponse
            if let Ok(response) = serde_json::from_str::<InferenceResponse>(&line) {
                if !response.success {
                    return Err(EimError::ExecutionError("Inference failed".to_string()));
                }
                return Ok(response);
            }

            // If that fails, check if it's an error response
            if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                if !error.success {
                    return Err(EimError::ExecutionError(
                        error.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
            }
            // Skip other message types
        }

        Err(EimError::SocketError("No inference response received".to_string()))
    }
}