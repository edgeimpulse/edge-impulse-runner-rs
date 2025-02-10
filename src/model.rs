use std::io::{BufReader, Write, BufRead};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::EimError;
use crate::messages::{
    HelloMessage, ModelInfo, ErrorResponse, ClassifyMessage,
    ConfigMessage, ConfigOptions, ConfigResponse, InferenceResponse
};
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
    #[allow(dead_code)]
    child: Option<Child>,
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
            child: None,
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
        // Send the classification request
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

        // Read responses until we get a classification result
        let mut reader = BufReader::new(&self.stream);
        let mut buffer = String::new();

        while reader.read_line(&mut buffer)? > 0 {
            if self.debug {
                println!("<- {}", buffer);
            }

            if let Ok(response) = serde_json::from_str::<InferenceResponse>(&buffer) {
                if response.success {
                    return Ok(response);
                }
            }

            buffer.clear();
        }

        Err(EimError::ExecutionError("No valid response received".to_string()))
    }

    /// Classify continuous data (for models that support it)
    pub fn classify_continuous(&mut self, features: Vec<f32>) -> Result<InferenceResponse, EimError> {
        self.classify(features, None)
    }

    /// Set whether to use continuous mode for classification
    pub fn set_continuous_mode(&mut self, enable: bool) -> Result<(), EimError> {
        let msg = ConfigMessage {
            config: ConfigOptions {
                continuous_mode: Some(enable),
            },
            id: self.next_message_id(),
        };

        let msg = serde_json::to_string(&msg)?;
        if self.debug {
            println!("-> {}", msg);
        }

        writeln!(self.stream, "{}", msg)
            .map_err(|e| EimError::SocketError(format!("Failed to send config message: {}", e)))?;

        let reader = BufReader::new(&self.stream);
        for line in reader.lines() {
            let line = line.map_err(|e|
                EimError::SocketError(format!("Failed to read response: {}", e)))?;

            if self.debug {
                println!("<- {}", line);
            }

            if let Ok(response) = serde_json::from_str::<ConfigResponse>(&line) {
                if !response.success {
                    return Err(EimError::ExecutionError("Failed to set configuration".to_string()));
                }
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

        Ok(())
    }

    #[allow(dead_code)]
    fn start_process(path: &Path, _debug: bool) -> Result<(Child, UnixStream), EimError> {
        // Create a temporary socket path in the system's temp directory
        let socket_path = std::env::temp_dir().join("eim_socket");

        // Remove any existing socket file to avoid "Address already in use" errors
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)
                .map_err(|e| EimError::SocketError(format!("Failed to remove existing socket: {}", e)))?;
        }

        // Start the EIM process, passing the socket path as an argument
        let child = Command::new(path)
            .arg(&socket_path)
            .spawn()
            .map_err(|e| EimError::ExecutionError(e.to_string()))?;

        // Attempt to connect to the socket with retries and timeout
        let stream = Self::connect_with_retry(&socket_path, Duration::from_secs(5))?;

        Ok((child, stream))
    }

    #[allow(dead_code)]
    fn restart(&mut self) -> Result<(), EimError> {
        // Kill the current process
        if let Some(mut child) = self.child.take() {
            child.kill()
                .map_err(|e| EimError::ExecutionError(format!("Failed to kill process: {}", e)))?;
            child.wait()
                .map_err(|e| EimError::ExecutionError(format!("Failed to wait for process: {}", e)))?;
        }

        // Start a new process
        let (child, stream) = Self::start_process(&self.path, self.debug)?;

        self.child = Some(child);
        self.stream = stream;

        // Send hello message to initialize
        self.send_hello()?;

        Ok(())
    }
}