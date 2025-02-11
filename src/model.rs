use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use std::fmt;
use std::collections::{VecDeque, HashMap};

use crate::error::EimError;
use crate::messages::{
    ClassifyMessage, ConfigMessage, ConfigOptions, ConfigResponse, ErrorResponse, HelloMessage,
    InferenceResponse, ModelInfo, InferenceResult,
};
use crate::types::ModelParameters;

/// Supported sensor types for Edge Impulse models.
///
/// These represent the different types of input data that an Edge Impulse model
/// can process. Each sensor type corresponds to a specific data collection method
/// and processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorType {
    /// Represents an unknown or unsupported sensor type
    Unknown,
    /// Audio input from microphone sensors
    Microphone,
    /// Motion data from accelerometer sensors
    Accelerometer,
    /// Visual input from camera sensors
    Camera,
    /// Location or orientation data from positional sensors
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

/// Debug callback type for receiving debug messages
pub type DebugCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Edge Impulse Model runner for Linux-based systems.
///
/// This struct manages the lifecycle of an Edge Impulse model, handling:
/// - Model process spawning and management
/// - Unix socket communication
/// - Model configuration
/// - Inference requests and responses
///
/// # Communication Protocol
///
/// The model runner communicates with the Edge Impulse model process through a Unix
/// socket. The protocol uses JSON messages for all communications, including:
/// - Hello messages for initialization
/// - Configuration messages
/// - Classification requests
/// - Inference responses
pub struct EimModel {
    /// Path to the Edge Impulse model file (.eim)
    path: std::path::PathBuf,
    /// Path to the Unix socket used for IPC
    socket_path: std::path::PathBuf,
    /// Active Unix socket connection to the model process
    socket: UnixStream,
    /// Enable debug logging of socket communications
    debug: bool,
    /// Optional debug callback for receiving debug messages
    debug_callback: Option<DebugCallback>,
    /// Handle to the model process (kept alive while model exists)
    _process: Child,
    /// Cached model information received during initialization
    model_info: Option<ModelInfo>,
    /// Atomic counter for generating unique message IDs
    message_id: AtomicU32,
    /// Optional child process handle for restart functionality
    #[allow(dead_code)]
    child: Option<Child>,
    continuous_state: Option<ContinuousState>,
}

#[derive(Debug)]
struct ContinuousState {
    feature_matrix: Vec<f32>,
    slice_offset: usize,
    feature_buffer_full: bool,
    maf_buffers: HashMap<String, MovingAverageFilter>,
}

impl ContinuousState {
    fn new(labels: Vec<String>) -> Self {
        Self {
            feature_matrix: Vec::new(),
            slice_offset: 0,
            feature_buffer_full: false,
            maf_buffers: labels.into_iter()
                .map(|label| (label, MovingAverageFilter::new(4)))
                .collect(),
        }
    }

    fn update_features(&mut self, features: &[f32], feature_size: usize) {
        // Add features to the matrix at the current slice offset
        while self.feature_matrix.len() < self.slice_offset + feature_size {
            self.feature_matrix.push(0.0);
        }

        for (i, &value) in features.iter().enumerate() {
            self.feature_matrix[self.slice_offset + i] = value;
        }

        // Update slice offset and buffer full status
        if !self.feature_buffer_full {
            self.slice_offset += feature_size;
            if self.slice_offset > (self.feature_matrix.len() - feature_size) {
                self.feature_buffer_full = true;
                self.slice_offset -= feature_size;
            }
        }
    }

    fn shift_features(&mut self, feature_size: usize) {
        for i in 0..(self.feature_matrix.len() - feature_size) {
            self.feature_matrix[i] = self.feature_matrix[i + feature_size];
        }
    }

    fn apply_maf(&mut self, classification: &mut HashMap<String, f32>) {
        for (label, value) in classification.iter_mut() {
            if let Some(maf) = self.maf_buffers.get_mut(label) {
                *value = maf.update(*value);
            }
        }
    }
}

#[derive(Debug)]
struct MovingAverageFilter {
    buffer: VecDeque<f32>,
    running_sum: f32,
}

impl MovingAverageFilter {
    fn new(window_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(window_size),
            running_sum: 0.0,
        }
    }

    fn update(&mut self, value: f32) -> f32 {
        if self.buffer.len() >= self.buffer.capacity() {
            if let Some(old) = self.buffer.pop_front() {
                self.running_sum -= old;
            }
        }
        self.buffer.push_back(value);
        self.running_sum += value;

        self.running_sum / self.buffer.len() as f32
    }
}

impl fmt::Debug for EimModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EimModel")
            .field("path", &self.path)
            .field("socket_path", &self.socket_path)
            .field("socket", &self.socket)
            .field("debug", &self.debug)
            .field("_process", &self._process)
            .field("model_info", &self.model_info)
            .field("message_id", &self.message_id)
            .field("child", &self.child)
            // Skip debug_callback field as it doesn't implement Debug
            .field("continuous_state", &self.continuous_state)
            .finish()
    }
}

impl EimModel {
    /// Creates a new EimModel instance from a path to the .eim file.
    ///
    /// This is the standard way to create a new model instance. The function will:
    /// 1. Validate the file extension
    /// 2. Spawn the model process
    /// 3. Establish socket communication
    /// 4. Initialize the model
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .eim file. Must be a valid Edge Impulse model file.
    ///
    /// # Returns
    ///
    /// Returns `Result<EimModel, EimError>` where:
    /// - `Ok(EimModel)` - Successfully created and initialized model
    /// - `Err(EimError)` - Failed to create model (invalid path, process spawn failure, etc.)
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

    /// Creates a new EimModel instance with a specific Unix socket path.
    ///
    /// Similar to `new()`, but allows specifying the socket path for communication.
    /// This is useful when you need control over the socket location or when running
    /// multiple models simultaneously.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .eim file
    /// * `socket_path` - Custom path where the Unix socket should be created
    pub fn new_with_socket<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        socket_path: S,
    ) -> Result<Self, EimError> {
        Self::new_with_socket_and_debug(path, socket_path, false)
    }

    /// Create a new EimModel instance with debug output enabled
    pub fn new_with_debug<P: AsRef<Path>>(path: P, debug: bool) -> Result<Self, EimError> {
        let socket_path = std::env::temp_dir().join("eim_socket");
        Self::new_with_socket_and_debug(path, &socket_path, debug)
    }

    /// Create a new EimModel instance with debug output enabled and a specific socket path
    pub fn new_with_socket_and_debug<P: AsRef<Path>, S: AsRef<Path>>(
        path: P,
        socket_path: S,
        debug: bool,
    ) -> Result<Self, EimError> {
        let path = path.as_ref();
        let socket_path = socket_path.as_ref();

        // Validate file extension
        if path.extension().and_then(|s| s.to_str()) != Some("eim") {
            return Err(EimError::InvalidPath);
        }

        // Start the process
        let process = std::process::Command::new(path)
            .arg(socket_path)
            .spawn()
            .map_err(|e| EimError::ExecutionError(e.to_string()))?;

        let socket = Self::connect_with_retry(socket_path, Duration::from_secs(5))?;

        let mut model = Self {
            path: path.to_path_buf(),
            socket_path: socket_path.to_path_buf(),
            socket,
            debug,
            _process: process,
            model_info: None,
            message_id: AtomicU32::new(1),
            child: None,
            debug_callback: None,
            continuous_state: None,
        };

        // Initialize the model by sending hello message
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
                    if e.kind() != std::io::ErrorKind::NotFound
                        && e.kind() != std::io::ErrorKind::ConnectionRefused
                    {
                        return Err(EimError::SocketError(format!(
                            "Failed to connect to socket: {}",
                            e
                        )));
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

    /// Set a debug callback function to receive debug messages
    pub fn set_debug_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.debug_callback = Some(Box::new(callback));
    }

    /// Internal helper to send debug messages
    fn debug_message(&self, message: &str) {
        if self.debug {
            println!("{}", message);
            if let Some(callback) = &self.debug_callback {
                callback(message);
            }
        }
    }

    fn send_hello(&mut self) -> Result<(), EimError> {
        let hello_msg = HelloMessage {
            hello: 1,
            id: self.next_message_id(),
        };

        let msg = serde_json::to_string(&hello_msg)?;
        println!("Sending hello message: {}", msg);

        writeln!(self.socket, "{}", msg)
            .map_err(|e| {
                println!("Failed to send hello: {}", e);
                EimError::SocketError(format!("Failed to send hello message: {}", e))
            })?;

        self.socket.flush().map_err(|e| {
            println!("Failed to flush hello: {}", e);
            EimError::SocketError(format!("Failed to flush socket: {}", e))
        })?;

        println!("Waiting for hello response...");

        let mut reader = BufReader::new(&self.socket);
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(n) => {
                println!("Read {} bytes: {}", n, line);

                match serde_json::from_str::<ModelInfo>(&line) {
                    Ok(info) => {
                        println!("Successfully parsed model info");
                        if !info.success {
                            println!("Model initialization failed");
                            return Err(EimError::ExecutionError(
                                "Model initialization failed".to_string(),
                            ));
                        }
                        println!("Got model info response, storing it");
                        self.model_info = Some(info);
                        return Ok(());
                    }
                    Err(e) => {
                        println!("Failed to parse model info: {}", e);
                        if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                            if !error.success {
                                println!("Got error response: {:?}", error);
                                return Err(EimError::ExecutionError(
                                    error.error.unwrap_or_else(|| "Unknown error".to_string()),
                                ));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("Failed to read hello response: {}", e);
                return Err(EimError::SocketError(format!("Failed to read response: {}", e)));
            }
        }

        println!("No valid hello response received");
        Err(EimError::SocketError(
            "No valid response received".to_string(),
        ))
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
        self.model_info
            .as_ref()
            .map(|info| SensorType::from(info.model_parameters.sensor))
            .ok_or_else(|| EimError::ExecutionError("Model info not available".to_string()))
    }

    /// Get the model parameters
    pub fn parameters(&self) -> Result<&ModelParameters, EimError> {
        self.model_info
            .as_ref()
            .map(|info| &info.model_parameters)
            .ok_or_else(|| EimError::ExecutionError("Model info not available".to_string()))
    }

    /// Regular classification - processes input features directly
    pub fn classify(&mut self, features: Vec<f32>, debug: Option<bool>) -> Result<InferenceResponse, EimError> {
        // First ensure we've sent the hello message and received model info
        if self.model_info.is_none() {
            println!("No model info, sending hello message...");  // Debug print
            self.send_hello()?;
            println!("Hello handshake completed");  // Debug print
        }

        let msg = ClassifyMessage {
            classify: features,
            id: self.next_message_id(),
            debug,
        };

        let msg = serde_json::to_string(&msg)?;
        println!("Sending classification message: {}", msg);  // Debug print

        writeln!(self.socket, "{}", msg).map_err(|e| {
            println!("Failed to send classification message: {}", e);  // Debug print
            EimError::SocketError(format!("Failed to send classify message: {}", e))
        })?;

        self.socket.flush().map_err(|e| {
            println!("Failed to flush classification message: {}", e);  // Debug print
            EimError::SocketError(format!("Failed to flush socket: {}", e))
        })?;

        println!("Classification message sent, waiting for response...");  // Debug print

        // Try reading response in blocking mode
        let mut reader = BufReader::new(&self.socket);
        let mut buffer = String::new();

        match reader.read_line(&mut buffer) {
            Ok(n) => {
                println!("Read {} bytes: {}", n, buffer);  // Debug print
                if let Ok(response) = serde_json::from_str::<InferenceResponse>(&buffer) {
                    if response.success {
                        println!("Got successful classification response");  // Debug print
                        return Ok(response);
                    }
                }
                println!("Response parsing failed: {}", buffer);  // Debug print
            }
            Err(e) => {
                println!("Error reading classification response: {}", e);  // Debug print
                return Err(EimError::SocketError(format!("Read error: {}", e)));
            }
        }

        println!("No valid classification response received");  // Debug print
        Err(EimError::ExecutionError("No valid response received".to_string()))
    }

    /// Continuous classification with buffering and smoothing
    pub fn classify_continuous(
        &mut self,
        features: Vec<f32>,
    ) -> Result<InferenceResponse, EimError> {
        // Initialize continuous state if needed
        if self.continuous_state.is_none() {
            let labels = self.model_info.as_ref()
                .map(|info| info.model_parameters.labels.clone())
                .unwrap_or_default();

            self.continuous_state = Some(ContinuousState::new(labels));
        }

        let feature_size = features.len();

        // Take ownership of the state temporarily
        let mut state = self.continuous_state.take().unwrap();

        // Update feature matrix with new data
        state.update_features(&features, feature_size);

        let result = if state.feature_buffer_full {
            // Run inference
            let mut result = self.classify(state.feature_matrix.clone(), None)?;

            // Apply moving average filter to smooth results
            if let InferenceResult::Classification { classification } = &mut result.result {
                for (label, value) in classification.iter_mut() {
                    if let Some(maf) = state.maf_buffers.get_mut(label) {
                        *value = maf.update(*value);
                    }
                }
            }

            // Shift feature buffer
            for i in 0..(state.feature_matrix.len() - feature_size) {
                state.feature_matrix[i] = state.feature_matrix[i + feature_size];
            }

            result
        } else {
            // Buffer not full yet
            InferenceResponse {
                success: true,
                id: self.next_message_id(),
                result: InferenceResult::Classification {
                    classification: HashMap::new(),
                },
            }
        };

        // Put the state back
        self.continuous_state = Some(state);

        Ok(result)
    }

    /// Stop continuous classification and clear buffers
    pub fn stop_continuous(&mut self) {
        self.continuous_state = None;
    }

    /// Set whether to use continuous mode for classification
    pub fn set_continuous_mode(&mut self, enable: bool) -> Result<(), EimError> {
        let msg = ConfigMessage {
            config: ConfigOptions {
                continuous_mode: Some(enable),
            },
            id: self.next_message_id(),
        };

        let msg_str = serde_json::to_string(&msg)?;
        self.debug_message(&format!("-> {}", msg_str));

        writeln!(self.socket, "{}", msg_str)
            .map_err(|e| EimError::SocketError(format!("Failed to send config message: {}", e)))?;

        let reader = BufReader::new(&self.socket);
        for line in reader.lines() {
            let line =
                line.map_err(|e| EimError::SocketError(format!("Failed to read response: {}", e)))?;

            self.debug_message(&format!("<- {}", line));

            if let Ok(response) = serde_json::from_str::<ConfigResponse>(&line) {
                if !response.success {
                    return Err(EimError::ExecutionError(format!(
                        "Failed to set configuration. Request: {}, Response: {}",
                        msg_str,
                        line
                    )));
                }
                return Ok(());
            }

            if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                if !error.success {
                    return Err(EimError::ExecutionError(format!(
                        "Failed to set configuration. Request: {}, Error Response: {}, Error Details: {}",
                        msg_str,
                        line,
                        error.error.unwrap_or_else(|| "No error details provided".to_string())
                    )));
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
            std::fs::remove_file(&socket_path).map_err(|e| {
                EimError::SocketError(format!("Failed to remove existing socket: {}", e))
            })?;
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
            child
                .kill()
                .map_err(|e| EimError::ExecutionError(format!("Failed to kill process: {}", e)))?;
            child.wait().map_err(|e| {
                EimError::ExecutionError(format!("Failed to wait for process: {}", e))
            })?;
        }

        // Start a new process
        let (child, stream) = Self::start_process(&self.path, self.debug)?;

        self.child = Some(child);
        self.socket = stream;

        // Send hello message to initialize
        self.send_hello()?;

        Ok(())
    }
}
