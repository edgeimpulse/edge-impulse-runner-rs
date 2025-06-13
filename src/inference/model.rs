use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::Child;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::error::EimError;
use crate::inference::messages::{
    ClassifyMessage, ErrorResponse, HelloMessage, InferenceResponse, InferenceResult, ModelInfo,
    SetThresholdMessage, SetThresholdResponse, ThresholdConfig,
};
use crate::types::{ModelParameters, ModelThreshold, SensorType, VisualAnomalyResult};

/// Debug callback type for receiving debug messages
pub type DebugCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Edge Impulse Model Runner for Rust
///
/// This module provides functionality for running Edge Impulse machine learning models on Linux systems.
/// It handles model lifecycle management, communication, and inference operations.
///
/// # Key Components
///
/// - `EimModel`: Main struct for managing Edge Impulse models
/// - `SensorType`: Enum representing supported sensor input types
/// - `ContinuousState`: Internal state management for continuous inference mode
/// - `MovingAverageFilter`: Smoothing filter for continuous inference results
///
/// # Features
///
/// - Model process management and Unix socket communication
/// - Support for both single-shot and continuous inference modes
/// - Debug logging and callback system
/// - Moving average filtering for continuous mode results
/// - Automatic retry mechanisms for socket connections
/// - Visual anomaly detection (FOMO AD) support with normalized scores
///
/// # Example Usage
///
/// ```no_run
/// use edge_impulse_runner::{EimModel, InferenceResult};
///
/// // Create a new model instance
/// let mut model = EimModel::new("path/to/model.eim").unwrap();
///
/// // For RGB images, features must be in 0xRRGGBB format (not normalized to [0,1])
/// let features = vec![0xFF0000 as f32, 0x00FF00 as f32, 0x0000FF as f32]; // Example: Red, Green, Blue pixels
/// let result = model.infer(features, None).unwrap();
///
/// // For visual anomaly detection models, normalize the results
/// if let InferenceResult::VisualAnomaly { anomaly, visual_anomaly_max, visual_anomaly_mean, visual_anomaly_grid } = result.result {
///     let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) =
///         model.normalize_visual_anomaly(
///             anomaly,
///             visual_anomaly_max,
///             visual_anomaly_mean,
///             &visual_anomaly_grid.iter()
///                 .map(|bbox| (bbox.value, bbox.x as u32, bbox.y as u32, bbox.width as u32, bbox.height as u32))
///                 .collect::<Vec<_>>()
///         );
///     println!("Anomaly score: {:.2}%", normalized_anomaly * 100.0);
/// }
/// ```
///
/// # Communication Protocol
///
/// The model communicates with the Edge Impulse process using JSON messages over Unix sockets:
/// 1. Hello message for initialization
/// 2. Model info response
/// 3. Classification requests
/// 4. Inference responses
///
/// # Error Handling
///
/// The module uses a custom `EimError` type for error handling, covering:
/// - Invalid file paths
/// - Socket communication errors
/// - Model execution errors
/// - JSON serialization/deserialization errors
///
/// # Visual Anomaly Detection
///
/// For visual anomaly detection models (FOMO AD):
/// - Scores are normalized relative to the model's minimum anomaly threshold
/// - Results include overall anomaly score, maximum score, mean score, and anomalous regions
/// - Region coordinates are provided in the original image dimensions
/// - All scores are clamped to [0,1] range and displayed as percentages
/// - Debug mode provides detailed information about thresholds and regions
///
/// # Image Feature Format
///
/// For RGB images, features must be in 0xRRGGBB format (not normalized to [0,1]).
/// For grayscale images, features must be in 0xGGGGGG format (repeating the grayscale value).
///
/// # Threshold Configuration
///
/// Models can be configured with different thresholds:
/// - Anomaly detection: `min_anomaly_score` threshold for visual anomaly detection
/// - Object detection: `min_score` threshold for object confidence
/// - Object tracking: `keep_grace`, `max_observations`, and `threshold` parameters
///
/// Thresholds can be updated at runtime using `set_learn_block_threshold`.
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
    model_parameters: ModelParameters,
}

#[derive(Debug)]
struct ContinuousState {
    feature_matrix: Vec<f32>,
    feature_buffer_full: bool,
    maf_buffers: HashMap<String, MovingAverageFilter>,
    slice_size: usize,
}

impl ContinuousState {
    fn new(labels: Vec<String>, slice_size: usize) -> Self {
        Self {
            feature_matrix: Vec::new(),
            feature_buffer_full: false,
            maf_buffers: labels
                .into_iter()
                .map(|label| (label, MovingAverageFilter::new(4)))
                .collect(),
            slice_size,
        }
    }

    fn update_features(&mut self, features: &[f32]) {
        // Add new features to the matrix
        self.feature_matrix.extend_from_slice(features);

        // Check if buffer is full
        if self.feature_matrix.len() >= self.slice_size {
            self.feature_buffer_full = true;
            // Keep only the most recent features if we've exceeded the buffer size
            if self.feature_matrix.len() > self.slice_size {
                self.feature_matrix
                    .drain(0..self.feature_matrix.len() - self.slice_size);
            }
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
    window_size: usize,
    sum: f32,
}

impl MovingAverageFilter {
    fn new(window_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
        }
    }

    fn update(&mut self, value: f32) -> f32 {
        if self.buffer.len() >= self.window_size {
            self.sum -= self.buffer.pop_front().unwrap();
        }
        self.buffer.push_back(value);
        self.sum += value;
        self.sum / self.buffer.len() as f32
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
            .field("model_parameters", &self.model_parameters)
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

    /// Ensure the model file has execution permissions for the current user
    fn ensure_executable<P: AsRef<Path>>(path: P) -> Result<(), EimError> {
        use std::os::unix::fs::PermissionsExt;

        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| EimError::ExecutionError(format!("Failed to get file metadata: {}", e)))?;

        let perms = metadata.permissions();
        let current_mode = perms.mode();
        if current_mode & 0o100 == 0 {
            // File is not executable for user, try to make it executable
            let mut new_perms = perms;
            new_perms.set_mode(current_mode | 0o100); // Add executable bit for user only
            std::fs::set_permissions(path, new_perms).map_err(|e| {
                EimError::ExecutionError(format!("Failed to set executable permissions: {}", e))
            })?;
        }
        Ok(())
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

        // Convert relative path to absolute path
        let absolute_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|_e| EimError::InvalidPath)?
                .join(path)
        };

        // Ensure the model file is executable
        Self::ensure_executable(&absolute_path)?;

        // Start the process
        let process = std::process::Command::new(&absolute_path)
            .arg(socket_path)
            .spawn()
            .map_err(|e| EimError::ExecutionError(e.to_string()))?;

        let socket = Self::connect_with_retry(socket_path, Duration::from_secs(5))?;

        let mut model = Self {
            path: absolute_path, // Store the absolute path
            socket_path: socket_path.to_path_buf(),
            socket,
            debug,
            _process: process,
            model_info: None,
            message_id: AtomicU32::new(1),
            child: None,
            debug_callback: None,
            continuous_state: None,
            model_parameters: ModelParameters::default(),
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
    ///
    /// When debug mode is enabled, this callback will be invoked with debug messages
    /// from the model runner. This is useful for logging or displaying debug information
    /// in your application.
    ///
    /// # Arguments
    ///
    /// * `callback` - Function that takes a string slice and handles the debug message
    pub fn set_debug_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.debug_callback = Some(Box::new(callback));
    }

    /// Send debug messages when debug mode is enabled
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
        self.debug_message(&format!("Sending hello message: {}", msg));

        writeln!(self.socket, "{}", msg).map_err(|e| {
            self.debug_message(&format!("Failed to send hello: {}", e));
            EimError::SocketError(format!("Failed to send hello message: {}", e))
        })?;

        self.socket.flush().map_err(|e| {
            self.debug_message(&format!("Failed to flush hello: {}", e));
            EimError::SocketError(format!("Failed to flush socket: {}", e))
        })?;

        self.debug_message("Waiting for hello response...");

        let mut reader = BufReader::new(&self.socket);
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(n) => {
                self.debug_message(&format!("Read {} bytes: {}", n, line));

                match serde_json::from_str::<ModelInfo>(&line) {
                    Ok(info) => {
                        self.debug_message("Successfully parsed model info");
                        if !info.success {
                            self.debug_message("Model initialization failed");
                            return Err(EimError::ExecutionError(
                                "Model initialization failed".to_string(),
                            ));
                        }
                        self.debug_message("Got model info response, storing it");
                        self.model_info = Some(info);
                        return Ok(());
                    }
                    Err(e) => {
                        self.debug_message(&format!("Failed to parse model info: {}", e));
                        if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                            if !error.success {
                                self.debug_message(&format!("Got error response: {:?}", error));
                                return Err(EimError::ExecutionError(
                                    error.error.unwrap_or_else(|| "Unknown error".to_string()),
                                ));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                self.debug_message(&format!("Failed to read hello response: {}", e));
                return Err(EimError::SocketError(format!(
                    "Failed to read response: {}",
                    e
                )));
            }
        }

        self.debug_message("No valid hello response received");
        Err(EimError::SocketError(
            "No valid response received".to_string(),
        ))
    }

    /// Get the path to the EIM file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the socket path used for communication
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

    /// Run inference on the input features
    ///
    /// This method automatically handles both continuous and non-continuous modes:
    ///
    /// ## Non-Continuous Mode
    /// - Each call is independent
    /// - All features must be provided in a single call
    /// - Results are returned immediately
    ///
    /// ## Continuous Mode (automatically enabled for supported models)
    /// - Features are accumulated across calls
    /// - Internal buffer maintains sliding window of features
    /// - Moving average filter smooths results
    /// - Initial calls may return empty results while buffer fills
    ///
    /// # Arguments
    ///
    /// * `features` - Vector of input features
    /// * `debug` - Optional debug flag to enable detailed output for this inference
    ///
    /// # Returns
    ///
    /// Returns `Result<InferenceResponse, EimError>` containing inference results
    pub fn infer(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError> {
        // Initialize model info if needed
        if self.model_info.is_none() {
            self.send_hello()?;
        }

        let uses_continuous_mode = self.requires_continuous_mode();

        if uses_continuous_mode {
            self.infer_continuous_internal(features, debug)
        } else {
            self.infer_single(features, debug)
        }
    }

    fn infer_continuous_internal(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError> {
        // Initialize continuous state if needed
        if self.continuous_state.is_none() {
            let labels = self
                .model_info
                .as_ref()
                .map(|info| info.model_parameters.labels.clone())
                .unwrap_or_default();
            let slice_size = self.input_size()?;

            self.continuous_state = Some(ContinuousState::new(labels, slice_size));
        }

        // Take ownership of state temporarily to avoid multiple mutable borrows
        let mut state = self.continuous_state.take().unwrap();
        state.update_features(&features);

        let response = if !state.feature_buffer_full {
            // Return empty response while building up the buffer
            Ok(InferenceResponse {
                success: true,
                id: self.next_message_id(),
                result: InferenceResult::Classification {
                    classification: HashMap::new(),
                },
            })
        } else {
            // Run inference on the full buffer
            let mut response = self.infer_single(state.feature_matrix.clone(), debug)?;

            // Apply moving average filter to the results
            if let InferenceResult::Classification {
                ref mut classification,
            } = response.result
            {
                state.apply_maf(classification);
            }

            Ok(response)
        };

        // Restore the state
        self.continuous_state = Some(state);

        response
    }

    fn infer_single(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EimError> {
        // First ensure we've sent the hello message and received model info
        if self.model_info.is_none() {
            self.debug_message("No model info, sending hello message...");
            self.send_hello()?;
            self.debug_message("Hello handshake completed");
        }

        let msg = ClassifyMessage {
            classify: features.clone(),
            id: self.next_message_id(),
            debug,
        };

        let msg_str = serde_json::to_string(&msg)?;
        self.debug_message(&format!(
            "Sending inference message with {} features",
            features.len()
        ));

        writeln!(self.socket, "{}", msg_str).map_err(|e| {
            self.debug_message(&format!("Failed to send inference message: {}", e));
            EimError::SocketError(format!("Failed to send inference message: {}", e))
        })?;

        self.socket.flush().map_err(|e| {
            self.debug_message(&format!("Failed to flush inference message: {}", e));
            EimError::SocketError(format!("Failed to flush socket: {}", e))
        })?;

        self.debug_message("Inference message sent, waiting for response...");

        // Set socket to non-blocking mode
        self.socket.set_nonblocking(true).map_err(|e| {
            self.debug_message(&format!("Failed to set non-blocking mode: {}", e));
            EimError::SocketError(format!("Failed to set non-blocking mode: {}", e))
        })?;

        let mut reader = BufReader::new(&self.socket);
        let mut buffer = String::new();
        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while start.elapsed() < timeout {
            match reader.read_line(&mut buffer) {
                Ok(0) => {
                    self.debug_message("EOF reached");
                    break;
                }
                Ok(n) => {
                    // Skip printing feature values in the response
                    if !buffer.contains("features:") && !buffer.contains("Features (") {
                        self.debug_message(&format!("Read {} bytes: {}", n, buffer));
                    }

                    if let Ok(response) = serde_json::from_str::<InferenceResponse>(&buffer) {
                        if response.success {
                            self.debug_message("Got successful inference response");
                            // Reset to blocking mode before returning
                            let _ = self.socket.set_nonblocking(false);
                            return Ok(response);
                        }
                    }
                    buffer.clear();
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No data available yet, sleep briefly and retry
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => {
                    self.debug_message(&format!("Read error: {}", e));
                    // Always try to reset blocking mode, even on error
                    let _ = self.socket.set_nonblocking(false);
                    return Err(EimError::SocketError(format!("Read error: {}", e)));
                }
            }
        }

        // Reset to blocking mode before returning
        let _ = self.socket.set_nonblocking(false);
        self.debug_message("Timeout reached");

        Err(EimError::ExecutionError(format!(
            "No valid response received within {} seconds",
            timeout.as_secs()
        )))
    }

    /// Check if model requires continuous mode
    fn requires_continuous_mode(&self) -> bool {
        self.model_info
            .as_ref()
            .map(|info| info.model_parameters.use_continuous_mode)
            .unwrap_or(false)
    }

    /// Get the required number of input features for this model
    ///
    /// Returns the number of features expected by the model for each classification.
    /// This is useful for:
    /// - Validating input size before classification
    /// - Preparing the correct amount of data
    /// - Padding or truncating inputs to match model requirements
    ///
    /// # Returns
    ///
    /// The number of input features required by the model
    pub fn input_size(&self) -> Result<usize, EimError> {
        self.model_info
            .as_ref()
            .map(|info| info.model_parameters.input_features_count as usize)
            .ok_or_else(|| EimError::ExecutionError("Model info not available".to_string()))
    }

    /// Set a threshold for a specific learning block
    ///
    /// This method allows updating thresholds for different types of blocks:
    /// - Anomaly detection (GMM)
    /// - Object detection
    /// - Object tracking
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold configuration to set
    ///
    /// # Returns
    ///
    /// Returns `Result<(), EimError>` indicating success or failure
    pub async fn set_learn_block_threshold(
        &mut self,
        threshold: ThresholdConfig,
    ) -> Result<(), EimError> {
        // First check if model info is available and supports thresholds
        if self.model_info.is_none() {
            self.debug_message("No model info available, sending hello message...");
            self.send_hello()?;
        }

        // Log the current model state
        if let Some(info) = &self.model_info {
            self.debug_message(&format!(
                "Current model type: {}",
                info.model_parameters.model_type
            ));
            self.debug_message(&format!(
                "Current model parameters: {:?}",
                info.model_parameters
            ));
        }

        let msg = SetThresholdMessage {
            set_threshold: threshold,
            id: self.next_message_id(),
        };

        let msg_str = serde_json::to_string(&msg)?;
        self.debug_message(&format!("Sending threshold message: {}", msg_str));

        writeln!(self.socket, "{}", msg_str).map_err(|e| {
            self.debug_message(&format!("Failed to send threshold message: {}", e));
            EimError::SocketError(format!("Failed to send threshold message: {}", e))
        })?;

        self.socket.flush().map_err(|e| {
            self.debug_message(&format!("Failed to flush threshold message: {}", e));
            EimError::SocketError(format!("Failed to flush socket: {}", e))
        })?;

        let mut reader = BufReader::new(&self.socket);
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(_) => {
                self.debug_message(&format!("Received response: {}", line));
                match serde_json::from_str::<SetThresholdResponse>(&line) {
                    Ok(response) => {
                        if response.success {
                            self.debug_message("Successfully set threshold");
                            Ok(())
                        } else {
                            self.debug_message("Server reported failure setting threshold");
                            Err(EimError::ExecutionError(
                                "Server reported failure setting threshold".to_string(),
                            ))
                        }
                    }
                    Err(e) => {
                        self.debug_message(&format!("Failed to parse threshold response: {}", e));
                        // Try to parse as error response
                        if let Ok(error) = serde_json::from_str::<ErrorResponse>(&line) {
                            Err(EimError::ExecutionError(
                                error.error.unwrap_or_else(|| "Unknown error".to_string()),
                            ))
                        } else {
                            Err(EimError::ExecutionError(format!(
                                "Invalid threshold response format: {}",
                                e
                            )))
                        }
                    }
                }
            }
            Err(e) => {
                self.debug_message(&format!("Failed to read threshold response: {}", e));
                Err(EimError::SocketError(format!(
                    "Failed to read response: {}",
                    e
                )))
            }
        }
    }

    /// Get the minimum anomaly score threshold from model parameters
    fn get_min_anomaly_score(&self) -> f32 {
        self.model_info
            .as_ref()
            .and_then(|info| {
                info.model_parameters
                    .thresholds
                    .iter()
                    .find_map(|t| match t {
                        ModelThreshold::AnomalyGMM {
                            min_anomaly_score, ..
                        } => Some(*min_anomaly_score),
                        _ => None,
                    })
            })
            .unwrap_or(6.0)
    }

    /// Normalize an anomaly score relative to the model's minimum threshold
    fn normalize_anomaly_score(&self, score: f32) -> f32 {
        (score / self.get_min_anomaly_score()).min(1.0)
    }

    /// Normalize a visual anomaly result
    pub fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> VisualAnomalyResult {
        let normalized_anomaly = self.normalize_anomaly_score(anomaly);
        let normalized_max = self.normalize_anomaly_score(max);
        let normalized_mean = self.normalize_anomaly_score(mean);
        let normalized_regions: Vec<_> = regions
            .iter()
            .map(|(value, x, y, w, h)| (self.normalize_anomaly_score(*value), *x, *y, *w, *h))
            .collect();

        (
            normalized_anomaly,
            normalized_max,
            normalized_mean,
            normalized_regions,
        )
    }
}
