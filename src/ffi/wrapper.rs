//! Safe Rust wrapper for Edge Impulse FFI bindings
//!
//! This module provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing you to run inference on trained models from Rust applications.

use std::error::Error;
use std::fmt;

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.4}", self.label, self.value)
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.4} (x={}, y={}, w={}, h={})",
            self.label, self.value, self.x, self.y, self.width, self.height
        )
    }
}

impl fmt::Display for TimingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Timing: dsp={} ms, classification={} ms, anomaly={} ms",
            self.dsp, self.classification, self.anomaly
        )
    }
}

/// Error type for Edge Impulse operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeImpulseError {
    Ok,
    ShapesDontMatch,
    Canceled,
    MemoryAllocationFailed,
    OutOfMemory,
    InputTensorWasNull,
    OutputTensorWasNull,
    AllocatedTensorWasNull,
    TfliteError,
    TfliteArenaAllocFailed,
    ReadSensor,
    MinSizeRatio,
    MaxSizeRatio,
    OnlySupportImages,
    ModelInputTensorWasNull,
    ModelOutputTensorWasNull,
    UnsupportedInferencingEngine,
    AllocWhileCacheLocked,
    NoValidImpulse,
    Other,
}

#[cfg(feature = "ffi")]
impl From<edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR> for EdgeImpulseError {
    fn from(error: edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR) -> Self {
        match error {
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK => EdgeImpulseError::Ok,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ERROR_SHAPES_DONT_MATCH => {
                EdgeImpulseError::ShapesDontMatch
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_CANCELED => EdgeImpulseError::Canceled,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ALLOC_FAILED => EdgeImpulseError::MemoryAllocationFailed,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OUT_OF_MEMORY => EdgeImpulseError::OutOfMemory,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::InputTensorWasNull
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::OutputTensorWasNull
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ERROR => EdgeImpulseError::TfliteError,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED => {
                EdgeImpulseError::TfliteArenaAllocFailed
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_DSP_ERROR => EdgeImpulseError::ReadSensor,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INVALID_SIZE => EdgeImpulseError::MinSizeRatio,
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES => {
                EdgeImpulseError::OnlySupportImages
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE => {
                EdgeImpulseError::UnsupportedInferencingEngine
            }
            edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_INFERENCE_ERROR => EdgeImpulseError::NoValidImpulse,
            _ => EdgeImpulseError::Other,
        }
    }
}

impl fmt::Display for EdgeImpulseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeImpulseError::Ok => write!(f, "Operation completed successfully"),
            EdgeImpulseError::ShapesDontMatch => {
                write!(f, "Input shapes don't match expected dimensions")
            }
            EdgeImpulseError::Canceled => write!(f, "Operation was canceled"),
            EdgeImpulseError::MemoryAllocationFailed => write!(f, "Memory allocation failed"),
            EdgeImpulseError::OutOfMemory => write!(f, "Out of memory"),
            EdgeImpulseError::InputTensorWasNull => write!(f, "Input tensor was null"),
            EdgeImpulseError::OutputTensorWasNull => write!(f, "Output tensor was null"),
            EdgeImpulseError::AllocatedTensorWasNull => write!(f, "Allocated tensor was null"),
            EdgeImpulseError::TfliteError => write!(f, "TensorFlow Lite error"),
            EdgeImpulseError::TfliteArenaAllocFailed => {
                write!(f, "TensorFlow Lite arena allocation failed")
            }
            EdgeImpulseError::ReadSensor => write!(f, "Error reading sensor data"),
            EdgeImpulseError::MinSizeRatio => write!(f, "Minimum size ratio not met"),
            EdgeImpulseError::MaxSizeRatio => write!(f, "Maximum size ratio exceeded"),
            EdgeImpulseError::OnlySupportImages => write!(f, "Only image input is supported"),
            EdgeImpulseError::ModelInputTensorWasNull => write!(f, "Model input tensor was null"),
            EdgeImpulseError::ModelOutputTensorWasNull => write!(f, "Model output tensor was null"),
            EdgeImpulseError::UnsupportedInferencingEngine => {
                write!(f, "Unsupported inferencing engine")
            }
            EdgeImpulseError::AllocWhileCacheLocked => {
                write!(f, "Allocation attempted while cache is locked")
            }
            EdgeImpulseError::NoValidImpulse => write!(f, "No valid impulse found"),
            EdgeImpulseError::Other => write!(f, "Unknown error occurred"),
        }
    }
}

impl Error for EdgeImpulseError {}

/// Result type for Edge Impulse operations
pub type EdgeImpulseResult<T> = Result<T, EdgeImpulseError>;

/// Opaque handle for Edge Impulse operations
pub struct EdgeImpulseHandle {
    #[cfg(feature = "ffi")]
    handle: *mut edge_impulse_ffi_rs::bindings::ei_impulse_handle_t,
}

impl EdgeImpulseHandle {
    /// Create a new Edge Impulse handle
    pub fn new() -> EdgeImpulseResult<Self> {
        #[cfg(feature = "ffi")]
        {
            let handle_ptr =
                std::ptr::null_mut::<edge_impulse_ffi_rs::bindings::ei_impulse_handle_t>();
            let result = unsafe { edge_impulse_ffi_rs::bindings::ei_ffi_init_impulse(handle_ptr) };
            let error = EdgeImpulseError::from(result);
            if error != EdgeImpulseError::Ok {
                return Err(error);
            }

            Ok(Self { handle: handle_ptr })
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }
}

impl Drop for EdgeImpulseHandle {
    fn drop(&mut self) {
        // Cleanup if needed
    }
}

/// Signal structure for holding audio or sensor data
pub struct Signal {
    #[cfg(feature = "ffi")]
    c_signal: Box<edge_impulse_ffi_rs::bindings::ei_signal_t>,
}

impl Signal {
    /// Create a new signal from raw data (f32 slice) using the SDK's signal_from_buffer
    pub fn from_raw_data(_data: &[f32]) -> EdgeImpulseResult<Self> {
        #[cfg(feature = "ffi")]
        {
            let mut c_signal = Box::new(edge_impulse_ffi_rs::bindings::ei_signal_t {
                get_data: [0u64; 4], // Initialize with zeros for std::function
                total_length: 0,
            });
            let result = unsafe {
                edge_impulse_ffi_rs::bindings::ei_ffi_signal_from_buffer(
                    _data.as_ptr(),
                    _data.len(),
                    &mut *c_signal,
                )
            };

            if result == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
                Ok(Self { c_signal })
            } else {
                Err(EdgeImpulseError::from(result))
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }

    #[cfg(feature = "ffi")]
    pub fn as_ptr(&self) -> *mut edge_impulse_ffi_rs::bindings::ei_signal_t {
        Box::as_ref(&self.c_signal) as *const edge_impulse_ffi_rs::bindings::ei_signal_t
            as *mut edge_impulse_ffi_rs::bindings::ei_signal_t
    }
}

/// Result structure for Edge Impulse inference
pub struct InferenceResult {
    #[cfg(feature = "ffi")]
    result: *mut edge_impulse_ffi_rs::bindings::ei_impulse_result_t,
}

impl Default for InferenceResult {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceResult {
    /// Create a new inference result
    pub fn new() -> Self {
        #[cfg(feature = "ffi")]
        {
            let result = unsafe {
                std::alloc::alloc_zeroed(std::alloc::Layout::new::<
                    edge_impulse_ffi_rs::bindings::ei_impulse_result_t,
                >()) as *mut edge_impulse_ffi_rs::bindings::ei_impulse_result_t
            };
            Self { result }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Self {}
        }
    }

    /// Get a raw pointer to the underlying ei_impulse_result_t (for advanced result parsing)
    #[cfg(feature = "ffi")]
    pub fn as_ptr(&self) -> *const edge_impulse_ffi_rs::bindings::ei_impulse_result_t {
        self.result as *const edge_impulse_ffi_rs::bindings::ei_impulse_result_t
    }

    /// Get a mutable raw pointer to the underlying ei_impulse_result_t (for advanced result parsing)
    #[cfg(feature = "ffi")]
    pub fn as_mut_ptr(&mut self) -> *mut edge_impulse_ffi_rs::bindings::ei_impulse_result_t {
        self.result
    }

    /// Get all classification results as safe Rust structs
    pub fn classifications(&self, _label_count: usize) -> Vec<ClassificationResult> {
        #[cfg(feature = "ffi")]
        {
            unsafe {
                let result = &*self.result;
                (0.._label_count)
                    .map(|i| {
                        let c = result.classification[i];
                        let label = if !c.label.is_null() {
                            std::ffi::CStr::from_ptr(c.label)
                                .to_string_lossy()
                                .into_owned()
                        } else {
                            eprintln!("[EdgeImpulse FFI] Warning: classification label pointer is null at index {i}");
                            String::new()
                        };
                        ClassificationResult {
                            label,
                            value: c.value,
                        }
                    })
                    .collect()
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            vec![]
        }
    }

    /// Get all bounding boxes as safe Rust structs
    pub fn bounding_boxes(&self) -> Vec<BoundingBox> {
        #[cfg(feature = "ffi")]
        {
            unsafe {
                let result = &*self.result;
                if result.bounding_boxes_count == 0 || result.bounding_boxes.is_null() {
                    if result.bounding_boxes_count > 0 {
                        eprintln!(
                            "[EdgeImpulse FFI] Warning: bounding_boxes pointer is null but count is {}",
                            result.bounding_boxes_count
                        );
                    }
                    return vec![];
                }
                let bbs = std::slice::from_raw_parts(
                    result.bounding_boxes,
                    result.bounding_boxes_count as usize,
                );
                bbs.iter()
                    .filter_map(|bb| {
                        if bb.value == 0.0 {
                            eprintln!("[EdgeImpulse FFI] Skipping bounding box with value 0.0");
                            return None;
                        }
                        let label = if !bb.label.is_null() {
                            std::ffi::CStr::from_ptr(bb.label)
                                .to_string_lossy()
                                .into_owned()
                        } else {
                            eprintln!(
                                "[EdgeImpulse FFI] Warning: bounding box label pointer is null"
                            );
                            String::new()
                        };
                        Some(BoundingBox {
                            label,
                            value: bb.value,
                            x: bb.x,
                            y: bb.y,
                            width: bb.width,
                            height: bb.height,
                        })
                    })
                    .collect()
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            vec![]
        }
    }

    /// Get visual anomaly detection results
    ///
    /// Returns visual anomaly detection results if available. The visual anomaly detection
    /// fields are now always present in the bindings, so this method will return Some
    /// if the model supports visual anomaly detection and has results.
    pub fn visual_anomaly(&self) -> Option<VisualAnomalyResult> {
        #[cfg(feature = "ffi")]
        {
            unsafe {
                let result = &*self.result;

                // Check if visual anomaly detection is available and has results
                if result.visual_ad_count == 0 {
                    return None;
                }

                // Get the visual anomaly result values
                let visual_ad_result = &result.visual_ad_result;
                let mean_value = visual_ad_result.mean_value;
                let max_value = visual_ad_result.max_value;

                // Get the grid cells
                let grid_cells = if result.visual_ad_grid_cells.is_null() {
                    eprintln!(
                        "[EdgeImpulse FFI] Warning: visual_ad_grid_cells pointer is null but count is {}",
                        result.visual_ad_count
                    );
                    vec![]
                } else {
                    let cells = std::slice::from_raw_parts(
                        result.visual_ad_grid_cells,
                        result.visual_ad_count as usize,
                    );
                    cells.iter()
                        .filter_map(|cell| {
                            if cell.value == 0.0 {
                                return None;
                            }
                            let label = if !cell.label.is_null() {
                                std::ffi::CStr::from_ptr(cell.label)
                                    .to_string_lossy()
                                    .into_owned()
                            } else {
                                eprintln!("[EdgeImpulse FFI] Warning: visual anomaly grid cell label pointer is null");
                                String::new()
                            };
                            Some(BoundingBox {
                                label,
                                value: cell.value,
                                x: cell.x,
                                y: cell.y,
                                width: cell.width,
                                height: cell.height,
                            })
                        })
                        .collect()
                };

                Some(VisualAnomalyResult {
                    mean_value,
                    max_value,
                    grid_cells,
                })
            }
        }
        #[cfg(not(feature = "ffi"))]
        {
            None
        }
    }

    /// Get timing information
    pub fn timing(&self) -> TimingResult {
        #[cfg(feature = "ffi")]
        {
            unsafe {
                let result = &*self.result;
                let t = &result.timing;
                TimingResult {
                    dsp: t.dsp,
                    classification: t.classification,
                    anomaly: t.anomaly,
                }
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            TimingResult {
                dsp: 0,
                classification: 0,
                anomaly: 0,
            }
        }
    }
}

impl Drop for InferenceResult {
    fn drop(&mut self) {
        #[cfg(feature = "ffi")]
        {
            if !self.result.is_null() {
                unsafe {
                    std::alloc::dealloc(
                        self.result as *mut u8,
                        std::alloc::Layout::new::<edge_impulse_ffi_rs::bindings::ei_impulse_result_t>(
                        ),
                    );
                }
            }
        }
    }
}

/// Main Edge Impulse classifier
pub struct EdgeImpulseClassifier {
    #[cfg(feature = "ffi")]
    initialized: bool,
}

impl Default for EdgeImpulseClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeImpulseClassifier {
    /// Create a new Edge Impulse classifier
    pub fn new() -> Self {
        #[cfg(feature = "ffi")]
        {
            Self { initialized: false }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Self {}
        }
    }

    /// Initialize the classifier
    pub fn init(&mut self) -> EdgeImpulseResult<()> {
        #[cfg(feature = "ffi")]
        {
            unsafe { edge_impulse_ffi_rs::bindings::ei_ffi_run_classifier_init() };
            self.initialized = true;
            Ok(())
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }

    /// Deinitialize the classifier
    pub fn deinit(&mut self) -> EdgeImpulseResult<()> {
        #[cfg(feature = "ffi")]
        {
            if self.initialized {
                unsafe { edge_impulse_ffi_rs::bindings::ei_ffi_run_classifier_deinit() };
                self.initialized = false;
            }
            Ok(())
        }

        #[cfg(not(feature = "ffi"))]
        {
            Ok(())
        }
    }

    /// Run classification on signal data
    pub fn run_classifier(
        &self,
        _signal: &Signal,
        _debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        #[cfg(feature = "ffi")]
        {
            if !self.initialized {
                return Err(EdgeImpulseError::Other);
            }
            let result = InferenceResult::new();
            let result_code = unsafe {
                edge_impulse_ffi_rs::bindings::ei_ffi_run_classifier(
                    _signal.as_ptr(),
                    result.result,
                    if _debug { 1 } else { 0 },
                )
            };
            if result_code == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
                Ok(result)
            } else {
                Err(EdgeImpulseError::from(result_code))
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }

    /// Run continuous classification on signal data
    pub fn run_classifier_continuous(
        &self,
        _signal: &Signal,
        _debug: bool,
        _enable_maf: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        #[cfg(feature = "ffi")]
        {
            if !self.initialized {
                return Err(EdgeImpulseError::Other);
            }
            let result = InferenceResult::new();
            let result_code = unsafe {
                edge_impulse_ffi_rs::bindings::ei_ffi_run_classifier_continuous(
                    _signal.as_ptr(),
                    result.result,
                    if _debug { 1 } else { 0 },
                    if _enable_maf { 1 } else { 0 },
                )
            };
            if result_code == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
                Ok(result)
            } else {
                Err(EdgeImpulseError::from(result_code))
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }

    /// Run inference on pre-processed features
    ///
    /// # Safety
    /// This function is unsafe because it takes a raw pointer `fmatrix` that may be dereferenced.
    /// The caller must ensure the pointer is valid and points to valid memory.
    pub unsafe fn run_inference(
        &self,
        _handle: &mut EdgeImpulseHandle,
        #[cfg(feature = "ffi")] fmatrix: *mut edge_impulse_ffi_rs::bindings::ei_feature_t,
        #[cfg(not(feature = "ffi"))] _fmatrix: *mut std::ffi::c_void,
        _debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        #[cfg(feature = "ffi")]
        {
            if !self.initialized {
                return Err(EdgeImpulseError::Other);
            }
            let result = InferenceResult::new();
            let result_code = unsafe {
                edge_impulse_ffi_rs::bindings::ei_ffi_run_inference(
                    _handle.handle,
                    fmatrix,
                    result.result,
                    if _debug { 1 } else { 0 },
                )
            };
            let error = EdgeImpulseError::from(result_code);
            if error == EdgeImpulseError::Ok {
                Ok(result)
            } else {
                Err(error)
            }
        }

        #[cfg(not(feature = "ffi"))]
        {
            Err(EdgeImpulseError::Other)
        }
    }
}

impl Drop for EdgeImpulseClassifier {
    fn drop(&mut self) {
        let _ = self.deinit();
    }
}

// Model metadata constants are available from edge_impulse_ffi_rs::model_metadata

/// Helper functions to access model metadata
pub struct ModelMetadata;

#[derive(Debug, Clone)]
pub struct ModelMetadataInfo {
    pub input_width: usize,
    pub input_height: usize,
    pub input_frames: usize,
    pub label_count: usize,
    pub project_name: &'static str,
    pub project_owner: &'static str,
    pub project_id: usize,
    pub deploy_version: usize,
    pub sensor: i32,
    pub inferencing_engine: usize,
    pub interval_ms: usize,
    pub frequency: usize,
    pub slice_size: usize,
    pub has_anomaly: bool,
    pub has_object_detection: bool,
    pub has_object_tracking: bool,
    pub raw_sample_count: usize,
    pub raw_samples_per_frame: usize,
    pub input_features_count: usize,
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub value: f32,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub label: String,
    pub value: f32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct TimingResult {
    pub dsp: i32,
    pub classification: i32,
    pub anomaly: i32,
}

/// Visual anomaly detection result
#[derive(Debug, Clone)]
pub struct VisualAnomalyResult {
    pub mean_value: f32,
    pub max_value: f32,
    pub grid_cells: Vec<BoundingBox>,
}

impl fmt::Display for VisualAnomalyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Visual Anomaly: mean={:.4}, max={:.4}, grid_cells={}",
            self.mean_value,
            self.max_value,
            self.grid_cells.len()
        )
    }
}

impl fmt::Display for ModelMetadataInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Metadata:")?;
        writeln!(
            f,
            "  Project: {} (ID: {})",
            self.project_name, self.project_id
        )?;
        writeln!(f, "  Owner: {}", self.project_owner)?;
        writeln!(f, "  Deploy version: {}", self.deploy_version)?;
        writeln!(
            f,
            "  Input: {}x{} frames: {}",
            self.input_width, self.input_height, self.input_frames
        )?;
        writeln!(f, "  Label count: {}", self.label_count)?;
        writeln!(f, "  Sensor: {}", self.sensor)?;
        writeln!(f, "  Inferencing engine: {}", self.inferencing_engine)?;
        writeln!(f, "  Interval (ms): {}", self.interval_ms)?;
        writeln!(f, "  Frequency: {}", self.frequency)?;
        writeln!(f, "  Slice size: {}", self.slice_size)?;
        writeln!(f, "  Has anomaly: {}", self.has_anomaly)?;
        writeln!(f, "  Has object detection: {}", self.has_object_detection)?;
        writeln!(f, "  Has object tracking: {}", self.has_object_tracking)?;
        writeln!(f, "  Raw sample count: {}", self.raw_sample_count)?;
        writeln!(f, "  Raw samples per frame: {}", self.raw_samples_per_frame)?;
        writeln!(f, "  Input features count: {}", self.input_features_count)?;
        Ok(())
    }
}

impl ModelMetadata {
    /// Get the model's required input width
    pub fn input_width() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_INPUT_WIDTH
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's required input height
    pub fn input_height() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_INPUT_HEIGHT
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's required input frame size (width * height)
    pub fn input_frame_size() -> usize {
        Self::input_width() * Self::input_height()
    }

    /// Get the number of input frames
    pub fn input_frames() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_INPUT_FRAMES
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the number of labels
    pub fn label_count() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_LABEL_COUNT
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the project name
    pub fn project_name() -> &'static str {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_PROJECT_NAME
        }
        #[cfg(not(feature = "ffi"))]
        {
            ""
        }
    }

    /// Get the project owner
    pub fn project_owner() -> &'static str {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_PROJECT_OWNER
        }
        #[cfg(not(feature = "ffi"))]
        {
            ""
        }
    }

    /// Get the project ID
    pub fn project_id() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_PROJECT_ID
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the deploy version
    pub fn deploy_version() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_PROJECT_DEPLOY_VERSION
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the sensor type
    pub fn sensor() -> i32 {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_SENSOR
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the inferencing engine
    pub fn inferencing_engine() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_INFERENCING_ENGINE
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's interval in ms
    pub fn interval_ms() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_INTERVAL_MS as usize
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's frequency
    pub fn frequency() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_FREQUENCY
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's slice size
    pub fn slice_size() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_SLICE_SIZE
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Whether the model has anomaly detection
    pub fn has_anomaly() -> bool {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_HAS_ANOMALY != 0
        }
        #[cfg(not(feature = "ffi"))]
        {
            false
        }
    }

    /// Whether the model has object detection
    pub fn has_object_detection() -> bool {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_OBJECT_DETECTION != 0
        }
        #[cfg(not(feature = "ffi"))]
        {
            false
        }
    }

    /// Whether the model has object tracking
    pub fn has_object_tracking() -> bool {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_OBJECT_TRACKING_ENABLED != 0
        }
        #[cfg(not(feature = "ffi"))]
        {
            false
        }
    }

    /// Get the model's raw sample count
    pub fn raw_sample_count() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_RAW_SAMPLE_COUNT
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's raw samples per frame
    pub fn raw_samples_per_frame() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    /// Get the model's input feature count
    pub fn input_features_count() -> usize {
        #[cfg(feature = "ffi")]
        {
            edge_impulse_ffi_rs::model_metadata::EI_CLASSIFIER_NN_INPUT_FRAME_SIZE
        }
        #[cfg(not(feature = "ffi"))]
        {
            0
        }
    }

    pub fn get() -> ModelMetadataInfo {
        ModelMetadataInfo {
            input_width: Self::input_width(),
            input_height: Self::input_height(),
            input_frames: Self::input_frames(),
            label_count: Self::label_count(),
            project_name: Self::project_name(),
            project_owner: Self::project_owner(),
            project_id: Self::project_id(),
            deploy_version: Self::deploy_version(),
            sensor: Self::sensor(),
            inferencing_engine: Self::inferencing_engine(),
            interval_ms: Self::interval_ms(),
            frequency: Self::frequency(),
            slice_size: Self::slice_size(),
            has_anomaly: Self::has_anomaly(),
            has_object_detection: Self::has_object_detection(),
            has_object_tracking: Self::has_object_tracking(),
            raw_sample_count: Self::raw_sample_count(),
            raw_samples_per_frame: Self::raw_samples_per_frame(),
            input_features_count: Self::input_features_count(),
        }
    }
}

/// Set the minimum confidence threshold for object detection
///
/// # Arguments
///
/// * `block_id` - The ID of the learning block that performs object detection
/// * `min_score` - The minimum confidence score (0.0 to 1.0) for detected objects
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the block was not found or is not an object detection block.
///
/// # Example
///
/// ```rust
/// use edge_impulse_runner::ffi::wrapper::set_object_detection_threshold;
///
/// // Set minimum confidence to 0.5 for object detection block
/// // Note: Use the actual block ID from your model metadata
/// if let Err(e) = set_object_detection_threshold(8, 0.5) {
///     eprintln!("Failed to set object detection threshold: {:?}", e);
/// }
/// ```
#[cfg(feature = "ffi")]
pub fn set_object_detection_threshold(block_id: u32, min_score: f32) -> EdgeImpulseResult<()> {
    let result = unsafe {
        edge_impulse_ffi_rs::bindings::ei_ffi_set_object_detection_threshold(block_id, min_score)
    };

    if result == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
        Ok(())
    } else {
        Err(result.into())
    }
}

#[cfg(not(feature = "ffi"))]
pub fn set_object_detection_threshold(_block_id: u32, _min_score: f32) -> EdgeImpulseResult<()> {
    Err(EdgeImpulseError::UnsupportedInferencingEngine)
}

/// Set the minimum anomaly score threshold for anomaly detection
///
/// # Arguments
///
/// * `block_id` - The ID of the learning block that performs anomaly detection
/// * `min_anomaly_score` - The minimum anomaly score threshold
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the block was not found or is not an anomaly detection block.
///
/// # Example
///
/// ```rust
/// use edge_impulse_runner::ffi::wrapper::set_anomaly_threshold;
///
/// // Set minimum anomaly score to 0.3 for anomaly detection block 1
/// if let Err(e) = set_anomaly_threshold(1, 0.3) {
///     eprintln!("Failed to set anomaly threshold: {:?}", e);
/// }
/// ```
#[cfg(feature = "ffi")]
pub fn set_anomaly_threshold(block_id: u32, min_anomaly_score: f32) -> EdgeImpulseResult<()> {
    let result = unsafe {
        edge_impulse_ffi_rs::bindings::ei_ffi_set_anomaly_threshold(block_id, min_anomaly_score)
    };

    if result == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
        Ok(())
    } else {
        Err(result.into())
    }
}

#[cfg(not(feature = "ffi"))]
pub fn set_anomaly_threshold(_block_id: u32, _min_anomaly_score: f32) -> EdgeImpulseResult<()> {
    Err(EdgeImpulseError::UnsupportedInferencingEngine)
}

/// Set the threshold parameters for object tracking
///
/// # Arguments
///
/// * `block_id` - The ID of the postprocessing block that performs object tracking
/// * `threshold` - The tracking threshold value
/// * `keep_grace` - The grace period for keeping tracks
/// * `max_observations` - The maximum number of observations to maintain
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the block was not found or is not an object tracking block.
///
/// # Example
///
/// ```rust
/// use edge_impulse_runner::ffi::wrapper::set_object_tracking_threshold;
///
/// // Set object tracking parameters for block 2
/// if let Err(e) = set_object_tracking_threshold(2, 0.7, 5, 10) {
///     eprintln!("Failed to set object tracking threshold: {:?}", e);
/// }
/// ```
#[cfg(feature = "ffi")]
pub fn set_object_tracking_threshold(
    block_id: u32,
    threshold: f32,
    keep_grace: u32,
    max_observations: u16,
) -> EdgeImpulseResult<()> {
    let result = unsafe {
        edge_impulse_ffi_rs::bindings::ei_ffi_set_object_tracking_threshold(
            block_id,
            threshold,
            keep_grace,
            max_observations,
        )
    };

    if result == edge_impulse_ffi_rs::bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
        Ok(())
    } else {
        Err(result.into())
    }
}

#[cfg(not(feature = "ffi"))]
pub fn set_object_tracking_threshold(
    _block_id: u32,
    _threshold: f32,
    _keep_grace: u32,
    _max_observations: u16,
) -> EdgeImpulseResult<()> {
    Err(EdgeImpulseError::UnsupportedInferencingEngine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_initialization() {
        let mut classifier = EdgeImpulseClassifier::new();
        // This will fail without the ffi feature, but that's expected
        let _ = classifier.init();
        let _ = classifier.deinit();
    }
}
