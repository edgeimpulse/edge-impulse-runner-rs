use crate::ffi::bindings::EI_IMPULSE_ERROR;
use std::error::Error;
use std::fmt;

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

impl From<EI_IMPULSE_ERROR> for EdgeImpulseError {
    fn from(error: EI_IMPULSE_ERROR) -> Self {
        match error {
            EI_IMPULSE_ERROR::EI_IMPULSE_OK => EdgeImpulseError::Ok,
            EI_IMPULSE_ERROR::EI_IMPULSE_ERROR_SHAPES_DONT_MATCH => {
                EdgeImpulseError::ShapesDontMatch
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_CANCELED => EdgeImpulseError::Canceled,
            EI_IMPULSE_ERROR::EI_IMPULSE_ALLOC_FAILED => EdgeImpulseError::MemoryAllocationFailed,
            EI_IMPULSE_ERROR::EI_IMPULSE_OUT_OF_MEMORY => EdgeImpulseError::OutOfMemory,
            EI_IMPULSE_ERROR::EI_IMPULSE_INPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::InputTensorWasNull
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL => {
                EdgeImpulseError::OutputTensorWasNull
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ERROR => EdgeImpulseError::TfliteError,
            EI_IMPULSE_ERROR::EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED => {
                EdgeImpulseError::TfliteArenaAllocFailed
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_DSP_ERROR => EdgeImpulseError::ReadSensor,
            EI_IMPULSE_ERROR::EI_IMPULSE_INVALID_SIZE => EdgeImpulseError::MinSizeRatio,
            EI_IMPULSE_ERROR::EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES => {
                EdgeImpulseError::OnlySupportImages
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE => {
                EdgeImpulseError::UnsupportedInferencingEngine
            }
            EI_IMPULSE_ERROR::EI_IMPULSE_INFERENCE_ERROR => EdgeImpulseError::NoValidImpulse,
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