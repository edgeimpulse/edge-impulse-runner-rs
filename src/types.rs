//! Common types and parameters used throughout the Edge Impulse Runner.
//!
//! This module contains the core data structures that define model configuration,
//! project information, and performance metrics. These types are used to configure
//! the model and interpret its outputs.

use serde::Deserialize;

/// Parameters that define a model's configuration and capabilities.
///
/// These parameters are received from the model during initialization and describe
/// the model's input requirements, processing settings, and output characteristics.
#[derive(Debug, Deserialize)]
pub struct ModelParameters {
    /// Number of axes for motion/positional data (e.g., 3 for xyz accelerometer)
    pub axis_count: u32,
    /// Sampling frequency in Hz for time-series data
    pub frequency: f32,
    /// Indicates if the model supports anomaly detection (0 = no, 1 = yes)
    pub has_anomaly: u32,
    /// Number of color channels in input images (1 = grayscale, 3 = RGB)
    pub image_channel_count: u32,
    /// Number of consecutive frames required for video input
    pub image_input_frames: u32,
    /// Required height of input images in pixels
    pub image_input_height: u32,
    /// Required width of input images in pixels
    pub image_input_width: u32,
    /// Method used to resize input images ("fit" or "fill")
    pub image_resize_mode: String,
    /// Type of inferencing engine (0 = TensorFlow Lite, 1 = TensorFlow.js)
    pub inferencing_engine: u32,
    /// Total number of input features expected by the model
    pub input_features_count: u32,
    /// Time interval between samples in milliseconds
    pub interval_ms: f32,
    /// Number of classification labels
    pub label_count: u32,
    /// Vector of classification labels
    pub labels: Vec<String>,
    /// Type of model ("classification", "object-detection", etc.)
    pub model_type: String,
    /// Type of input sensor (see SensorType enum)
    pub sensor: u32,
    /// Size of the processing window for time-series data
    pub slice_size: u32,
    /// Confidence threshold for detections (0.0 to 1.0)
    pub threshold: f32,
    /// Whether the model supports continuous mode operation
    pub use_continuous_mode: bool,
}

/// Information about the Edge Impulse project that created the model.
///
/// Contains metadata about the project's origin and version.
#[derive(Deserialize, Debug)]
pub struct ProjectInfo {
    /// Version number of the deployment
    pub deploy_version: u32,
    /// Unique project identifier
    pub id: u32,
    /// Name of the project
    pub name: String,
    /// Username of the project owner
    pub owner: String,
}

/// Performance timing information for different processing stages.
///
/// Provides detailed timing breakdowns for each step of the inference pipeline,
/// useful for performance monitoring and optimization.
#[derive(Deserialize, Debug)]
pub struct TimingInfo {
    /// Time spent on digital signal processing (DSP) in microseconds
    pub dsp: u32,
    /// Time spent on classification inference in microseconds
    pub classification: u32,
    /// Time spent on anomaly detection in microseconds
    pub anomaly: u32,
    /// Time spent on JSON serialization/deserialization in microseconds
    pub json: u32,
    /// Time spent reading from standard input in microseconds
    pub stdin: u32,
}

/// Represents a detected object's location and classification.
///
/// Used in object detection models to specify where objects were found
/// in an image and their classification details.
#[derive(Deserialize, Debug)]
pub struct BoundingBox {
    /// Height of the bounding box in pixels
    pub height: u32,
    /// Classification label for the detected object
    pub label: String,
    /// Confidence score for the detection (0.0 to 1.0)
    pub value: f32,
    /// Width of the bounding box in pixels
    pub width: u32,
    /// X-coordinate of the top-left corner
    pub x: u32,
    /// Y-coordinate of the top-left corner
    pub y: u32,
}

/// Represents the type of sensor used for data collection.
///
/// This enum defines the supported sensor types for Edge Impulse models,
/// allowing the system to properly handle different types of input data.
#[derive(Debug, Clone, Copy)]
pub enum SensorType {
    /// Camera sensor for image/video input
    Camera,
    /// Microphone sensor for audio input
    Microphone,
    /// Accelerometer sensor for motion data
    Accelerometer,
    /// Positional sensor for location/orientation data
    Positional,
    /// Other sensor types not covered by specific variants
    Other,
}
