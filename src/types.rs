//! Common types and parameters used throughout the Edge Impulse Runner.
//!
//! This module contains the core data structures that define model configuration,
//! project information, and performance metrics. These types are used to configure
//! the model and interpret its outputs.

use serde::Deserialize;
use serde::Serialize;

/// Enum representing different types of anomaly detection supported by the model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerHelloHasAnomaly {
    None = 0,
    KMeans = 1,
    GMM = 2,
    VisualGMM = 3,
}

impl From<u32> for RunnerHelloHasAnomaly {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::KMeans,
            2 => Self::GMM,
            3 => Self::VisualGMM,
            _ => Self::None,
        }
    }
}

/// Parameters that define a model's configuration and capabilities.
///
/// These parameters are received from the model during initialization and describe
/// the model's input requirements, processing settings, and output characteristics.
#[derive(Debug, Deserialize, Clone)]
pub struct ModelParameters {
    /// Number of axes for motion/positional data (e.g., 3 for xyz accelerometer)
    pub axis_count: u32,
    /// Sampling frequency in Hz for time-series data
    pub frequency: f32,
    /// Indicates if the model supports anomaly detection
    #[serde(deserialize_with = "deserialize_anomaly_type")]
    pub has_anomaly: RunnerHelloHasAnomaly,
    /// Indicates if the model supports object tracking (0 = no, 1 = yes)
    #[serde(default)]
    pub has_object_tracking: bool,
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
    pub sensor: i32,
    /// Size of the processing window for time-series data
    pub slice_size: u32,
    /// Vector of thresholds for different types of detections
    #[serde(default)]
    pub thresholds: Vec<ModelThreshold>,
    /// Whether the model supports continuous mode operation
    pub use_continuous_mode: bool,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            axis_count: 0,
            frequency: 0.0,
            has_anomaly: RunnerHelloHasAnomaly::None,
            has_object_tracking: false,
            image_channel_count: 0,
            image_input_frames: 1,
            image_input_height: 0,
            image_input_width: 0,
            image_resize_mode: String::from("fit"),
            inferencing_engine: 0,
            input_features_count: 0,
            interval_ms: 0.0,
            label_count: 0,
            labels: Vec::new(),
            model_type: String::from("classification"),
            sensor: -1,
            slice_size: 0,
            thresholds: Vec::new(),
            use_continuous_mode: false,
        }
    }
}

fn deserialize_anomaly_type<'de, D>(deserializer: D) -> Result<RunnerHelloHasAnomaly, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    Ok(RunnerHelloHasAnomaly::from(value))
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ModelThreshold {
    #[serde(rename = "object_detection")]
    ObjectDetection { id: u32, min_score: f32 },
    #[serde(rename = "anomaly_gmm")]
    AnomalyGMM { id: u32, min_anomaly_score: f32 },
    #[serde(rename = "object_tracking")]
    ObjectTracking {
        id: u32,
        keep_grace: u32,
        max_observations: u32,
        threshold: f32,
    },
}

impl Default for ModelThreshold {
    fn default() -> Self {
        Self::ObjectDetection {
            id: 0,
            min_score: 0.5,
        }
    }
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
#[derive(Debug, Deserialize, Serialize)]
pub struct BoundingBox {
    /// Height of the bounding box in pixels
    pub height: i32,
    /// Classification label for the detected object
    pub label: String,
    /// Confidence score for the detection (0.0 to 1.0)
    pub value: f32,
    /// Width of the bounding box in pixels
    pub width: i32,
    /// X-coordinate of the top-left corner
    pub x: i32,
    /// Y-coordinate of the top-left corner
    pub y: i32,
}

/// Represents the normalized results of visual anomaly detection
pub type VisualAnomalyResult = (f32, f32, f32, Vec<(f32, u32, u32, u32, u32)>);

/// Represents the type of sensor used for data collection.
///
/// This enum defines the supported sensor types for Edge Impulse models,
/// mapping to the numeric values used in the protocol:
/// - -1 or unknown: Unknown
/// - 1: Microphone
/// - 2: Accelerometer
/// - 3: Camera
/// - 4: Positional
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorType {
    /// Unknown or unsupported sensor type (-1 or default)
    Unknown = -1,
    /// Microphone sensor for audio input (1)
    Microphone = 1,
    /// Accelerometer sensor for motion data (2)
    Accelerometer = 2,
    /// Camera sensor for image/video input (3)
    Camera = 3,
    /// Positional sensor for location/orientation data (4)
    Positional = 4,
}

impl From<i32> for SensorType {
    fn from(value: i32) -> Self {
        match value {
            -1 => SensorType::Unknown,
            1 => SensorType::Microphone,
            2 => SensorType::Accelerometer,
            3 => SensorType::Camera,
            4 => SensorType::Positional,
            _ => SensorType::Unknown,
        }
    }
}
