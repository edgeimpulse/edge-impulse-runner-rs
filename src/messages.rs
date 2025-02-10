//! Message types for Edge Impulse model communication.
//!
//! This module defines the message structures used for communication between
//! the runner and the Edge Impulse model process via Unix sockets. All messages
//! are serialized to JSON for transmission.
//!
//! The communication follows a request-response pattern with the following types:
//! - Initialization messages (`HelloMessage`)
//! - Classification requests (`ClassifyMessage`)
//! - Model information responses (`ModelInfo`)
//! - Inference results (`InferenceResponse`)
//! - Configuration messages (`ConfigMessage`)
//! - Error responses (`ErrorResponse`)

use crate::types::ModelParameters;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Initial handshake message sent to the model process.
///
/// This message is sent when establishing communication with the model to
/// initialize the connection and receive model information.
#[derive(Serialize, Debug)]
pub struct HelloMessage {
    /// Protocol version number
    pub hello: u32,
    /// Unique message identifier
    pub id: u32,
}

/// Message containing features for classification.
///
/// Used to send preprocessed input features to the model for inference.
#[derive(Serialize, Debug)]
pub struct ClassifyMessage {
    /// Vector of preprocessed features matching the model's input requirements
    pub classify: Vec<f32>,
    /// Unique message identifier
    pub id: u32,
    /// Optional flag to enable debug output for this classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<bool>,
}

/// Response containing model information and parameters.
///
/// Received after sending a `HelloMessage`, contains essential information
/// about the model's configuration and capabilities.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// Indicates if the model initialization was successful
    pub success: bool,
    /// Message identifier matching the request
    #[allow(dead_code)]
    pub id: u32,
    /// Model parameters including input size, type, and other configuration
    pub model_parameters: ModelParameters,
    /// Project information from Edge Impulse
    #[allow(dead_code)]
    pub project: ProjectInfo,
}

/// Represents different types of inference results.
///
/// Models can produce different types of outputs depending on their type:
/// - Classification models return class probabilities
/// - Object detection models return bounding boxes and optional classifications
#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum InferenceResult {
    /// Result from a classification model
    Classification {
        /// Map of class names to their probability scores
        classification: HashMap<String, f32>,
    },
    /// Result from an object detection model
    ObjectDetection {
        /// Vector of detected objects with their bounding boxes
        bounding_boxes: Vec<BoundingBox>,
        /// Optional classification results for the entire image
        #[serde(default)]
        classification: HashMap<String, f32>,
    },
}

/// Response containing inference results.
///
/// Received after sending a `ClassifyMessage`, contains the model's
/// predictions and confidence scores.
#[derive(Deserialize, Debug)]
pub struct InferenceResponse {
    /// Indicates if the inference was successful
    pub success: bool,
    /// Message identifier matching the request
    pub id: u32,
    /// The actual inference results
    pub result: InferenceResult,
}

/// Response indicating an error condition.
///
/// Received when an error occurs during model communication or inference.
#[derive(Deserialize, Debug)]
pub struct ErrorResponse {
    /// Always false for error responses
    pub success: bool,
    /// Optional error message describing what went wrong
    #[serde(default)]
    pub error: Option<String>,
    /// Message identifier matching the request, if available
    #[allow(dead_code)]
    #[serde(default)]
    pub id: Option<u32>,
}

/// Message for configuring model runtime options.
///
/// Used to modify model behavior during runtime.
#[derive(Serialize, Debug)]
pub(crate) struct ConfigMessage {
    /// Configuration options to apply
    pub config: ConfigOptions,
    /// Unique message identifier
    pub id: u32,
}

/// Options for model configuration.
///
/// Contains various settings that can be applied to modify
/// model behavior.
#[derive(Serialize, Debug)]
pub(crate) struct ConfigOptions {
    /// Enable/disable continuous mode for streaming inference
    pub continuous_mode: Option<bool>,
}

/// Response to a configuration request.
///
/// Indicates whether the configuration was successfully applied.
#[derive(Deserialize, Debug)]
pub(crate) struct ConfigResponse {
    /// Indicates if the configuration was successful
    pub success: bool,
    /// Message identifier matching the request
    #[allow(dead_code)]
    pub id: u32,
}

/// Represents a bounding box for object detection results.
///
/// Contains the position, size, and classification information
/// for a detected object.
#[derive(Deserialize, Debug)]
pub struct BoundingBox {
    /// Class label for the detected object
    pub label: String,
    /// Confidence score for the detection (0.0 to 1.0)
    pub value: f32,
    /// X-coordinate of the top-left corner
    pub x: i32,
    /// Y-coordinate of the top-left corner
    pub y: i32,
    /// Width of the bounding box
    pub width: i32,
    /// Height of the bounding box
    pub height: i32,
}
