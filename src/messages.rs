use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::*;
use crate::types::ModelParameters;

#[derive(Serialize, Debug)]
pub struct HelloMessage {
    pub hello: u32,
    pub id: u32,
}

#[derive(Serialize, Debug)]
pub struct ClassifyMessage {
    pub classify: Vec<f32>,
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub success: bool,
    pub id: u32,
    pub model_parameters: ModelParameters,
    pub project: ProjectInfo,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum InferenceResult {
    Classification {
        classification: HashMap<String, f32>,
    },
    ObjectDetection {
        bounding_boxes: Vec<BoundingBox>,
        #[serde(default)]
        classification: HashMap<String, f32>,
    },
}

#[derive(Deserialize, Debug)]
pub struct InferenceResponse {
    pub success: bool,
    pub id: u32,
    pub result: InferenceResult,
}

#[derive(Deserialize, Debug)]
pub struct ErrorResponse {
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub id: Option<u32>,
}

#[derive(Serialize, Debug)]
pub(crate) struct ConfigMessage {
    pub config: ConfigOptions,
    pub id: u32,
}

#[derive(Serialize, Debug)]
pub(crate) struct ConfigOptions {
    pub continuous_mode: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct ConfigResponse {
    pub success: bool,
    pub id: u32,
}

#[derive(Deserialize, Debug)]
pub struct BoundingBox {
    pub label: String,
    pub value: f32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}