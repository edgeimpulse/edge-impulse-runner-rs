use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::*;

#[derive(Serialize)]
pub struct HelloMessage {
    pub hello: u32,
    pub id: u32,
}

#[derive(Serialize)]
pub struct ClassifyMessage {
    pub classify: Vec<f32>,
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub model_parameters: ModelParameters,
    pub project: ProjectInfo,
    pub success: bool,
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
        visual_anomaly_grid: Option<Vec<BoundingBox>>,
        #[serde(default)]
        visual_anomaly_max: Option<f32>,
        #[serde(default)]
        visual_anomaly_mean: Option<f32>,
        #[serde(default)]
        anomaly: Option<f32>,
    },
}

#[derive(Deserialize, Debug)]
pub struct InferenceResponse {
    pub success: bool,
    pub result: InferenceResult,
    pub timing: TimingInfo,
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