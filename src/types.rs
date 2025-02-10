use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ModelParameters {
    pub axis_count: u32,
    pub frequency: f32,
    pub has_anomaly: u32,
    pub image_channel_count: u32,
    pub image_input_frames: u32,
    pub image_input_height: u32,
    pub image_input_width: u32,
    pub image_resize_mode: String,
    pub inferencing_engine: u32,
    pub input_features_count: u32,
    pub interval_ms: u32,
    pub label_count: u32,
    pub labels: Vec<String>,
    pub model_type: String,
    pub sensor: u32,
    pub slice_size: u32,
    pub use_continuous_mode: bool,
    pub threshold: f32,
}

#[derive(Deserialize, Debug)]
pub struct ProjectInfo {
    pub deploy_version: u32,
    pub id: u32,
    pub name: String,
    pub owner: String,
}

#[derive(Deserialize, Debug)]
pub struct TimingInfo {
    pub dsp: u32,
    pub classification: u32,
    pub anomaly: u32,
    pub json: u32,
    pub stdin: u32,
}

#[derive(Deserialize, Debug)]
pub struct BoundingBox {
    pub height: u32,
    pub label: String,
    pub value: f32,
    pub width: u32,
    pub x: u32,
    pub y: u32,
}