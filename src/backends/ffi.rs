use super::{BackendConfig, InferenceBackend};
use crate::error::EdgeImpulseError;
use crate::ffi::{EdgeImpulseClassifier, ModelMetadata, Signal};
use crate::inference::messages::InferenceResponse;
use crate::types::{
    BoundingBox, ModelParameters, ModelThreshold, RunnerHelloHasAnomaly, SensorType,
    VisualAnomalyResult,
};
use std::sync::Arc;

/// FFI backend implementation using Edge Impulse FFI bindings
pub struct FfiBackend {
    /// Cached model parameters
    parameters: ModelParameters,
    /// Edge Impulse classifier instance
    classifier: EdgeImpulseClassifier,
    /// Debug callback
    debug_callback: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl FfiBackend {
    /// Create a new FFI backend
    pub fn new(config: BackendConfig) -> Result<Self, EdgeImpulseError> {
        let BackendConfig::Ffi { debug: _ } = config else {
            return Err(EdgeImpulseError::InvalidOperation(
                "Invalid config type for FFI backend".to_string(),
            ));
        };

        // Initialize the Edge Impulse classifier
        let mut classifier = EdgeImpulseClassifier::new();
        classifier.init().map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!(
                "Failed to initialize Edge Impulse classifier: {}",
                e
            ))
        })?;

        // Get model parameters from the compiled-in model metadata
        let parameters = Self::get_model_parameters()?;

        Ok(Self {
            parameters,
            classifier,
            debug_callback: None,
        })
    }

    /// Extract model parameters from the compiled-in model metadata
    fn get_model_parameters() -> Result<ModelParameters, EdgeImpulseError> {
        let metadata = ModelMetadata::get();

        // Convert sensor type (unused but kept for future use)
        let _sensor_type = match metadata.sensor {
            1 => SensorType::Microphone,
            2 => SensorType::Accelerometer,
            3 => SensorType::Camera,
            4 => SensorType::Positional,
            _ => SensorType::Unknown,
        };

        // Convert anomaly type
        let has_anomaly = if metadata.has_anomaly {
            RunnerHelloHasAnomaly::GMM // Default to GMM, could be enhanced to detect specific type
        } else {
            RunnerHelloHasAnomaly::None
        };

        // Get labels from the model metadata
        let labels = Self::get_model_labels();

        // Create thresholds based on model configuration
        let mut thresholds = Vec::new();
        if metadata.has_object_detection {
            thresholds.push(ModelThreshold::ObjectDetection {
                id: 0,
                min_score: 0.2, // Default threshold, could be made configurable
            });
        }

        Ok(ModelParameters {
            axis_count: 1, // Default for most models
            frequency: metadata.frequency as f32,
            has_anomaly,
            has_object_tracking: metadata.has_object_tracking,
            image_channel_count: if metadata.sensor == 3 { 3 } else { 0 }, // Camera = RGB
            image_input_frames: metadata.input_frames as u32,
            image_input_height: metadata.input_height as u32,
            image_input_width: metadata.input_width as u32,
            image_resize_mode: "fit".to_string(), // Default resize mode
            inferencing_engine: metadata.inferencing_engine as u32,
            input_features_count: (metadata.input_width * metadata.input_height) as u32,
            interval_ms: metadata.interval_ms as f32,
            label_count: metadata.label_count as u32,
            labels,
            model_type: if metadata.has_object_detection {
                "object-detection".to_string()
            } else {
                "classification".to_string()
            },
            sensor: metadata.sensor,
            slice_size: metadata.slice_size as u32,
            thresholds,
            use_continuous_mode: false, // Default to false
        })
    }

    /// Get model labels from the compiled-in model
    fn get_model_labels() -> Vec<String> {
        // For now, return a default label. In a real implementation,
        // this would read from the compiled model metadata
        vec!["object".to_string()]
    }

    /// Log debug message if callback is set
    fn debug_log(&self, message: &str) {
        if let Some(callback) = &self.debug_callback {
            callback(message);
        }
    }
}

impl InferenceBackend for FfiBackend {
    fn new(config: BackendConfig) -> Result<Self, EdgeImpulseError> {
        Self::new(config)
    }

    fn infer(
        &mut self,
        features: Vec<f32>,
        debug: Option<bool>,
    ) -> Result<InferenceResponse, EdgeImpulseError> {
        let debug_enabled = debug.unwrap_or(false);

        if debug_enabled {
            self.debug_log(&format!("Running inference on {} features", features.len()));
        }

        // Create a signal from the input features
        let signal = Signal::from_raw_data(&features).map_err(|e| {
            EdgeImpulseError::InvalidOperation(format!(
                "Failed to create signal from features: {}",
                e
            ))
        })?;

        // Run the classifier
        let result = self
            .classifier
            .run_classifier(&signal, debug_enabled)
            .map_err(|e| {
                EdgeImpulseError::InvalidOperation(format!("Failed to run classifier: {}", e))
            })?;

        // Extract results based on model type
        let inference_result = if self.parameters.model_type == "object-detection" {
            // Object detection model
            let bounding_boxes = result
                .bounding_boxes()
                .into_iter()
                .map(|bb| BoundingBox {
                    label: bb.label,
                    value: bb.value,
                    x: bb.x as i32,
                    y: bb.y as i32,
                    width: bb.width as i32,
                    height: bb.height as i32,
                })
                .collect();

            crate::inference::messages::InferenceResult::ObjectDetection {
                bounding_boxes,
                classification: std::collections::HashMap::new(), // Object detection doesn't have separate classification
            }
        } else {
            // Classification model
            let classifications = result
                .classifications(self.parameters.label_count as usize)
                .into_iter()
                .map(|c| (c.label, c.value))
                .collect();

            crate::inference::messages::InferenceResult::Classification {
                classification: classifications,
            }
        };

        // Note: Timing info is not available in the current InferenceResponse structure
        // The timing information is available from result.timing() but not included in the response

        Ok(InferenceResponse {
            success: true,
            id: 0, // This will be set by the caller
            result: inference_result,
        })
    }

    fn parameters(&self) -> Result<&ModelParameters, EdgeImpulseError> {
        Ok(&self.parameters)
    }

    fn sensor_type(&self) -> Result<SensorType, EdgeImpulseError> {
        Ok(match self.parameters.sensor {
            1 => SensorType::Microphone,
            2 => SensorType::Accelerometer,
            3 => SensorType::Camera,
            4 => SensorType::Positional,
            _ => SensorType::Unknown,
        })
    }

    fn input_size(&self) -> Result<usize, EdgeImpulseError> {
        Ok(self.parameters.input_features_count as usize)
    }

    fn set_debug_callback(&mut self, callback: Box<dyn Fn(&str) + Send + Sync>) {
        self.debug_callback = Some(Arc::from(callback));
    }

    fn normalize_visual_anomaly(
        &self,
        anomaly: f32,
        max: f32,
        mean: f32,
        regions: &[(f32, u32, u32, u32, u32)],
    ) -> VisualAnomalyResult {
        // Simple normalization - in a real implementation, this would use
        // the model's specific normalization logic
        let normalized_anomaly = (anomaly / max).min(1.0).max(0.0);
        let normalized_max = 1.0;
        let normalized_mean = (mean / max).min(1.0).max(0.0);

        let normalized_regions = regions
            .iter()
            .map(|&(value, x, y, w, h)| {
                let normalized_value = (value / max).min(1.0).max(0.0);
                (normalized_value, x, y, w, h)
            })
            .collect();

        (
            normalized_anomaly,
            normalized_max,
            normalized_mean,
            normalized_regions,
        )
    }
}
