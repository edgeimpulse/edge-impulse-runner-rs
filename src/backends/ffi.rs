use super::{BackendConfig, InferenceBackend};
use crate::error::EdgeImpulseError;
use crate::ffi::{EdgeImpulseClassifier, ModelMetadata, Signal};
use crate::inference::messages::InferenceResponse;
use crate::types::{
    BoundingBox, ModelParameters, ModelThreshold, RunnerHelloHasAnomaly, SensorType,
    VisualAnomalyResult,
};
use std::sync::Arc;

/// Type alias for the debug callback to reduce complexity
type DebugCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// FFI backend implementation using Edge Impulse FFI bindings
pub struct FfiBackend {
    /// Cached model parameters
    parameters: ModelParameters,
    /// Edge Impulse classifier instance
    classifier: EdgeImpulseClassifier,
    /// Debug callback
    debug_callback: Option<DebugCallback>,
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
                "Failed to initialize Edge Impulse classifier: {e}"
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
            let ffi_thresholds = edge_impulse_ffi_rs::thresholds::get_model_thresholds();
            let object_detection_thresholds = ffi_thresholds.object_detection_thresholds();

            for threshold in object_detection_thresholds {
                thresholds.push(ModelThreshold::ObjectDetection {
                    id: threshold.id as u32,
                    min_score: threshold.min_score,
                });
            }

            if thresholds.is_empty() {
                thresholds.push(ModelThreshold::ObjectDetection {
                    id: 0,
                    min_score: 0.2,
                });
            }
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

    /// Update the cached model parameters with a new threshold
    fn update_cached_threshold(&mut self, new_threshold: crate::types::ModelThreshold) {
        // Find and update the existing threshold with the same ID, or add a new one
        let mut found = false;

        // First, try to find and update existing threshold
        for i in 0..self.parameters.thresholds.len() {
            let should_update = match (&self.parameters.thresholds[i], &new_threshold) {
                (
                    crate::types::ModelThreshold::ObjectDetection { id: existing_id, .. },
                    crate::types::ModelThreshold::ObjectDetection { id: new_id, .. },
                ) if *existing_id == *new_id => true,
                (
                    crate::types::ModelThreshold::AnomalyGMM { id: existing_id, .. },
                    crate::types::ModelThreshold::AnomalyGMM { id: new_id, .. },
                ) if *existing_id == *new_id => true,
                (
                    crate::types::ModelThreshold::ObjectTracking { id: existing_id, .. },
                    crate::types::ModelThreshold::ObjectTracking { id: new_id, .. },
                ) if *existing_id == *new_id => true,
                _ => false,
            };

            if should_update {
                self.parameters.thresholds[i] = new_threshold.clone();
                found = true;
                break;
            }
        }

        // If no existing threshold was found, add the new one
        if !found {
            self.parameters.thresholds.push(new_threshold);
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
                "Failed to create signal from features: {e}"
            ))
        })?;

        // Run the classifier
        let result = self
            .classifier
            .run_classifier(&signal, debug_enabled)
            .map_err(|e| {
                EdgeImpulseError::InvalidOperation(format!("Failed to run classifier: {e}"))
            })?;

        // Extract results based on model type
        let inference_result = if self.parameters.has_anomaly == RunnerHelloHasAnomaly::GMM {
            // Visual anomaly detection model
            if let Some(visual_anomaly_result) = result.visual_anomaly() {
                let visual_anomaly_grid = visual_anomaly_result
                    .grid_cells
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

                crate::inference::messages::InferenceResult::VisualAnomaly {
                    visual_anomaly_grid,
                    visual_anomaly_max: visual_anomaly_result.max_value,
                    visual_anomaly_mean: visual_anomaly_result.mean_value,
                    anomaly: 0.0, // Visual anomaly detection doesn't use the regular anomaly score
                }
            } else {
                // Fallback to classification if visual anomaly detection data is not available
                let classifications = result
                    .classifications(self.parameters.label_count as usize)
                    .into_iter()
                    .map(|c| (c.label, c.value))
                    .collect();

                crate::inference::messages::InferenceResult::Classification {
                    classification: classifications,
                }
            }
        } else if self.parameters.model_type == "object-detection" {
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
        // Output raw values without normalization, matching Edge Impulse Linux runner behavior
        // Normalization should be done in the UI/overlay layer for visualization
        let raw_anomaly = anomaly;
        let raw_max = max;
        let raw_mean = mean;

        // Keep raw region values
        let raw_regions = regions
            .iter()
            .map(|(value, x, y, w, h)| (*value, *x, *y, *w, *h))
            .collect();

        (raw_anomaly, raw_max, raw_mean, raw_regions)
    }

    fn set_threshold(&mut self, threshold: crate::types::ModelThreshold) -> Result<(), EdgeImpulseError> {
        // For FFI backend, we need to update the cached parameters
        // The actual threshold setting would need to be implemented in the FFI bindings
        // For now, we'll update the cached parameters and log a warning

        self.debug_log(&format!("Setting threshold: {:?}", threshold));

        // Update the cached model parameters
        self.update_cached_threshold(threshold.clone());

        // Use the new FFI threshold setting functions
        match threshold {
            crate::types::ModelThreshold::ObjectDetection { id, min_score } => {
                match crate::ffi::set_object_detection_threshold(id, min_score) {
                    Ok(()) => {
                        self.debug_log(&format!("Successfully set object detection threshold for block {} to {}", id, min_score));
                    }
                    Err(e) => {
                        self.debug_log(&format!("Failed to set object detection threshold: {:?}", e));
                        return Err(EdgeImpulseError::InvalidOperation(
                            format!("Failed to set object detection threshold: {:?}", e)
                        ));
                    }
                }
            }
            crate::types::ModelThreshold::AnomalyGMM { id, min_anomaly_score } => {
                match crate::ffi::set_anomaly_threshold(id, min_anomaly_score) {
                    Ok(()) => {
                        self.debug_log(&format!("Successfully set anomaly threshold for block {} to {}", id, min_anomaly_score));
                    }
                    Err(e) => {
                        self.debug_log(&format!("Failed to set anomaly threshold: {:?}", e));
                        return Err(EdgeImpulseError::InvalidOperation(
                            format!("Failed to set anomaly threshold: {:?}", e)
                        ));
                    }
                }
            }
            crate::types::ModelThreshold::ObjectTracking { id, keep_grace, max_observations, threshold } => {
                match crate::ffi::set_object_tracking_threshold(id, threshold, keep_grace, max_observations as u16) {
                    Ok(()) => {
                        self.debug_log(&format!("Successfully set object tracking threshold for block {} to {}", id, threshold));
                    }
                    Err(e) => {
                        self.debug_log(&format!("Failed to set object tracking threshold: {:?}", e));
                        return Err(EdgeImpulseError::InvalidOperation(
                            format!("Failed to set object tracking threshold: {:?}", e)
                        ));
                    }
                }
            }
            crate::types::ModelThreshold::Unknown { id, unknown } => {
                self.debug_log(&format!("Unknown threshold type for block {}: {}", id, unknown));
                return Err(EdgeImpulseError::InvalidOperation(
                    format!("Unknown threshold type for block {}: {}", id, unknown)
                ));
            }
        }

        Ok(())
    }
}
