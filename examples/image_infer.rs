//! Image Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform image classification
//! using a trained model on a single image file.
//!
//! Usage:
//!   # EIM mode (requires model file)
//!   cargo run --example image_infer -- --model <path_to_model> --image <path_to_image> [--debug]
//!
//!   # FFI mode (no model file needed)
//!   cargo run --example image_infer --features ffi -- --image <path_to_image> [--debug]

use clap::Parser;
use edge_impulse_runner::types::ModelThreshold;
use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
use image::{self};
use std::time::Instant;

#[cfg(feature = "ffi")]
use edge_impulse_runner::ffi::ModelMetadata;

/// Command line parameters for the image classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Edge Impulse model file (not needed for FFI mode)
    #[arg(short, long)]
    model: Option<String>,

    /// Path to the image file to process
    #[arg(short, long)]
    image: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,

    /// Set object detection threshold (0.0 to 1.0)
    #[arg(long)]
    threshold: Option<f32>,
}

fn process_image(
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
    debug: bool,
) -> Vec<f32> {
    let mut features = Vec::with_capacity((width * height) as usize);

    if channels == 1 {
        // For grayscale images, repeat the value across channels using bit shifting
        for &pixel in image_data.iter() {
            // Create a 24-bit value by repeating the grayscale value across all channels
            let feature = ((pixel as u32) << 16) | ((pixel as u32) << 8) | (pixel as u32);
            features.push(feature as f32);
        }
    } else {
        // For RGB images, combine channels
        for chunk in image_data.chunks(3) {
            if chunk.len() == 3 {
                // Combine RGB channels using bit shifting
                let feature =
                    ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
                features.push(feature as f32);
            }
        }
    }

    if debug {
        // Calculate some statistics about the features
        let min = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;

        println!(
            "Feature statistics: min={:.3}, max={:.3}, mean={:.3}, count={}",
            min,
            max,
            mean,
            features.len()
        );
    }

    features
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize the model - auto-detect which backend to use based on available features
    let mut model = {
        #[cfg(feature = "ffi")]
        {
            // FFI mode - no model file needed (default)
            println!("Using FFI mode (default)");
            let model = if args.debug {
                EdgeImpulseModel::new_with_debug(true)?
            } else {
                EdgeImpulseModel::new()?
            };

            // Print model metadata for FFI mode
            let metadata = ModelMetadata::get();
            println!("\nModel Metadata:");
            println!("===============");
            println!("Project ID: {}", metadata.project_id);
            println!("Project Owner: {}", metadata.project_owner);
            println!("Project Name: {}", metadata.project_name);
            println!("Deploy Version: {}", metadata.deploy_version);
            println!(
                "Model Type: {}",
                if metadata.has_object_detection {
                    "Object Detection"
                } else {
                    "Classification"
                }
            );
            println!(
                "Input Dimensions: {}x{}x{}",
                metadata.input_width, metadata.input_height, metadata.input_frames
            );
            println!("Input Features: {}", metadata.input_features_count);
            println!("Label Count: {}", metadata.label_count);
            println!(
                "Sensor Type: {}",
                match metadata.sensor {
                    1 => "Microphone",
                    2 => "Accelerometer",
                    3 => "Camera",
                    4 => "Positional",
                    _ => "Unknown",
                }
            );
            println!(
                "Inferencing Engine: {}",
                match metadata.inferencing_engine {
                    1 => "uTensor",
                    2 => "TensorFlow Lite",
                    3 => "CubeAI",
                    4 => "TensorFlow Lite Full",
                    _ => "Other",
                }
            );
            println!("Has Anomaly Detection: {}", metadata.has_anomaly);
            println!("Has Object Tracking: {}", metadata.has_object_tracking);
            println!("===============\n");

            model
        }
        #[cfg(all(feature = "eim", not(feature = "ffi")))]
        {
            // Only EIM feature available, use EIM mode
            println!("Using EIM mode (FFI not available)");
            let model_path = args.model.ok_or("Model path is required for EIM mode")?;
            if args.debug {
                EdgeImpulseModel::new_eim_with_debug(&model_path, true)?
            } else {
                EdgeImpulseModel::new_eim(&model_path)?
            }
        }
        #[cfg(not(any(feature = "ffi", feature = "eim")))]
        {
            return Err("No backend available. Enable either 'ffi' or 'eim' feature.".into());
        }
    };

    // Get model parameters
    let model_params = model.parameters()?.clone();

    // Apply threshold if provided
    if let Some(threshold) = args.threshold {
        println!("Setting object detection threshold to {}", threshold);

        // Find object detection thresholds and apply the new threshold
        for threshold_config in &model_params.thresholds {
            if let ModelThreshold::ObjectDetection { id, .. } = threshold_config {
                let new_threshold = ModelThreshold::ObjectDetection {
                    id: *id,
                    min_score: threshold,
                };

                match model.set_threshold(new_threshold) {
                    Ok(()) => println!(
                        "Successfully set object detection threshold for block ID {} to {}",
                        id, threshold
                    ),
                    Err(e) => println!("Failed to set threshold for block ID {}: {}", id, e),
                }
            }
        }
    }

    // Get the min_anomaly_score threshold
    let min_anomaly_score = model_params
        .thresholds
        .iter()
        .find_map(|t| match t {
            ModelThreshold::AnomalyGMM {
                min_anomaly_score, ..
            } => Some(min_anomaly_score),
            _ => None,
        })
        .unwrap_or(&6.0);

    // Print model parameters if debug is enabled
    if args.debug {
        println!("\nModel Parameters:");
        println!("----------------");
        println!("{model_params:#?}");

        // Print threshold information
        println!("\nThresholds:");
        for threshold in &model_params.thresholds {
            match threshold {
                ModelThreshold::AnomalyGMM {
                    id,
                    min_anomaly_score,
                } => {
                    println!("  Anomaly GMM (ID: {id}): min_anomaly_score = {min_anomaly_score}");
                }
                ModelThreshold::ObjectDetection { id, min_score } => {
                    println!("  Object Detection (ID: {id}): min_score = {min_score}");
                }
                ModelThreshold::ObjectTracking {
                    id,
                    keep_grace,
                    max_observations,
                    threshold,
                } => {
                    println!(
                        "  Object Tracking (ID: {id}): keep_grace = {keep_grace}, max_observations = {max_observations}, threshold = {threshold}"
                    );
                }
                ModelThreshold::Unknown { id, unknown } => {
                    println!("  Unknown (ID: {id}): unknown = {unknown}");
                }
            }
        }
        println!("----------------\n");
    }

    // Load and process the image
    let img = image::open(&args.image)?;
    let img = img.resize_exact(
        model_params.image_input_width,
        model_params.image_input_height,
        image::imageops::FilterType::Triangle,
    );

    // Convert to RGB bytes if needed
    let image_data = if model_params.image_channel_count == 1 {
        img.to_luma8().into_raw()
    } else {
        img.to_rgb8().into_raw()
    };

    // Process the image with timing
    let processing_start = Instant::now();
    let features = process_image(
        image_data,
        model_params.image_input_width,
        model_params.image_input_height,
        model_params.image_channel_count,
        args.debug,
    );
    let processing_duration = processing_start.elapsed();

    if args.debug {
        println!(
            "Image processing took: {:.2}ms",
            processing_duration.as_millis()
        );
    }

    // Run inference with timing
    let inference_start = Instant::now();
    let inference_result = model.infer(features, Some(args.debug))?;
    let inference_duration = inference_start.elapsed();

    if args.debug {
        println!("Inference took: {:.2}ms", inference_duration.as_millis());
    }

    let result = inference_result.result;

    // Handle the result
    match result {
        InferenceResult::Classification { classification } => {
            println!("Classification: {classification:?}");
        }
        InferenceResult::ObjectDetection {
            bounding_boxes,
            classification,
        } => {
            println!("Detected objects: {bounding_boxes:?}");
            if !classification.is_empty() {
                println!("Classification: {classification:?}");
            }
        }
        InferenceResult::VisualAnomaly {
            visual_anomaly_grid,
            visual_anomaly_max,
            visual_anomaly_mean,
            anomaly,
        } => {
            // Print raw values for debugging
            println!("\nRaw anomaly values:");
            println!("  Overall: {anomaly:.2}");
            println!("  Maximum: {visual_anomaly_max:.2}");
            println!("  Mean: {visual_anomaly_mean:.2}");

            // Debug output for the grid
            if args.debug {
                println!("\nVisual anomaly grid:");
                println!("  Number of regions: {}", visual_anomaly_grid.len());
                for (i, bbox) in visual_anomaly_grid.iter().enumerate() {
                    println!(
                        "  Region {}: value={:.2}, x={}, y={}, w={}, h={}",
                        i, bbox.value, bbox.x, bbox.y, bbox.width, bbox.height
                    );
                }
            }

            println!("\nThreshold information:");
            println!("  min_anomaly_score: {min_anomaly_score}");

            // Normalize all scores using the model's normalization method
            let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) = model
                .normalize_visual_anomaly(
                    anomaly,
                    visual_anomaly_max,
                    visual_anomaly_mean,
                    &visual_anomaly_grid
                        .iter()
                        .map(|bbox| {
                            (
                                bbox.value,
                                bbox.x as u32,
                                bbox.y as u32,
                                bbox.width as u32,
                                bbox.height as u32,
                            )
                        })
                        .collect::<Vec<_>>(),
                );

            println!("\nRaw scores:");
            println!("  Overall score: {:.2}", normalized_anomaly);
            println!("  Maximum score: {:.2}", normalized_max);
            println!("  Mean score: {:.2}", normalized_mean);

            // Show all detected regions (if any exist)
            if !normalized_regions.is_empty() {
                println!("\nDetected regions:");
                for (value, x, y, w, h) in normalized_regions {
                    println!(
                        "  Region (raw: {:.2}): x={}, y={}, width={}, height={}",
                        value, x, y, w, h
                    );
                }
            } else {
                println!("\nNo regions detected");
                if args.debug {
                    println!("  Note: This could be because:");
                    println!("  1. The model didn't detect any anomalies above the threshold");
                    println!("  2. The visual_anomaly_grid is empty");
                    println!("  3. All region values are zero or below threshold");
                    println!(
                        "  4. The min_anomaly_score threshold ({min_anomaly_score}) is too high"
                    );
                }
            }
        }
    }

    Ok(())
}
