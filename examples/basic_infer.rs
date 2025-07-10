//! Basic Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner for basic classification tasks.
//! It accepts raw feature data as a comma-separated string and runs inference using the specified model.
//!
//! The program will:
//! 1. Load an Edge Impulse model (EIM file or FFI mode)
//! 2. Parse input features from a string
//! 3. Run inference on the data
//! 4. Output classification results with probabilities
//!
//! Usage:
//!   # EIM mode (requires model file)
//!   cargo run --example basic_infer -- --model <path_to_model> --features "0.1,0.2,..." [--debug]
//!
//!   # FFI mode (no model file needed)
//!   cargo run --example basic_infer --features ffi -- --features "0.1,0.2,..." [--debug]

use clap::Parser;
#[cfg(feature = "ffi")]
use edge_impulse_runner::ffi::ModelMetadata;
use edge_impulse_runner::types::ModelThreshold;
use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .eim model file (not needed for FFI mode)
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Raw features string (comma-separated float values)
    #[arg(short, long, allow_hyphen_values = true)]
    features: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,

    /// Use FFI mode instead of EIM mode
    #[arg(long, default_value_t = false)]
    ffi: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse features from raw string
    let features: Vec<f32> = args
        .features
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("Failed to parse feature"))
        .collect();

    // Create model instance based on mode
    let mut model = if args.ffi {
        // FFI mode - no model file needed
        println!("Using FFI mode");
        let model = EdgeImpulseModel::new_ffi(args.debug)?;
        #[cfg(feature = "ffi")]
        {
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
        }
        model
    } else {
        // EIM mode - model file required
        let model_path = args.model.ok_or("Model path is required for EIM mode")?;

        // Adjust path to be relative to project root if not absolute
        let model_path = if !model_path.is_absolute() {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&model_path)
        } else {
            model_path
        };

        if args.debug {
            println!("Loading model from: {}", model_path.display());
        }
        EdgeImpulseModel::new(&model_path)?
    };

    // Run inference
    let result = model.infer(features, Some(args.debug))?.result;

    // Get model parameters
    let model_params = model.parameters()?;

    // Handle the result
    match result {
        InferenceResult::Classification { classification } => {
            println!("Classification: {:?}", classification);
        }
        InferenceResult::ObjectDetection {
            bounding_boxes,
            classification,
        } => {
            println!("Detected objects: {:?}", bounding_boxes);
            if !classification.is_empty() {
                println!("Classification: {:?}", classification);
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
            println!("  Overall: {:.2}", anomaly);
            println!("  Maximum: {:.2}", visual_anomaly_max);
            println!("  Mean: {:.2}", visual_anomaly_mean);

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
            let min_anomaly_score = model_params
                .thresholds
                .iter()
                .find_map(|t| match t {
                    ModelThreshold::AnomalyGMM {
                        min_anomaly_score, ..
                    } => Some(*min_anomaly_score),
                    _ => None,
                })
                .unwrap_or(6.0);
            println!("  min_anomaly_score: {}", min_anomaly_score);

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

            println!("\nNormalized scores:");
            println!("  Overall score: {:.2}%", normalized_anomaly * 100.0);
            println!("  Maximum score: {:.2}%", normalized_max * 100.0);
            println!("  Mean score: {:.2}%", normalized_mean * 100.0);

            // Show all detected regions (if any exist)
            if !normalized_regions.is_empty() {
                println!("\nDetected regions:");
                for (value, x, y, w, h) in normalized_regions {
                    println!(
                        "  Region (normalized: {:.2}%): x={}, y={}, width={}, height={}",
                        value * 100.0,
                        x,
                        y,
                        w,
                        h
                    );
                }
            } else {
                println!("\nNo regions detected");
                if args.debug {
                    println!("  Note: This could be because:");
                    println!("  1. The model didn't detect any anomalies above the threshold");
                    println!("  2. The visual_anomaly_grid is empty");
                    println!("  3. The normalization process filtered out all regions");
                    println!(
                        "  4. The min_anomaly_score threshold ({}) is too high",
                        min_anomaly_score
                    );
                }
            }
        }
    }

    Ok(())
}
