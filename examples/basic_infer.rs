//! Basic Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner for basic classification tasks.
//! It accepts raw feature data as a comma-separated string and runs inference using the specified model.
//!
//! The program will:
//! 1. Load an Edge Impulse model
//! 2. Parse input features from a string
//! 3. Run inference on the data
//! 4. Output classification results with probabilities
//!
//! Usage:
//!   cargo run --example basic_infer -- --model <path_to_model> --features "0.1,0.2,..." [--debug]

use clap::Parser;
use edge_impulse_runner::types::ModelThreshold;
use edge_impulse_runner::{EimModel, InferenceResult};
use std::error::Error;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .eim model file
    #[arg(short, long)]
    model: PathBuf,

    /// Raw features string (comma-separated float values)
    #[arg(short, long, allow_hyphen_values = true)]
    features: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse features from raw string
    let features: Vec<f32> = args
        .features
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("Failed to parse feature"))
        .collect();

    // Adjust path to be relative to project root if not absolute
    let model_path = if !args.model.is_absolute() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&args.model)
    } else {
        args.model
    };

    // Create model instance with optional debug output
    let mut model = if args.debug {
        println!("Loading model from: {}", model_path.display());
        EimModel::new(&model_path)?
    } else {
        EimModel::new(&model_path)?
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
