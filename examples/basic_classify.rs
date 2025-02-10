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
//!   cargo run --example basic_classify -- --model <path_to_model> --features "0.1,0.2,..." [--debug]

use clap::Parser;
use edge_impulse_runner::{EimModel, InferenceResult};
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

/// Parse a comma-separated string of features into a vector of f32 values
///
/// # Arguments
/// * `s` - Input string containing comma-separated float values
///
/// # Returns
/// * `Result<Vec<f32>, String>` - Vector of parsed float values or error message
///
/// # Example
/// ```
/// let features = parse_features("0.1,0.2,-0.3")?;
/// assert_eq!(features, vec![0.1, 0.2, -0.3]);
/// ```
fn parse_features(s: &str) -> Result<Vec<f32>, String> {
    s.trim_matches(|c| c == '\'' || c == '"')
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .map_err(|e| format!("Invalid feature value '{}': {}", s, e))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse features from raw string
    let features = parse_features(&args.features)?;

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
    let result = model.classify(features, Some(args.debug))?;

    // Print results based on the inference type
    println!("Results:");
    match result.result {
        InferenceResult::Classification { classification } => {
            // For classification models, print each class with its probability
            for (label, probability) in classification {
                println!("{}: {:.2}%", label, probability * 100.0);
            }
        }
        InferenceResult::ObjectDetection { bounding_boxes, .. } => {
            // For object detection models, print detected objects with locations
            for bbox in bounding_boxes {
                println!(
                    "Found {} at ({}, {}) with confidence {:.2}%",
                    bbox.label,
                    bbox.x,
                    bbox.y,
                    bbox.value * 100.0
                );
            }
        }
    }

    Ok(())
}
