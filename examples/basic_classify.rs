use clap::Parser;
use edge_impulse_runner::{EimModel, InferenceResult};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .eim model file (relative to current directory or absolute)
    #[arg(short, long)]
    model: PathBuf,

    /// Features array as comma-separated values (e.g., "0.1,0.2,0.3")
    #[arg(short, long)]
    features: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Convert relative path to absolute
    let model_path = if args.model.is_relative() {
        std::env::current_dir()?.join(args.model)
    } else {
        args.model
    };

    // Verify the file exists
    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()).into());
    }

    // Parse features string into vector
    let features: Vec<f32> = args.features
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<Result<Vec<f32>, _>>()?;

    // Create model instance
    let mut model = if args.debug {
        println!("Loading model from: {}", model_path.display());
        EimModel::new_with_debug(&model_path, true)?
    } else {
        EimModel::new(&model_path)?
    };

    // Run inference
    let result = model.classify(features, None)?;

    // Print results
    println!("Results:");
    match result.result {
        InferenceResult::Classification { classification } => {
            for (label, probability) in classification {
                println!("{}: {:.2}%", label, probability * 100.0);
            }
        }
        InferenceResult::ObjectDetection { bounding_boxes, .. } => {
            for bbox in bounding_boxes {
                println!("Found {} at ({}, {}) with confidence {:.2}%",
                    bbox.label, bbox.x, bbox.y, bbox.value * 100.0);
            }
        }
    }

    Ok(())
}