use clap::Parser;
use edge_impulse_runner::{EimModel, InferenceResult};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .eim model file
    #[arg(short, long)]
    model: PathBuf,

    /// Raw features string
    #[arg(short, long, allow_hyphen_values = true)]
    features: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn parse_features(s: &str) -> Result<Vec<f32>, String> {
    s.trim_matches(|c| c == '\'' || c == '"')
        .split(',')
        .map(|s| s.trim().parse::<f32>()
            .map_err(|e| format!("Invalid feature value '{}': {}", s, e)))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse features from raw string
    let features = parse_features(&args.features)?;

    // Adjust path to be relative to project root
    let model_path = if !args.model.is_absolute() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&args.model)
    } else {
        args.model
    };

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