//! Image Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform image classification
//! using a trained model on a single image file.
//!
//! Usage:
//!   cargo run --example image_infer -- --model <path_to_model> --image <path_to_image> [--debug]

use clap::Parser;
use edge_impulse_runner::{EimModel, InferenceResult};
use image::{self};
use serde_json::json;
use std::error::Error;

/// Command line parameters for the image classification example
#[derive(Parser, Debug)]
struct ImageInferParams {
    /// Path to the Edge Impulse model file (.eim)
    #[clap(short, long)]
    model: String,

    /// Path to the image file to infer
    #[clap(short, long)]
    image: String,

    /// Enable debug output
    #[clap(short, long)]
    debug: bool,
}

fn process_image(
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    _channels: u32,
    debug: bool,
) -> Vec<f32> {
    let mut features = Vec::with_capacity((width * height) as usize);

    // Process RGB channels (3 bytes at a time)
    for chunk in image_data.chunks(3) {
        if chunk.len() == 3 {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];
            // Convert to single feature value using
            // (r << 16) + (g << 8) + b
            let feature = ((r as u32) << 16) + ((g as u32) << 8) + (b as u32);
            features.push(feature as f32);
        }
    }

    if debug {
        println!(
            "Sending classification message with first 20 features: {:?}",
            &features[..20.min(features.len())]
        );
    }

    features
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let params = ImageInferParams::parse();

    // Initialize Edge Impulse model
    let mut model = EimModel::new_with_debug(&params.model, params.debug)?;

    // Get model parameters
    let model_params = model.parameters()?;

    // Print model parameters if debug is enabled
    if params.debug {
        println!("\nModel Parameters:");
        println!("----------------");
        println!("{:#?}", model_params);
        println!("----------------\n");
    }

    // Load and process the image
    let img = image::open(&params.image)?;
    let img = img.resize_exact(
        model_params.image_input_width,
        model_params.image_input_height,
        image::imageops::FilterType::Triangle,
    );

    // Convert to RGB bytes
    let rgb_img = img.to_rgb8();
    let image_data = rgb_img.into_raw();

    // Process the image
    let features = process_image(
        image_data,
        model_params.image_input_width,
        model_params.image_input_height,
        model_params.image_channel_count,
        params.debug,
    );

    if params.debug {
        println!(
            "Sending classification message with first 20 features: {:?}",
            &features[..20.min(features.len())]
        );
    }

    let features: Vec<f32> = features.into_iter().map(|x| x as f32 / 255.0).collect();

    let _classification_message = json!({
        "id": 2,
        "classify": {
            "features": features
        }
    });

    // Run inference
    match model.infer(features, Some(params.debug))?.result {
        InferenceResult::Classification { classification } => {
            println!("Classification results:");
            println!("----------------------");
            for (label, value) in classification {
                println!("{}: {:.2}%", label, value * 100.0);
            }
        }
        InferenceResult::ObjectDetection {
            bounding_boxes,
            classification,
        } => {
            if !bounding_boxes.is_empty() {
                println!("Detected objects:");
                println!("----------------");
                for bbox in bounding_boxes {
                    println!(
                        "- {} ({:.2}%): x={}, y={}, width={}, height={}",
                        bbox.label,
                        bbox.value * 100.0,
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height
                    );
                }
            }
            if !classification.is_empty() {
                println!("\nClassification results:");
                println!("----------------------");
                for (label, value) in classification {
                    println!("{}: {:.2}%", label, value * 100.0);
                }
            }
        }
    }

    Ok(())
}
