//! Image Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform image classification
//! using a trained model on a single image file.
//!
//! Usage:
//!   cargo run --example image_infer -- --model <path_to_model> --image <path_to_image> [--debug]

use clap::Parser;
use edge_impulse_runner::types::ModelThreshold;
use edge_impulse_runner::{EimModel, InferenceResult};
use image::{self};
use std::error::Error;

/// Command line parameters for the image classification example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Edge Impulse model file
    #[arg(short, long)]
    model: String,

    /// Path to the image file to process
    #[arg(short, long)]
    image: String,

    /// Enable debug output
    #[arg(short, long, default_value_t = false)]
    debug: bool,
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

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Initialize the model
    let mut model = EimModel::new_with_debug(&args.model, args.debug)?;

    // Get model parameters
    let model_params = model.parameters()?;

    // Get the min_anomaly_score threshold
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

    // Print model parameters if debug is enabled
    if args.debug {
        println!("\nModel Parameters:");
        println!("----------------");
        println!("{:#?}", model_params);

        // Print threshold information
        println!("\nThresholds:");
        for threshold in &model_params.thresholds {
            match threshold {
                ModelThreshold::AnomalyGMM {
                    id,
                    min_anomaly_score,
                } => {
                    println!(
                        "  Anomaly GMM (ID: {}): min_anomaly_score = {}",
                        id, min_anomaly_score
                    );
                }
                ModelThreshold::ObjectDetection { id, min_score } => {
                    println!("  Object Detection (ID: {}): min_score = {}", id, min_score);
                }
                ModelThreshold::ObjectTracking {
                    id,
                    keep_grace,
                    max_observations,
                    threshold,
                } => {
                    println!(
                        "  Object Tracking (ID: {}): keep_grace = {}, max_observations = {}, threshold = {}",
                        id, keep_grace, max_observations, threshold
                    );
                }
                ModelThreshold::Unknown { id, unknown } => {
                    println!("  Unknown (ID: {}): unknown = {}", id, unknown);
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

    // Process the image
    let features = process_image(
        image_data,
        model_params.image_input_width,
        model_params.image_input_height,
        model_params.image_channel_count,
        args.debug,
    );

    // Run inference
    let result = model.infer(features, Some(args.debug))?.result;

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
