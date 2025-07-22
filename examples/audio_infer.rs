//! Audio Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform audio classification
//! using a trained model. It supports various WAV file formats and handles both mono and stereo inputs.
//!
//! The program will:
//! 1. Load an Edge Impulse model (EIM file or FFI mode)
//! 2. Read and preprocess a WAV file
//! 3. Run inference on the audio data
//! 4. Output the classification results
//!
//! Usage:
//!   # EIM mode (requires model file)
//!   cargo run --example audio_infer -- --model <path_to_model> --audio <path_to_wav> [--debug]
//!
//!   # FFI mode (no model file needed)
//!   cargo run --example audio_infer --features ffi -- --audio <path_to_wav> [--debug]

use clap::Parser;
use edge_impulse_runner::EdgeImpulseModel;
#[cfg(feature = "ffi")]
use edge_impulse_runner::ffi::ModelMetadata;
use std::path::PathBuf;

/// Command line parameters for the audio classification example
#[derive(Parser, Debug)]
struct AudioInferParams {
    /// Path to the Edge Impulse model file (.eim) (not needed for FFI mode)
    #[clap(short, long)]
    model: Option<String>,

    /// Path to the input WAV file
    #[clap(short, long)]
    audio: String,

    /// Enable debug output
    #[clap(short, long)]
    debug: bool,

    /// Use EIM mode (legacy, not recommended)
    #[clap(long)]
    eim: bool,

    /// Set object detection threshold (0.0 to 1.0)
    #[clap(long)]
    threshold: Option<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = AudioInferParams::parse();

    // Create model instance based on mode
    let mut model = if params.eim {
        // EIM mode - model file required
        #[cfg(feature = "eim")]
        {
            println!("Using EIM mode (legacy)");
            let model_path = params.model.ok_or("Model path is required for EIM mode")?;
            if params.debug {
                EdgeImpulseModel::new_eim_with_debug(&model_path, true)?
            } else {
                EdgeImpulseModel::new_eim(&model_path)?
            }
        }
        #[cfg(not(feature = "eim"))]
        {
            return Err("EIM mode requires the 'eim' feature to be enabled".into());
        }
    } else {
        // Auto-detect which backend to use based on available features
        #[cfg(feature = "ffi")]
        {
            // FFI mode - no model file needed (default)
            println!("Using FFI mode (default)");
            let model = if params.debug {
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
            let model_path = params.model.ok_or("Model path is required for EIM mode")?;
            if params.debug {
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

    // Apply threshold if provided
    if let Some(threshold) = params.threshold {
        println!("Setting object detection threshold to {}", threshold);

        // Get model parameters to find object detection thresholds
        let model_params = model.parameters()?;

        // Collect object detection thresholds to avoid borrowing issues
        let object_detection_thresholds: Vec<_> = model_params
            .thresholds
            .iter()
            .filter_map(|t| {
                if let edge_impulse_runner::types::ModelThreshold::ObjectDetection { id, .. } = t {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        // Apply the new threshold to each object detection block
        for block_id in object_detection_thresholds {
            let new_threshold = edge_impulse_runner::types::ModelThreshold::ObjectDetection {
                id: block_id,
                min_score: threshold,
            };

            match model.set_threshold(new_threshold) {
                Ok(()) => println!(
                    "Successfully set object detection threshold for block ID {} to {}",
                    block_id, threshold
                ),
                Err(e) => println!("Failed to set threshold for block ID {}: {}", block_id, e),
            }
        }
    }

    let audio_path = PathBuf::from(&params.audio);

    // Read audio file and convert to features
    let mut audio_file = hound::WavReader::open(audio_path)?;
    println!("Audio file specs: {:?}", audio_file.spec());
    let samples: Vec<f32> = audio_file
        .samples::<i16>()
        .map(|s| s.unwrap() as f32)
        .collect();

    println!("Read {} samples from audio file", samples.len());
    println!("Model expects {} samples", model.input_size()?);

    // Ensure we have enough samples
    if samples.len() < model.input_size()? {
        return Err(format!(
            "Audio file too short. Got {} samples, need {}",
            samples.len(),
            model.input_size()?
        )
        .into());
    }

    // Take exactly the number of samples we need
    let samples = samples[..model.input_size()?].to_vec();
    println!("Using {} samples for classification", samples.len());

    // Run inference
    let result = model.infer(samples, Some(params.debug))?;
    println!("Classification result: {result:?}");

    Ok(())
}
