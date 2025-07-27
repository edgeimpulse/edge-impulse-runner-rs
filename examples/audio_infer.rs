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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = AudioInferParams::parse();

    // Create model instance - auto-detect which backend to use based on available features
    let mut model = {
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

    let audio_path = PathBuf::from(&params.audio);

    // Read audio file and convert to features
    let mut audio_file = hound::WavReader::open(audio_path)?;
    println!("Audio file specs: {:?}", audio_file.spec());
    let samples: Vec<f32> = audio_file
        .samples::<i16>()
        .map(|s| s.unwrap() as f32)
        .collect();

    println!("Read {} samples from audio file", samples.len());

    // For audio models, we need to get the raw sample count from metadata
    #[cfg(feature = "ffi")]
    let required_samples = {
        let metadata = ModelMetadata::get();
        let count = metadata.raw_sample_count;
        println!("Model expects {} raw audio samples", count);
        count
    };

    #[cfg(not(feature = "ffi"))]
    let required_samples = {
        let count = model.input_size()?;
        println!("Model expects {} samples", count);
        count
    };

    // Ensure we have enough samples
    if samples.len() < required_samples {
        return Err(format!(
            "Audio file too short. Got {} samples, need {}",
            samples.len(),
            required_samples
        )
        .into());
    }

    // Take exactly the number of raw samples we need
    let samples = samples[..required_samples].to_vec();
    println!(
        "Using {} raw audio samples for classification",
        samples.len()
    );

    // Run inference
    let result = model.infer(samples, Some(params.debug))?;
    println!("Classification result: {result:?}");

    Ok(())
}
