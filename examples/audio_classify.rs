//! Audio Classification Example
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform audio classification
//! using a trained model. It supports various WAV file formats and handles both mono and stereo inputs.
//!
//! The program will:
//! 1. Load an Edge Impulse model
//! 2. Read and preprocess a WAV file
//! 3. Run inference on the audio data
//! 4. Output the classification results
//!
//! Usage:
//!   cargo run --example audio_classify -- --model <path_to_model> --audio <path_to_wav> [--debug]

use clap::Parser;
use edge_impulse_runner::EimModel;
use hound;
use std::error::Error;
use std::path::PathBuf;

/// Command line parameters for the audio classification example
#[derive(Parser, Debug)]
struct AudioClassifyParams {
    /// Path to the Edge Impulse model file (.eim)
    #[clap(short, long)]
    model: String,

    /// Path to the input WAV file
    #[clap(short, long)]
    audio: String,

    /// Enable debug output
    #[clap(short, long)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let params = AudioClassifyParams::parse();
    let mut model = EimModel::new(&params.model)?;

    // Get model configuration
    let model_params = model.parameters()?;

    let audio_path = PathBuf::from(&params.audio);

    // Read audio file
    println!("Reading audio file: {}", audio_path.display());
    let mut reader = hound::WavReader::open(&audio_path)?;
    let spec = reader.spec();

    println!(
        "Model expects {} samples at {}Hz",
        model_params.input_features_count, model_params.frequency
    );

    // Convert samples to f32, regardless of input format
    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
            24 | 32 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / 2147483648.0)
                .collect(),
            8 => reader
                .samples::<i8>()
                .map(|s| s.unwrap() as f32 / 128.0)
                .collect(),
            _ => {
                return Err(format!("Unsupported bits per sample: {}", spec.bits_per_sample).into())
            }
        },
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    // If stereo, average channels to mono
    if spec.channels == 2 {
        let mono_samples: Vec<f32> = samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();
        samples = mono_samples;
    }

    // Verify sample rate matches model requirements
    let model_sample_rate = model_params.frequency as u32;
    if spec.sample_rate != model_sample_rate {
        return Err(format!(
            "Sample rate mismatch: file is {}Hz but model expects {}Hz. Resampling not yet implemented.",
            spec.sample_rate, model_sample_rate
        ).into());
    }

    // Ensure we have exactly the number of samples the model expects
    let required_samples = model_params.input_features_count as usize;
    println!("Using slice size of {} samples", required_samples);

    if samples.len() < required_samples {
        // Pad with zeros if too short
        samples.extend(vec![0.0; required_samples - samples.len()]);
    } else if samples.len() > required_samples {
        // Truncate if too long
        samples.truncate(required_samples);
    }

    // Run inference
    let result = model.classify(samples, Some(params.debug))?;
    println!("Classification result: {:?}", result);

    Ok(())
}
