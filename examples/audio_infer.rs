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
//!   cargo run --example audio_infer -- --model <path_to_model> --audio <path_to_wav> [--debug]

use clap::Parser;
use edge_impulse_runner::EimModel;
use hound;
use std::error::Error;
use std::path::PathBuf;

/// Command line parameters for the audio classification example
#[derive(Parser, Debug)]
struct AudioInferParams {
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
    let params = AudioInferParams::parse();
    let mut model = EimModel::new(&params.model)?;

    let audio_path = PathBuf::from(&params.audio);

    // Read audio file and convert to features
    let mut audio_file = hound::WavReader::open(audio_path)?;
    println!("Audio file specs: {:?}", audio_file.spec());
    let samples: Vec<f32> = audio_file
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
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
    println!("Classification result: {:?}", result);

    Ok(())
}
