use clap::Parser;
use edge_impulse_runner::EimModel;
use std::path::PathBuf;

/// Simple program to run Edge Impulse models
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .eim model file
    #[arg(short, long)]
    model: PathBuf,

    /// Enable debug output (show all messages sent/received)
    #[arg(short, long)]
    debug: bool,
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Convert to absolute path
    let model_path = if args.model.is_absolute() {
        args.model
    } else {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .join(&args.model)
    };

    println!("Loading model from: {}", model_path.display());

    // Try to create and initialize the model
    match EimModel::new_with_debug(&model_path, args.debug) {
        Ok(_model) => {
            println!("Successfully loaded model");
            // TODO: Add model interaction code here
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    }
}