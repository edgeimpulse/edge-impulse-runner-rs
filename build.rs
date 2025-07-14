use std::env;
use std::path::Path;

fn main() {
    println!("cargo:info=Runner build script starting...");

    // Check if FFI feature is enabled by looking for the CARGO_FEATURE_FFI environment variable
    let ffi_enabled = env::var("CARGO_FEATURE_FFI").is_ok();

    println!("cargo:info=FFI feature enabled: {ffi_enabled}");

    if ffi_enabled {
        println!("cargo:info=FFI feature enabled, checking for model files...");

        // Check if model files exist in the edge-impulse-ffi-rs dependency
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let ffi_dep_path = Path::new(&manifest_dir).join("edge-impulse-ffi-rs");

        // Check for essential model components
        let sdk_dir = ffi_dep_path.join("model/edge-impulse-sdk");
        let model_parameters_dir = ffi_dep_path.join("model/model-parameters");
        let tflite_model_dir = ffi_dep_path.join("model/tflite-model");

        let has_valid_model =
            sdk_dir.exists() && model_parameters_dir.exists() && tflite_model_dir.exists();

        if has_valid_model {
            println!("cargo:info=Found valid model files in edge-impulse-ffi-rs dependency");
            println!("cargo:info=No environment variables required for FFI mode");
        } else {
            println!(
                "cargo:info=No valid model files found, checking for environment variables..."
            );

            // Check for EI_MODEL environment variable first
            if let Ok(model_path) = env::var("EI_MODEL") {
                println!("cargo:info=Found EI_MODEL environment variable: {model_path}");
                println!(
                    "cargo:info=The edge-impulse-ffi-rs dependency will handle copying from this path"
                );
            } else {
                println!(
                    "cargo:info=No EI_MODEL environment variable found, checking for API credentials..."
                );

                // Check for required environment variables
                let project_id = env::var("EI_PROJECT_ID");
                let api_key = env::var("EI_API_KEY");

                match (project_id, api_key) {
                    (Ok(project_id), Ok(api_key)) => {
                        println!("cargo:info=Found EI_PROJECT_ID: {project_id}");
                        println!(
                            "cargo:info=Found EI_API_KEY: {}...",
                            &api_key[..api_key.len().min(8)]
                        );
                        println!("cargo:info=Environment variables are set correctly for FFI mode");
                    }
                    (Err(_), Err(_)) => {
                        eprintln!(
                            "cargo:error=FFI mode requires environment variables to provide model files"
                        );
                        eprintln!("cargo:error=Please either:");
                        eprintln!("cargo:error=  1. Set EI_MODEL environment variable:");
                        eprintln!("cargo:error=     export EI_MODEL=/path/to/your/model");
                        eprintln!("cargo:error=  2. Set API environment variables:");
                        eprintln!("cargo:error=     export EI_PROJECT_ID=your_project_id");
                        eprintln!("cargo:error=     export EI_API_KEY=your_api_key");
                        std::process::exit(1);
                    }
                    (Err(_), _) => {
                        eprintln!(
                            "cargo:error=FFI mode requires environment variables to provide model files"
                        );
                        eprintln!("cargo:error=Please either:");
                        eprintln!("cargo:error=  1. Set EI_MODEL environment variable:");
                        eprintln!("cargo:error=     export EI_MODEL=/path/to/your/model");
                        eprintln!("cargo:error=  2. Set EI_PROJECT_ID environment variable:");
                        eprintln!("cargo:error=     export EI_PROJECT_ID=your_project_id");
                        std::process::exit(1);
                    }
                    (_, Err(_)) => {
                        eprintln!(
                            "cargo:error=FFI mode requires environment variables to provide model files"
                        );
                        eprintln!("cargo:error=Please either:");
                        eprintln!("cargo:error=  1. Set EI_MODEL environment variable:");
                        eprintln!("cargo:error=     export EI_MODEL=/path/to/your/model");
                        eprintln!("cargo:error=  2. Set EI_API_KEY environment variable:");
                        eprintln!("cargo:error=     export EI_API_KEY=your_api_key");
                        std::process::exit(1);
                    }
                }
            }
        }
    } else {
        println!("cargo:info=FFI feature not enabled, skipping environment variable checks");
    }
}
