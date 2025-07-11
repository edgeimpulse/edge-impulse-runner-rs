use std::env;

fn main() {
    println!("cargo:info=Runner build script starting...");

    // Check if FFI feature is enabled by looking for the CARGO_FEATURE_FFI environment variable
    let ffi_enabled = env::var("CARGO_FEATURE_FFI").is_ok();

    println!("cargo:info=FFI feature enabled: {}", ffi_enabled);

    if ffi_enabled {
        println!("cargo:info=FFI feature enabled, checking required environment variables...");

        // Check for required environment variables
        let project_id = env::var("EI_PROJECT_ID");
        let api_key = env::var("EI_API_KEY");

        match (project_id, api_key) {
            (Ok(project_id), Ok(api_key)) => {
                println!("cargo:info=Found EI_PROJECT_ID: {}", project_id);
                println!("cargo:info=Found EI_API_KEY: {}...", &api_key[..api_key.len().min(8)]);
                println!("cargo:info=Environment variables are set correctly for FFI mode");
            }
            (Err(_), Err(_)) => {
                eprintln!("cargo:error=FFI mode requires EI_PROJECT_ID and EI_API_KEY environment variables to be set");
                eprintln!("cargo:error=Please set these variables before building:");
                eprintln!("cargo:error=  export EI_PROJECT_ID=your_project_id");
                eprintln!("cargo:error=  export EI_API_KEY=your_api_key");
                std::process::exit(1);
            }
            (Err(_), _) => {
                eprintln!("cargo:error=FFI mode requires EI_PROJECT_ID environment variable to be set");
                eprintln!("cargo:error=Please set this variable before building:");
                eprintln!("cargo:error=  export EI_PROJECT_ID=your_project_id");
                std::process::exit(1);
            }
            (_, Err(_)) => {
                eprintln!("cargo:error=FFI mode requires EI_API_KEY environment variable to be set");
                eprintln!("cargo:error=Please set this variable before building:");
                eprintln!("cargo:error=  export EI_API_KEY=your_api_key");
                std::process::exit(1);
            }
        }
    } else {
        println!("cargo:info=FFI feature not enabled, skipping environment variable checks");
    }
}
