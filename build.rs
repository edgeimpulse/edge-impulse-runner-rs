use std::env;

fn main() {
    println!("cargo:info=Runner build script starting...");

    if cfg!(feature = "ffi") {
        println!("cargo:info=FFI feature enabled, looking for edge-impulse-ffi-rs...");

        // Debug: print all DEP_* environment variables
        for (key, value) in env::vars() {
            if key.starts_with("DEP_EDGE_IMPULSE_FFI_RS_") {
                println!("cargo:info=Found env var: {key} = {value}");
            }
        }

        if let Ok(lib_dir) = env::var("DEP_EDGE_IMPULSE_FFI_RS_ROOT") {
            println!("cargo:info=Found FFI root directory: {lib_dir}");
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rustc-link-lib=static=edge_impulse_ffi_rs");
        } else {
            println!("cargo:warning=Could not find edge-impulse-ffi-rs build output directory.");
            println!("cargo:warning=Available DEP_* vars:");
            for (key, value) in env::vars() {
                if key.starts_with("DEP_") {
                    println!("cargo:info=  {key} = {value}");
                }
            }
        }
    } else {
        println!("cargo:info=FFI feature not enabled");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
