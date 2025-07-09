use std::path::Path;

fn main() {
    // Only run this when the ffi feature is enabled
    if cfg!(feature = "ffi") {
        let lib_dir = Path::new("lib");

        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native=lib");
            println!("cargo:rustc-link-lib=static=edge_impulse_ffi_rs");

            // Re-run if lib directory changes
            println!("cargo:rerun-if-changed=lib");
        } else {
            // Fallback to using the crate dependency
            println!("cargo:warning=lib/ directory not found, using crate dependency for edge-impulse-ffi-rs");
        }
    }

    // Re-run if build script changes
    println!("cargo:rerun-if-changed=build.rs");
}