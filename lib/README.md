# FFI Library Directory

This directory contains the compiled Edge Impulse FFI library for direct linking.

The library is generated from [edge-impulse-ffi-rs](https://github.com/edgeimpulse/edge-impulse-ffi-rs), which provides Rust FFI bindings for the Edge Impulse C++ SDK.

## Setup Instructions

To use FFI mode in `edge-impulse-runner-rs`, you need to:

1. **Clone and set up edge-impulse-ffi-rs**:
   ```bash
   git clone https://github.com/edgeimpulse/edge-impulse-ffi-rs
   cd edge-impulse-ffi-rs
   ```

2. **Add your Edge Impulse model**:
   - Replace the `model/` folder with your exported C++ model
   - For detailed instructions on building with different platforms, TensorFlow Lite versions, and hardware accelerators, see the [edge-impulse-ffi-rs documentation](https://github.com/edgeimpulse/edge-impulse-ffi-rs)

3. **Build the FFI library**:
   ```bash
   cargo build --release
   ```

4. **Copy the library to this directory**:
   ```bash
   cp target/release/libedge_impulse_ffi_rs.a ../edge-impulse-runner-rs/lib/
   ```

5. **Build edge-impulse-runner-rs with FFI support**:
   ```bash
   cd ../edge-impulse-runner-rs
   cargo build --features ffi
   ```

## File Types

The following library files are supported:
- **Static library**: `libedge_impulse_ffi_rs.a` (Linux/macOS)
- **Dynamic library**: `libedge_impulse_ffi_rs.so` (Linux)
- **Dynamic library**: `libedge_impulse_ffi_rs.dylib` (macOS)



## Notes

- This directory is tracked in git but library files are ignored
- The build script will automatically link against libraries in this directory when the `ffi` feature is enabled
- If no library is found, the build will fall back to using the crate dependency