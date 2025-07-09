# Edge Impulse Runner Rust Workspace

[![Edge Impulse Tests](https://github.com/edgeimpulse/edge-impulse-runner-rs/actions/workflows/edge-impulse-runner.yml/badge.svg)](https://github.com/edgeimpulse/edge-impulse-runner-rs/actions/workflows/edge_impulse_runner.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://edgeimpulse.github.io/edge-impulse-runner-rs/)

> **⚠️ Note: This workspace requires the Rust nightly toolchain due to transitive dependencies.**
>
> To build and run, you must:
> 1. Install nightly: `rustup install nightly`
> 2. Set nightly for this project: `rustup override set nightly`

---

## Workspace Structure

This repository is a Cargo workspace containing:

- **runner/**: The main `edge-impulse-runner` Rust library for running Edge Impulse models (EIM and FFI modes).
- **ffi/**: The `edge-impulse-ffi-rs` crate, providing Rust FFI bindings to the Edge Impulse C++ SDK. Used when building the runner with the `ffi` feature.

---

## Features

### Dual Backend Support
- **EIM Mode** (default): Traditional socket-based communication with Edge Impulse binary files
- **FFI Mode**: Direct FFI calls to the Edge Impulse C++ SDK for improved performance

### Inference Capabilities
- Run Edge Impulse models on Linux and MacOS
- Support for different model types:
  - Classification models
  - Object detection models
  - Visual anomaly detection models
- Support for different sensor types:
  - Camera
  - Microphone
  - Accelerometer
  - Positional sensors
- Continuous classification mode support
- Debug output option

### Data Ingestion
- Upload data to Edge Impulse projects
- Support for multiple data categories:
  - Training data
  - Testing data
  - Anomaly data
- Handle various file formats:
  - Images (JPG, PNG)
  - Audio (WAV)
  - Video (MP4, AVI)
  - Sensor data (CBOR, JSON, CSV)

---

## Building

From the repository root, you can build either mode:

### EIM Mode (default, no FFI)
```sh
cargo build -p edge-impulse-runner
```

### FFI Mode (direct C++ SDK integration)
```sh
cargo build -p edge-impulse-runner --features ffi
```

This will automatically build the FFI crate and link it to the runner.

---

## Running Examples

This workspace contains examples in both the runner and FFI crates.

### Runner Crate Examples

The runner crate examples support both EIM and FFI modes:

```sh
# EIM mode examples (require model file)
cargo run -p edge-impulse-runner --example basic_infer -- --model /path/to/model.eim --features "0.1,0.2,0.3"
cargo run -p edge-impulse-runner --example image_infer -- --model /path/to/model.eim --image /path/to/image.png
cargo run -p edge-impulse-runner --example audio_infer -- --model /path/to/model.eim --audio /path/to/audio.wav
cargo run -p edge-impulse-runner --example video_infer -- --model /path/to/model.eim --video /path/to/video.mp4

# FFI mode examples (no model file needed)
cargo run -p edge-impulse-runner --example basic_infer -- --ffi --features "0.1,0.2,0.3"
cargo run -p edge-impulse-runner --example image_infer -- --ffi --image /path/to/image.png
cargo run -p edge-impulse-runner --example audio_infer -- --ffi --audio /path/to/audio.wav
cargo run -p edge-impulse-runner --example video_infer -- --ffi --video /path/to/video.mp4
```

**Note**: FFI mode examples require the `ffi` feature to be enabled:
```sh
cargo run -p edge-impulse-runner --features ffi --example basic_infer -- --ffi --features "0.1,0.2,0.3"
```

### FFI Crate Examples

The FFI crate contains standalone examples that work directly with the C++ SDK:

```sh
# Build and run FFI examples
cargo run -p edge-impulse-ffi-rs --example ffi_image_infer -- --image /path/to/image.png
```

**Important**: For FFI mode, you must first set up your model in the `ffi/model/` directory. See the [FFI README](ffi/README.md) for detailed instructions on building FFI bindings.

---

## Using the FFI Crate

To use FFI mode, you must:

1. **Set up your model**: Place your Edge Impulse exported C++ model (including `edge-impulse-sdk/`, `model-parameters/`, etc.) in the `ffi/model/` directory.

2. **Build the workspace with FFI feature**:
   ```sh
   cargo build -p edge-impulse-runner --features ffi
   ```

3. **Check the FFI README**: For advanced build flags, platform-specific instructions, and details on how the FFI build system works, see [`ffi/README.md`](ffi/README.md).

---

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
edge-impulse-runner = "2.0.0"
```

For FFI mode, add the `ffi` feature:
```toml
[dependencies]
edge-impulse-runner = { version = "2.0.0", features = ["ffi"] }
```

---

## Quick Start

### EIM Mode (Default)

```rust
use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new model instance using EIM binary
    let mut model = EdgeImpulseModel::new("path/to/model.eim")?;

    // Prepare normalized features (e.g., image pixels, audio samples)
    let features: Vec<f32> = vec![0.1, 0.2, 0.3];

    // Run inference
    let result = model.infer(features, None)?;

    // Process results
    match result.result {
        InferenceResult::Classification { classification } => {
            println!("Classification: {:?}", classification);
        }
        InferenceResult::ObjectDetection { bounding_boxes, classification } => {
            println!("Detected objects: {:?}", bounding_boxes);
            if !classification.is_empty() {
                println!("Classification: {:?}", classification);
            }
        }
        InferenceResult::VisualAnomaly { visual_anomaly_grid, visual_anomaly_max, visual_anomaly_mean, anomaly } => {
            let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) =
                model.normalize_visual_anomaly(
                    anomaly,
                    visual_anomaly_max,
                    visual_anomaly_mean,
                    &visual_anomaly_grid.iter()
                        .map(|bbox| (bbox.value, bbox.x as u32, bbox.y as u32, bbox.width as u32, bbox.height as u32))
                        .collect::<Vec<_>>()
                );
            println!("Anomaly score: {:.2}%", normalized_anomaly * 100.0);
        }
    }
    Ok(())
}
```

### FFI Mode

```rust
use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new model instance using FFI (requires "ffi" feature)
    let mut model = EdgeImpulseModel::new_ffi(false)?; // false = no debug

    // Prepare normalized features
    let features: Vec<f32> = vec![0.1, 0.2, 0.3];

    // Run inference
    let result = model.infer(features, None)?;

    // Process results (same as EIM mode)
    match result.result {
        InferenceResult::Classification { classification } => {
            println!("Classification: {:?}", classification);
        }
        // ... handle other result types
    }
    Ok(())
}
```

---

## Cargo Features

The crate supports different backends through Cargo features:

```toml
[dependencies]
edge-impulse-runner = { version = "2.0.0", features = ["ffi"] }
```

### Available Features

- **`eim`** (default): Enable EIM binary communication mode
- **`ffi`**: Enable FFI direct mode (requires `edge-impulse-ffi-rs` dependency)

### Feature Combinations

- **Default (`features = ["eim"]`)**: Only EIM mode available
- **`features = ["ffi"]`**: Only FFI mode available

**Note**: Only one backend should be enabled at a time. Enabling both features simultaneously is not supported and may cause conflicts.

---

## Performance Comparison

| Aspect | EIM Mode | FFI Mode |
|--------|----------|----------|
| **Startup Time** | Slower (process spawn + socket setup) | Faster (direct initialization) |
| **Inference Latency** | Higher (IPC overhead) | Lower (direct calls) |
| **Memory Usage** | Higher (separate process) | Lower (shared memory) |
| **Deployment** | Requires `.eim` files | Requires compiled model |
| **Flexibility** | Dynamic model loading | Static model compilation |

---

## Migration from 1.0.0 to 2.0.0

Version 2.0.0 introduces significant improvements and new features while maintaining backward compatibility for EIM mode.

### What's New in 2.0.0

#### 🚀 **FFI Mode Support**
- **Direct FFI calls**: New FFI backend for improved performance
- **No inter-process communication**: Eliminates socket overhead
- **Lower latency**: Direct calls to Edge Impulse C++ SDK
- **Reduced memory usage**: No separate model process

#### 🔄 **Unified API**
- **Consistent naming**: `EimModel` renamed to `EdgeImpulseModel` for clarity
- **Dual backend support**: Same API works with both EIM and FFI modes
- **Feature-based selection**: Choose backend via Cargo features

#### 🛠 **Enhanced Architecture**
- **Trait-based backends**: Clean abstraction for different inference engines
- **Improved error handling**: Better error types and messages
- **Consistent feature processing**: Unified approach for both modes

### Migration Guide

#### For Existing 1.0.0 Users

**Required Update (Breaking Change)**
```rust
// 1.0.0 code (no longer works)
use edge_impulse_runner::EimModel;  // ❌ EimModel no longer exists

let mut model = EimModel::new("model.eim")?;
let result = model.infer(features, None)?;
```

**Updated for 2.0.0**
```rust
// 2.0.0 code (required)
use edge_impulse_runner::EdgeImpulseModel;  // ✅ New unified API

let mut model = EdgeImpulseModel::new("model.eim")?;
let result = model.infer(features, None)?;
```

#### Adding FFI Support

To use the new FFI mode:

1. **Update Cargo.toml**:
   ```toml
   [dependencies]
   edge-impulse-runner = { version = "2.0.0", features = ["ffi"] }
   ```

2. **Use FFI mode**:
   ```rust
   use edge_impulse_runner::EdgeImpulseModel;

   // FFI mode (requires compiled model)
   let mut model = EdgeImpulseModel::new_ffi(false)?;
   let result = model.infer(features, None)?;
   ```

#### Breaking Changes

- **API Rename**: `EimModel` → `EdgeImpulseModel` (required update)
- **Import Changes**: Update `use` statements to use new type name
- **Same Functionality**: All methods and behavior remain identical
- **New features**: FFI mode requires the `ffi` Cargo feature

#### Performance Improvements

| Metric | 1.0.0 (EIM only) | 2.0.0 (FFI mode) | Improvement |
|--------|------------------|------------------|-------------|
| Startup time | ~100-200ms | ~10-20ms | **5-10x faster** |
| Inference latency | ~5-15ms | ~1-3ms | **3-5x faster** |
| Memory usage | ~50-100MB | ~20-40MB | **50% reduction** |
| Deployment | Requires .eim files | Compiled into binary | **Simpler** |

---

## Prerequisites

### EIM Mode
- Edge Impulse model files (`.eim`)
- Unix-like system (Linux, macOS)

### FFI Mode
- `edge-impulse-ffi-rs` crate dependency (available from [GitHub](https://github.com/edgeimpulse/edge-impulse-ffi-rs))
- Compiled model integrated into your binary
- For custom models, you can generate your own FFI bindings using the `edge-impulse-ffi-rs` repository as a template

Some functionality (particularly video capture) requires GStreamer to be installed:
- **macOS**: Install both runtime and development packages from gstreamer.freedesktop.org
- **Linux**: Install required packages (libgstreamer1.0-dev and related packages)

---

## Error Handling

The crate uses the `EimError` type to provide detailed error information:

```rust
use edge_impulse_runner::{EdgeImpulseModel, EimError};

match EdgeImpulseModel::new("model.eim") {
    Ok(mut model) => {
        match model.infer(vec![0.1, 0.2, 0.3], None) {
            Ok(result) => println!("Success!"),
            Err(EimError::InvalidInput(msg)) => println!("Invalid input: {}", msg),
            Err(e) => println!("Other error: {}", e),
        }
    },
    Err(e) => println!("Failed to load model: {}", e),
}
```

---

## Architecture

### Backend Abstraction

The library uses a trait-based backend abstraction that allows switching between different inference engines:

```rust
pub trait InferenceBackend: Send + Sync {
    fn new(config: BackendConfig) -> Result<Self, EimError> where Self: Sized;
    fn infer(&mut self, features: Vec<f32>, debug: Option<bool>) -> Result<InferenceResponse, EimError>;
    fn parameters(&self) -> Result<&ModelParameters, EimError>;
    // ... other methods
}
```

### EIM Backend
- Communicates with Edge Impulse binary files over Unix sockets
- Supports all features including continuous mode and threshold configuration
- Requires `.eim` files to be present

### FFI Backend
- Direct FFI calls to the Edge Impulse C++ SDK
- Improved performance with no inter-process communication overhead
- Requires the `edge-impulse-ffi-rs` crate as a dependency
- Model must be compiled into the binary

---

## Inference Communication Protocol (EIM Mode)

The Edge Impulse Inference Runner uses a Unix socket-based IPC mechanism to communicate with the model process in EIM mode. The protocol is JSON-based and follows a request-response pattern.

### Protocol Messages

#### 1. Initialization
When a new model is created, the following sequence occurs:

##### HelloMessage (Runner -> Model):
```json
{
    "hello": 1,
    "id": 1
}
```

##### ModelInfo Response (Model -> Runner):
```json
{
    "success": true,
    "id": 1,
    "model_parameters": {
        "axis_count": 1,
        "frequency": 16000.0,
        "has_anomaly": 0,
        "image_channel_count": 3,
        "image_input_frames": 1,
        "image_input_height": 96,
        "image_input_width": 96,
        "image_resize_mode": "fit-shortest",
        "inferencing_engine": 4,
        "input_features_count": 9216,
        "interval_ms": 1.0,
        "label_count": 1,
        "labels": ["class1"],
        "model_type": "classification",
        "sensor": 3,
        "slice_size": 2304,
        "threshold": 0.5,
        "use_continuous_mode": false
    },
    "project": {
        "deploy_version": 1,
        "id": 12345,
        "name": "Project Name",
        "owner": "Owner Name"
    }
}
```

#### 2. Inference
For each inference request:

##### ClassifyMessage (Runner -> Model):
```json
{
    "classify": [0.1, 0.2, 0.3],
    "id": 1,
    "debug": false
}
```

##### InferenceResponse (Model -> Runner):
For classification models:
```json
{
    "success": true,
    "id": 2,
    "result": {
        "classification": {
            "class1": 0.8,
            "class2": 0.2
        }
    }
}
```

For object detection models:
```json
{
    "success": true,
    "id": 2,
    "result": {
        "bounding_boxes": [
            {
                "label": "object1",
                "value": 0.95,
                "x": 100,
                "y": 150,
                "width": 50,
                "height": 50
            }
        ],
        "classification": {
            "class1": 0.8,
            "class2": 0.2
        }
    }
}
```

For visual anomaly detection models:
```json
{
    "success": true,
    "id": 2,
    "result": {
        "visual_anomaly": {
            "anomaly": 5.23,
            "visual_anomaly_max": 7.89,
            "visual_anomaly_mean": 4.12,
            "visual_anomaly_grid": [
                {
                    "value": 0.955,
                    "x": 24,
                    "y": 40,
                    "width": 8,
                    "height": 16
                }
            ]
        }
    }
}
```

#### 3. Error Response
When errors occur:

##### ErrorResponse (Model -> Runner):
```json
{
    "success": false,
    "error": "Error message",
    "id": 2
}
```

---

## Modules

- `error`: Error types and handling
- `inference`: Model management and inference functionality
- `backends`: Backend abstraction and implementations
- `ingestion`: Data upload and project management
- `types`: Common types and parameters

---

## License

BSD-3-Clause