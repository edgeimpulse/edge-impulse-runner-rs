# Edge Impulse Runner for Rust

A Rust library for running Edge Impulse Linux models with support for both EIM binary communication and direct FFI calls.

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

## Cargo Features

The crate supports different backends through Cargo features:

```toml
[dependencies]
edge-impulse-runner = { version = "1.0", features = ["ffi"] }
```

### Available Features

- **`eim`** (default): Enable EIM binary communication mode
- **`ffi`**: Enable FFI direct mode (requires `edge-impulse-ffi-rs` dependency)

### Feature Combinations

- **Default**: Only EIM mode available
- **`features = ["ffi"]`**: Only FFI mode available
- **`features = ["eim", "ffi"]`**: Both modes available

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

## Performance Comparison

| Aspect | EIM Mode | FFI Mode |
|--------|----------|----------|
| **Startup Time** | Slower (process spawn + socket setup) | Faster (direct initialization) |
| **Inference Latency** | Higher (IPC overhead) | Lower (direct calls) |
| **Memory Usage** | Higher (separate process) | Lower (shared memory) |
| **Deployment** | Requires `.eim` files | Requires compiled model |
| **Flexibility** | Dynamic model loading | Static model compilation |

## Prerequisites

### EIM Mode
- Edge Impulse model files (`.eim`)
- Unix-like system (Linux, macOS)

### FFI Mode
- `edge-impulse-ffi-rs` crate dependency
- Compiled model integrated into your binary

Some functionality (particularly video capture) requires GStreamer to be installed:
- **macOS**: Install both runtime and development packages from gstreamer.freedesktop.org
- **Linux**: Install required packages (libgstreamer1.0-dev and related packages)

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

## Modules

- `error`: Error types and handling
- `inference`: Model management and inference functionality
- `backends`: Backend abstraction and implementations
- `ingestion`: Data upload and project management
- `types`: Common types and parameters

## License

BSD-3-Clause