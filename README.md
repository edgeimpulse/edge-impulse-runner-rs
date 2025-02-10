# Edge Impulse Runner
A Rust library for running Edge Impulse Linux models (EIM). This crate provides a safe and easy-to-use interface for interacting with Edge Impulse machine learning models compiled for Linux.

## Features
- Run Edge Impulse models (.eim files) on Linux
- Support for different model types:
  - Classification models
  - Object detection models
  - Anomaly detection
- Support for different sensor types:
  - Camera
  - Microphone
  - Accelerometer
  - Positional
- Continuous classification mode support
- Debug output option

## Installation
Add this to your `Cargo.toml`:
```[dependencies]
edge_impulse_runner = "0.1.0"
```

## Quick Start
```rust
use edge_impulse_runner::EimModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new model instance
    let mut model = EimModel::new("path/to/model.eim")?;

    // Get model information
    let params = model.parameters()?;
    println!("Model type: {}", params.model_type);

    // Check sensor type
    match model.sensor_type()? {
        SensorType::Camera => println!("Camera model"),
        SensorType::Microphone => println!("Audio model"),
        SensorType::Accelerometer => println!("Motion model"),
        SensorType::Positional => println!("Position model"),
        SensorType::Unknown => println!("Unknown sensor type"),
    }

    // Run inference with some features
    let features = vec![0.1, 0.2, 0.3];
    let result = model.classify(features, None)?;

    // Handle the results based on model type
    match result.result {
        InferenceResult::Classification { classification } => {
            for (label, probability) in classification {
                println!("{}: {:.2}", label, probability);
            }
        }
        InferenceResult::ObjectDetection { bounding_boxes, .. } => {
            for bbox in bounding_boxes {
                println!("Found {} at ({}, {}) with confidence {:.2}",
                    bbox.label, bbox.x, bbox.y, bbox.value);
            }
        }
    }

    Ok(())
}
```

For continuous classification:

```rust
use edge_impulse_runner::EimModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = EimModel::new("path/to/model.eim")?;

    // Ensure model supports continuous mode
    let params = model.parameters()?;
    if !params.use_continuous_mode {
        println!("Model doesn't support continuous mode");
        return Ok(());
    }

    // Run continuous classification
    let features = vec![0.1, 0.2, 0.3];
    let result = model.classify_continuous(features)?;

    // Handle results...
    Ok(())
}
```

### Example Application
The crate includes an example application that demonstrates how to load and use an EIM model. To run it:
```
# Basic usage
cargo run --example run_model -- --model path/to/model.eim

# With debug output
cargo run --example run_model -- --model path/to/model.eim --debug
```

### Error Handling
The crate provides detailed error types through `EimError`:

- `FileError`: Issues with accessing the EIM file
- `InvalidPath`: Invalid file path or extension
- `ExecutionError`: Problems running the EIM model
- `SocketError`: Unix socket communication issues
- `JsonError`: JSON serialization/deserialization errors
- `InvalidInput`: Invalid input features or parameters
- `InvalidOperation`: Operation not supported by the model (e.g., continuous mode)

Example error handling:
```rust
use edge_impulse_runner::{EimModel, EimError};

fn main() {
    match EimModel::new("path/to/model.eim") {
        Ok(mut model) => {
            match model.classify(vec![0.1, 0.2, 0.3], None) {
                Ok(result) => println!("Classification successful"),
                Err(EimError::InvalidInput(msg)) => println!("Invalid input: {}", msg),
                Err(EimError::ExecutionError(msg)) => println!("Model execution failed: {}", msg),
                Err(e) => println!("Other error: {}", e),
            }
        },
        Err(EimError::FileError(e)) => println!("File error: {}", e),
        Err(e) => println!("Error creating model: {}", e),
    }
}
```

### Development
#### Running Tests
```cargo test```

## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).


## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.