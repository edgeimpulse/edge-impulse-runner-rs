# Edge Impulse Runner
A Rust library for running Edge Impulse Linux models (EIM). This crate provides a safe and easy-to-use interface for interacting with Edge Impulse machine learning models compiled for Linux.

## Features
* Load and initialize Edge Impulse Linux models
* Automatic socket handling and connection management
* Robust error handling
* Debug mode for development and troubleshooting

## Installation
Add this to your `Cargo.toml`:
```
[dependencies]
edge_impulse_runner = "0.1.0"
```

## Quick Start
```rust
use edge_impulse_runner::EimModel;

fn main() {
    // Create a new model instance
    match EimModel::new("path/to/model.eim") {
        Ok(model) => {
            println!("Model loaded successfully!");
            // Use the model...
        },
        Err(e) => eprintln!("Failed to load model: {}", e),
    }
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
The crate provides detailed error types for various failure scenarios:
* `FileError`: Issues with accessing the EIM file
* `InvalidPath`: Invalid file path or extension
* `ExecutionError`: Problems running the EIM model
* `SocketError`: Unix socket communication issues
* `JsonError`: JSON serialization/deserialization errors

### Development
#### Running Tests
```cargo test```

## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).