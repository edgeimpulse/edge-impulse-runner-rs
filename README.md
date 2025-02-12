# Edge Impulse Runner
[![Edge Impulse Runner Tests](https://github.com/ferjm/edge-impulse-runner-rs/actions/workflows/edge-impulse-runner.yml/badge.svg)](https://github.com/ferjm/edge-impulse-runner-rs/actions/workflows/edge-impulse-runner.yml)

A Rust library for running Edge Impulse Linux models (EIM). This crate provides a safe and easy-to-use interface for interacting with Edge Impulse machine learning models compiled for Linux and MacOS.

## Features
- Run Edge Impulse models (.eim files) on Linux and MacOS
- Support for different model types:
  - Classification models
  - Object detection models
- Support for different sensor types:
  - Camera
  - Microphone
- Continuous classification mode support
- Debug output option

## Communication Protocol

The Edge Impulse Runner uses a Unix socket-based IPC mechanism to communicate with the model process. The protocol is JSON-based and follows a request-response pattern.

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

#### 2. Classification
For each inference request:

##### ClassifyMessage (Runner -> Model):
```json
{
    "classify": [0.1, 0.2, 0.3],
    "id": 2,
    "debug": true
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
        SensorType::Other => println!("Other sensor type"),
    }

    // Run inference with normalized features
    let raw_features = vec![128, 128, 128];  // Example raw values
    let features: Vec<f32> = raw_features.into_iter().map(|x| x as f32 / 255.0).collect();
    let result = model.classify(features, None)?;

    // Handle the results based on model type
    match result.result {
        InferenceResult::Classification { classification } => {
            for (label, probability) in classification {
                println!("{}: {:.2}", label, probability);
            }
        }
        InferenceResult::ObjectDetection { bounding_boxes, classification } => {
            for bbox in bounding_boxes {
                println!("Found {} at ({}, {}) with confidence {:.2}",
                    bbox.label, bbox.x, bbox.y, bbox.value);
            }
            if !classification.is_empty() {
                println!("\nOverall classification:");
                for (label, prob) in classification {
                    println!("{}: {:.2}", label, prob);
                }
            }
        }
    }

    Ok(())
}
```

## Examples

All examples are located in the `examples` directory and accept a `--debug` flag to enable debug output.

These examples have only been tested on MacOS.

### Basic Classification
The simplest example allows you to run inference by providing the features array via command line:

```bash
# Format: comma-separated feature values
cargo run --example basic_classify -- --model path/to/model.eim --features "0.1,0.2,0.3"
```

The features array format depends on your model:
- For audio models: Raw audio samples
- For image models: RGB pixel values
- For accelerometer: X, Y, Z values
- For other sensors: Check your model's specifications

### Audio Classification

The repository includes an example that demonstrates audio classification using Edge Impulse models.

To run the audio classification example:

```bash
cargo run --example audio_classify -- --model <path_to_model.eim> --audio <path_to_audio.wav> [--debug]
```

Example output:
```
Audio file specs: WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: Int }
Read 16000 samples from audio file
Model expects 16000 samples
Using 16000 samples for classification
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Classification result: InferenceResponse { success: true, id: 2, result: Classification { classification: {"noise": 0.96875, "no": 0.015625, "yes": 0.01953125} } }
```

### Image Classification
The repository includes an example that demonstrates image classification using Edge Impulse models:

To run the image classification example:
```bash
cargo run --example image_classify -- --model path/to/model.eim --image path/to/image.jpg
```

Example output:
```
Detected objects:
----------------
- mug (90.62%): x=24, y=40, width=8, height=16
```

### Video Classification
The repository includes an example that demonstrates video classification using Edge Impulse models:

To run the video classification example:
```bash
cargo run --example video_classify -- --model path/to/model.eim
```

Example output:
```
Model Parameters:
----------------
ModelParameters {
    axis_count: 1,
    frequency: 0.0,
    has_anomaly: 0,
    image_channel_count: 3,
    image_input_frames: 1,
    image_input_height: 96,
    image_input_width: 96,
    image_resize_mode: "fit-shortest",
    inferencing_engine: 4,
    input_features_count: 9216,
    interval_ms: 1.0,
    label_count: 1,
    labels: [
        "mug",
    ],
    model_type: "constrained_object_detection",
    sensor: 3,
    slice_size: 2304,
    threshold: 0.5,
    use_continuous_mode: false,
}
----------------

Model expects 96x96 input with 3 channels (9216 features)
Image format will be 96x96 with 3 channels
Setting up pipeline for 96x96 input with 3 channels
2025-02-12 07:17:48.758 video_classify[65602:37375812] WARNING: AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras. Please use AVCaptureDeviceTypeContinuityCamera and add NSCameraUseContinuityCameraDeviceType to your Info.plist.
Playing... (Ctrl+C to stop)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
No objects detected
Detected objects: [BoundingBox { label: "mug", value: 0.53125, x: 56, y: 48, width: 8, height: 8 }]
Detected objects: [BoundingBox { label: "mug", value: 0.53125, x: 56, y: 40, width: 8, height: 8 }]
Detected objects: [BoundingBox { label: "mug", value: 0.53125, x: 48, y: 40, width: 8, height: 8 }]
Detected objects: [BoundingBox { label: "mug", value: 0.53125, x: 56, y: 40, width: 8, height: 8 }]
Detected objects: [BoundingBox { label: "mug", value: 0.5625, x: 48, y: 40, width: 8, height: 8 }]
```

## Development
### Running Tests
```cargo test```

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

## Acknowledgments
This crate is designed to work with Edge Impulse's machine learning models. For more information about Edge Impulse and their ML deployment solutions, visit [Edge Impulse](https://edgeimpulse.com/).


## License
This project is licensed under the BSD 3-Clause Clear License - see the LICENSE file for details.