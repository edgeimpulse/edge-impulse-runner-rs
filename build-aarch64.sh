#!/bin/bash

# Script to build edge-impulse-runner-rs for aarch64 using Docker

set -e

echo "=== Building edge-impulse-runner-rs for aarch64 ==="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed or not in PATH"
    exit 1
fi

# Copy model to local model/ directory if EI_MODEL is set
if [ -n "$EI_MODEL" ]; then
    echo "Copying model from $EI_MODEL to local model/ directory..."

    # Create model directory if it doesn't exist
    mkdir -p model

    # Copy the model files
    if [ -d "$EI_MODEL" ]; then
        # Copy directories that should exist in a valid model
        for dir in edge-impulse-sdk model-parameters tflite-model; do
            if [ -d "$EI_MODEL/$dir" ]; then
                echo "Copying $dir directory..."
                rm -rf "model/$dir"
                cp -r "$EI_MODEL/$dir" "model/"
            fi
        done

        # Also copy tensorflow-lite if it exists (for full TFLite builds)
        if [ -d "$EI_MODEL/tensorflow-lite" ]; then
            echo "Copying tensorflow-lite directory..."
            rm -rf "model/tensorflow-lite"
            cp -r "$EI_MODEL/tensorflow-lite" "model/"
        fi

        echo "Model copied successfully!"
    else
        echo "Error: EI_MODEL path $EI_MODEL does not exist"
        exit 1
    fi
fi

# Build the Docker image if it doesn't exist
echo "Building Docker image..."
docker-compose build

# Build the project for aarch64
echo "Building for aarch64..."
# Set EI_MODEL to the local model directory where we copied the model
docker-compose run --rm -e EI_MODEL="/app/model" aarch64-build bash -c "
    echo 'Building edge-impulse-runner-rs for aarch64...'

    # Clean previous builds
    cargo clean

    # Build with FFI feature enabled
    cargo build --target aarch64-unknown-linux-gnu --features ffi --release

    # Build examples
    cargo build --target aarch64-unknown-linux-gnu --features ffi --release --examples

    echo 'Build completed successfully!'
    echo 'Binary location: target/aarch64-unknown-linux-gnu/release/'
    ls -la target/aarch64-unknown-linux-gnu/release/
"

echo "=== Build completed ==="