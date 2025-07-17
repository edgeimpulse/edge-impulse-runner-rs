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

# Validate model configuration
echo "Checking model configuration..."

# First, check if there's already a valid model in the local model/ directory
if [ -d "model/edge-impulse-sdk" ] && [ -d "model/model-parameters" ] && [ -d "model/tflite-model" ]; then
    echo "Valid model found in local model/ directory"
    echo "Using existing model for build"

# Check if EI_MODEL is set and points to a valid path
elif [ -n "$EI_MODEL" ]; then
    echo "EI_MODEL is set to: $EI_MODEL"

    if [ -d "$EI_MODEL" ]; then
        echo "EI_MODEL path exists and is a directory"

        # Create model directory if it doesn't exist
        mkdir -p model

        # Copy the model files
        echo "Copying model from $EI_MODEL to local model/ directory..."

        # Copy directories that should exist in a valid model
        for dir in edge-impulse-sdk model-parameters tflite-model; do
            if [ -d "$EI_MODEL/$dir" ]; then
                echo "Copying $dir directory..."
                rm -rf "model/$dir"
                cp -r "$EI_MODEL/$dir" "model/"
            else
                echo "Warning: $dir directory not found in $EI_MODEL"
            fi
        done

        # Also copy tensorflow-lite if it exists
        if [ -d "$EI_MODEL/tensorflow-lite" ]; then
            echo "Copying tensorflow-lite directory..."
            rm -rf "model/tensorflow-lite"
            cp -r "$EI_MODEL/tensorflow-lite" "model/"
        fi

        echo "Model copied successfully!"

        # Verify the copy was successful
        if [ -d "model/edge-impulse-sdk" ] && [ -d "model/model-parameters" ] && [ -d "model/tflite-model" ]; then
            echo "Model validation successful: all required directories are present"
        else
            echo "Error: Model copy failed - required directories are missing"
            echo "Expected: model/edge-impulse-sdk, model/model-parameters, model/tflite-model"
            exit 1
        fi

    else
        echo "Error: EI_MODEL path '$EI_MODEL' is not a directory"
        exit 1
    fi

# Check if EI_PROJECT_ID and EI_API_KEY are set for automatic download
elif [ -n "$EI_PROJECT_ID" ] && [ -n "$EI_API_KEY" ]; then
    echo "EI_PROJECT_ID and EI_API_KEY are set - model will be downloaded during build"
    echo "Project ID: $EI_PROJECT_ID"
    echo "API Key: ${EI_API_KEY:0:8}..."

else
    echo "Error: No valid model configuration found"
    echo ""
    echo "Please provide one of the following:"
    echo "1. Place a valid Edge Impulse model in the local model/ directory"
    echo "2. Set EI_MODEL to point to a directory containing your Edge Impulse model:"
    echo "   export EI_MODEL=/path/to/your/model"
    echo ""
    echo "3. Set EI_PROJECT_ID and EI_API_KEY to download a model from Edge Impulse Studio:"
    echo "   export EI_PROJECT_ID=your-project-id"
    echo "   export EI_API_KEY=your-api-key"
    echo ""
    echo "Example:"
    echo "   EI_MODEL=~/Downloads/model-person-detection ./build-aarch64.sh"
    echo "   or"
    echo "   EI_PROJECT_ID=12345 EI_API_KEY=your-key ./build-aarch64.sh"
    exit 1
fi

# Build the Docker image if it doesn't exist
echo "Building Docker image..."
docker-compose build

# Build the project for aarch64
echo "Building for aarch64..."
docker-compose run --rm aarch64-build bash -c "
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