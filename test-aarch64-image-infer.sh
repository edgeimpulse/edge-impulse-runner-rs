#!/bin/bash

# Script to test the image_infer example in aarch64 Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -i, --image PATH        Use specific image file (default: creates test image)"
    echo "  -d, --debug             Enable debug output"
    echo "  -e, --env VAR=VAL       Set environment variable (can be used multiple times)"
    echo "  -c, --clean             Clean build before testing"
    echo "  -m, --model PATH        Set EI_MODEL environment variable"
    echo ""
    echo "Examples:"
    echo "  $0 -d                    # Run with debug and default test image"
    echo "  $0 -i my_image.jpg -d    # Run with specific image and debug"
    echo "  $0 -e EI_MODEL=/path/to/model -c  # Run with custom model and clean build"
}

# Parse command line arguments
IMAGE_PATH=""
DEBUG_FLAG=""
ENV_VARS=""
CLEAN_BUILD=false
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -i|--image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        -e|--env)
            ENV_VARS="$ENV_VARS -e $2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -m|--model)
            MODEL_PATH="$2"
            ENV_VARS="$ENV_VARS -e EI_MODEL=$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

print_status "=== Testing image_infer example in aarch64 Docker container ==="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed or not in PATH"
    exit 1
fi

# Create assets directory if it doesn't exist
mkdir -p examples/assets

# Handle image path
if [ -z "$IMAGE_PATH" ]; then
    print_status "Creating test image in examples/assets/..."
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" | base64 -d > examples/assets/test_image.png
    IMAGE_PATH="examples/assets/test_image.png"
    DOCKER_IMAGE_PATH="$IMAGE_PATH"
    print_success "Created test image: $IMAGE_PATH"
else
    if [ ! -f "$IMAGE_PATH" ]; then
        print_error "Image file not found: $IMAGE_PATH"
        exit 1
    fi

    # If the image is outside the current directory, copy it to the assets folder
    IMAGE_FILENAME=$(basename "$IMAGE_PATH")
    if [[ "$IMAGE_PATH" != "./"* ]] && [[ "$IMAGE_PATH" != "$(pwd)/"* ]]; then
        print_status "Copying image to examples/assets/ for Docker access..."
        cp "$IMAGE_PATH" "examples/assets/$IMAGE_FILENAME"
        DOCKER_IMAGE_PATH="examples/assets/$IMAGE_FILENAME"
        print_success "Image copied to: $DOCKER_IMAGE_PATH"
    else
        DOCKER_IMAGE_PATH="$IMAGE_PATH"
    fi

    print_success "Using image: $DOCKER_IMAGE_PATH"
fi

# Build Docker image if needed
print_status "Building Docker image..."
docker-compose build

# Build the example
print_status "Building image_infer example..."
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning previous build..."
    docker-compose run --rm $ENV_VARS aarch64-build bash -c "
        cargo clean
        cargo build --example image_infer --target aarch64-unknown-linux-gnu --features ffi --release
    "
else
    docker-compose run --rm $ENV_VARS aarch64-build bash -c "
        cargo build --example image_infer --target aarch64-unknown-linux-gnu --features ffi --release
    "
fi

# Check if build was successful
if docker-compose run --rm $ENV_VARS aarch64-build bash -c "test -f './target/aarch64-unknown-linux-gnu/release/examples/image_infer'"; then
    print_success "Example built successfully!"
    docker-compose run --rm $ENV_VARS aarch64-build bash -c "ls -la target/aarch64-unknown-linux-gnu/release/examples/image_infer"
else
    print_error "Build failed - binary not found"
    exit 1
fi

# Run the example
print_status "Running image_infer example..."
docker-compose run --rm $ENV_VARS aarch64-build bash -c "
    echo 'Running example with image: $DOCKER_IMAGE_PATH'
    echo 'Debug flag: $DEBUG_FLAG'

    # Check if binary exists
    if [ ! -f './target/aarch64-unknown-linux-gnu/release/examples/image_infer' ]; then
        echo 'Binary not found!'
        exit 1
    fi

    # Run the example
    ./target/aarch64-unknown-linux-gnu/release/examples/image_infer --image $DOCKER_IMAGE_PATH $DEBUG_FLAG

    echo 'Example execution completed!'
"

if [ $? -eq 0 ]; then
    print_success "Example ran successfully!"
    print_success "=== Test completed successfully ==="
else
    print_error "Example execution failed!"
    exit 1
fi