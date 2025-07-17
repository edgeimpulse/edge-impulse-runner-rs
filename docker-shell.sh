#!/bin/bash

# Script to open an interactive shell in the aarch64 Docker container

echo "=== Opening interactive shell in aarch64 Docker container ==="
echo "You can now manually build and test the examples."
echo ""
echo "Useful commands:"
echo "  cargo build --example image_infer --target aarch64-unknown-linux-gnu --features ffi --release"
echo "  ./target/aarch64-unknown-linux-gnu/release/examples/image_infer --image examples/assets/test_image.png --debug"
echo "  cargo clean"
echo ""

# Build environment variables string from command line arguments
ENV_STRING=""
for arg in "$@"; do
    if [[ $arg == *"="* ]]; then
        ENV_STRING="$ENV_STRING -e $arg"
    fi
done

# Open interactive shell
docker-compose run --rm $ENV_STRING aarch64-build bash