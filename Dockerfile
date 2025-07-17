# Use Ubuntu 22.04 as base image for aarch64 cross-compilation
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TARGET_LINUX_AARCH64=1
ENV USE_FULL_TFLITE=1

# Install system dependencies including GStreamer and GLib development packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    python3 \
    python3-pip \
    wget \
    # Clang and LLVM for bindgen
    clang \
    libclang-dev \
    llvm-dev \
    # GStreamer and GLib development packages for examples
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libglib2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libgdk-pixbuf2.0-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for aarch64-unknown-linux-gnu target
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add aarch64 target
RUN rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation tools
RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# Set up cross-compilation environment variables
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Clone the edge-impulse-ffi-rs repository
RUN git clone https://github.com/edgeimpulse/edge-impulse-ffi-rs.git /edge-impulse-ffi-rs
WORKDIR /edge-impulse-ffi-rs
RUN git checkout 68bc8b2

# Set up the workspace
WORKDIR /workspace

# Copy the project files
COPY . .

# Create a script to run the build
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Building for aarch64..."\n\
export TARGET_LINUX_AARCH64=1\n\
export USE_FULL_TFLITE=1\n\
export EI_ENGINE=tflite-eon\n\
\n\
# Set model path if provided\n\
if [ -n "$EI_MODEL" ]; then\n\
    echo "Using model from: $EI_MODEL"\n\
    export EI_MODEL="$EI_MODEL"\n\
fi\n\
\n\
# Build the library\n\
cargo build --target aarch64-unknown-linux-gnu --release\n\
\n\
# Build examples\n\
echo "Building examples..."\n\
cargo build --target aarch64-unknown-linux-gnu --release --examples\n\
\n\
echo "Build completed successfully!"\n\
echo "Library location: target/aarch64-unknown-linux-gnu/release/libedge_impulse_runner.rlib"\n\
echo "Examples location: target/aarch64-unknown-linux-gnu/release/examples/"\n\
' > /usr/local/bin/build-aarch64.sh && chmod +x /usr/local/bin/build-aarch64.sh

# Default command
CMD ["/usr/local/bin/build-aarch64.sh"]