FROM rust:1.90.0-bookworm

# Install cross-compilation tools and dependencies
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    build-essential \
    pkg-config \
    libclang-dev \
    clang \
    cmake \
    make \
    libstdc++-11-dev:arm64 \
    libc6-dev:arm64 \
    curl \
    wget \
    unzip \
    git \
    # GStreamer and GLib development packages for examples
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
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
    gfortran

WORKDIR /app

# Install rustfmt for aarch64 target
RUN rustup target add aarch64-unknown-linux-gnu && \
    rustup component add rustfmt

# Create symlinks to make cargo and rustc available in PATH
RUN ln -sf /usr/local/rustup/toolchains/*/bin/cargo /usr/local/bin/cargo && \
    ln -sf /usr/local/rustup/toolchains/*/bin/rustc /usr/local/bin/rustc

# Copy the project files
COPY . .

# Set up environment for cross-compilation
ENV TARGET_LINUX_AARCH64=1
ENV USE_FULL_TFLITE=1
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
ENV CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++

# Create a build script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Building for aarch64-unknown-linux-gnu..."\n\
\n\
# Set cross-compilation environment variables\n\
export TARGET_LINUX_AARCH64=1\n\
export USE_FULL_TFLITE=1\n\
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc\n\
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++\n\
\n\
# Build for aarch64 with FFI feature enabled\n\
cargo build --target aarch64-unknown-linux-gnu --features ffi --release\n\
\n\
# Build examples\n\
echo "Building examples..."\n\
cargo build --target aarch64-unknown-linux-gnu --features ffi --release --examples\n\
\n\
echo "Build completed successfully!"\n\
echo "Output files:"\n\
ls -la target/aarch64-unknown-linux-gnu/release/\n\
' > /usr/local/bin/build-aarch64.sh && chmod +x /usr/local/bin/build-aarch64.sh

# Default command
CMD ["/usr/local/bin/build-aarch64.sh"]