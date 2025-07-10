//! FFI module for Edge Impulse C++ SDK bindings
//!
//! This module provides safe Rust bindings for the Edge Impulse C++ SDK,
//! allowing direct FFI calls to run inference on trained models.

mod wrapper;

// Re-export the wrapper's public API
pub use wrapper::*;
