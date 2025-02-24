//! Error types for the Edge Impulse Runner.
//!
//! This module defines the error types that can occur during model loading,
//! execution, and communication. It provides a comprehensive error handling
//! system that helps identify and debug issues when working with Edge Impulse
//! models.
//!
//! The main error type is `EimError`, which encompasses all possible error
//! conditions that can occur within the library.

use thiserror::Error;
use std::io;
use std::fmt;
use std::error::Error;

/// Represents all possible errors that can occur in the Edge Impulse Runner.
///
/// This enum implements the standard Error trait using thiserror and provides
/// detailed error messages for each error case. It handles both internal errors
/// and wrapped errors from external dependencies.
#[derive(Error, Debug)]
pub enum EimError {
    /// Indicates a failure in file system operations when accessing the EIM file.
    ///
    /// This error occurs when there are problems reading, writing, or accessing
    /// the model file. It wraps the standard IO error for more details.
    #[error("Failed to access EIM file: {0}")]
    FileError(#[from] std::io::Error),

    /// Indicates that the provided path to the EIM file is invalid.
    ///
    /// This error occurs when:
    /// - The path doesn't exist
    /// - The file extension is not .eim
    /// - The path points to a directory instead of a file
    #[error("Invalid EIM file path")]
    InvalidPath,

    /// Indicates a failure during model execution.
    ///
    /// This error occurs when:
    /// - The model process crashes
    /// - The model fails to initialize
    /// - There's an internal model error during inference
    #[error("Failed to execute EIM model: {0}")]
    ExecutionError(String),

    /// Indicates a failure in Unix socket communication.
    ///
    /// This error occurs when:
    /// - The socket connection fails
    /// - Messages can't be sent or received
    /// - The socket connection is unexpectedly closed
    #[error("Socket communication error: {0}")]
    SocketError(String),

    /// Indicates a failure in JSON serialization or deserialization.
    ///
    /// This error occurs when:
    /// - Messages can't be encoded to JSON
    /// - Responses can't be decoded from JSON
    /// - The JSON structure doesn't match expected schema
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Indicates that the provided input data is invalid.
    ///
    /// This error occurs when:
    /// - Input features don't match expected dimensions
    /// - Input values are out of valid ranges
    /// - Required input parameters are missing
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Indicates that an invalid operation was attempted.
    ///
    /// This error occurs when:
    /// - Operations are called in wrong order
    /// - Unsupported operations are requested
    /// - Operation parameters are incompatible
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

#[derive(Debug)]
pub enum IngestionError {
    Server {
        status_code: u16,
        message: String,
    },
    Config(String),
    Network(reqwest::Error),
    Json(serde_json::Error),
    Header(reqwest::header::InvalidHeaderValue),
    Io(io::Error),
}

impl fmt::Display for IngestionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestionError::Server { status_code, message } => {
                write!(f, "Server error {}: {}", status_code, message)
            }
            IngestionError::Config(msg) => write!(f, "Configuration error: {}", msg),
            IngestionError::Network(e) => write!(f, "Network error: {}", e),
            IngestionError::Json(e) => write!(f, "JSON error: {}", e),
            IngestionError::Header(e) => write!(f, "Header error: {}", e),
            IngestionError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl Error for IngestionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IngestionError::Network(e) => Some(e),
            IngestionError::Json(e) => Some(e),
            IngestionError::Header(e) => Some(e),
            IngestionError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for IngestionError {
    fn from(err: reqwest::Error) -> Self {
        IngestionError::Network(err)
    }
}

impl From<serde_json::Error> for IngestionError {
    fn from(err: serde_json::Error) -> Self {
        IngestionError::Json(err)
    }
}

impl From<reqwest::header::InvalidHeaderValue> for IngestionError {
    fn from(err: reqwest::header::InvalidHeaderValue) -> Self {
        IngestionError::Header(err)
    }
}

impl From<io::Error> for IngestionError {
    fn from(err: io::Error) -> Self {
        IngestionError::Io(err)
    }
}
