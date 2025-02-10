use thiserror::Error;

#[derive(Error, Debug)]
pub enum EimError {
    #[error("Failed to access EIM file: {0}")]
    FileError(#[from] std::io::Error),
    #[error("Invalid EIM file path")]
    InvalidPath,
    #[error("Failed to execute EIM model: {0}")]
    ExecutionError(String),
    #[error("Socket communication error: {0}")]
    SocketError(String),
    #[error("Failed to serialize/deserialize JSON: {0}")]
    JsonError(#[from] serde_json::Error),
}