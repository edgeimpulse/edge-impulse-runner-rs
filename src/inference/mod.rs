pub mod messages;
pub mod model;
// Removed eim_original module - functionality moved to backends/eim.rs

#[cfg(test)]
mod tests {
    use crate::{EdgeImpulseModel, EimError};
    use std::env;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use std::process::Command;
    use tempfile;

    /// Creates a mock EIM executable for testing
    ///
    /// This function creates a shell script that simulates an EIM model by:
    /// 1. Accepting a socket path argument
    /// 2. Creating a Unix socket at that path using socat
    /// 3. Responding to the hello message with a valid JSON response
    fn create_mock_eim() -> std::path::PathBuf {
        let manifest_dir =
            env::var("CARGO_MANIFEST_DIR").expect("Failed to get manifest directory");
        let mock_path = Path::new(&manifest_dir).join("mock_eim.sh");
        let response_path = Path::new(&manifest_dir).join("mock_response.json");

        // Create the response JSON file
        let response_json = r#"{"success":true,"id":1,"model_parameters":{"axis_count":3,"frequency":62.5,"has_anomaly":1,"image_channel_count":0,"image_input_frames":0,"image_input_height":0,"image_input_width":0,"image_resize_mode":"none","inferencing_engine":4,"input_features_count":375,"interval_ms":16,"label_count":6,"labels":["drink","fistbump","idle","snake","updown","wave"],"model_type":"classification","sensor":2,"slice_size":31,"threshold":0.6,"use_continuous_mode":false},"project":{"deploy_version":271,"id":1,"name":"Test Project","owner":"Test Owner"}}"#;
        std::fs::write(&response_path, response_json).unwrap();

        // Create the mock script that reads from the response file
        let mock_script = format!(
            r#"#!/bin/sh
SOCKET_PATH=$1
socat UNIX-LISTEN:$SOCKET_PATH,fork SYSTEM:'cat {}'"#,
            response_path.display()
        );

        let mut file = File::create(&mock_path).unwrap();
        file.write_all(mock_script.as_bytes()).unwrap();

        // Make the script executable
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&mock_path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&mock_path, perms).unwrap();

        mock_path
    }

    #[test]
    fn test_missing_file_error() {
        // Create a temporary directory for the socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("test.socket");

        // Test with a non-existent file
        let result = EdgeImpulseModel::new_with_socket("unknown.eim", &socket_path);
        match result {
            Err(EimError::ExecutionError(msg)) if msg.contains("No such file") => (),
            other => panic!("Expected ExecutionError for missing file, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_extension() {
        // Verify that attempting to load a file without .eim extension returns InvalidPath
        let temp_file = std::env::temp_dir().join("test.txt");
        std::fs::write(&temp_file, "dummy content").unwrap();

        let result = EimModel::new(&temp_file);
        match result {
            Err(EimError::InvalidPath) => (),
            _ => panic!("Expected InvalidPath when file has wrong extension"),
        }
    }

    #[test]
    fn test_successful_connection() {
        // Check if socat is available (required for this test)
        let socat_check = Command::new("which")
            .arg("socat")
            .output()
            .expect("Failed to check for socat");

        if !socat_check.status.success() {
            println!("Skipping test: socat is not installed");
            return;
        }

        // Create a temporary directory for the socket
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("test.socket");

        // Create and set up the mock EIM executable
        let mock_path = create_mock_eim();
        let response_path = mock_path.with_extension("json");
        let mut mock_path_with_eim = mock_path.clone();
        mock_path_with_eim.set_extension("eim");
        std::fs::rename(&mock_path, &mock_path_with_eim).unwrap();

        // Test the connection with the custom socket path
        let result = EimModel::new_with_socket(&mock_path_with_eim, &socket_path);
        assert!(
            result.is_ok(),
            "Failed to create EIM model: {:?}",
            result.err()
        );

        // Clean up the test files
        if mock_path_with_eim.exists() {
            std::fs::remove_file(&mock_path_with_eim).unwrap_or_else(|e| {
                println!("Warning: Failed to remove mock EIM file: {}", e);
            });
        }
        if response_path.exists() {
            std::fs::remove_file(&response_path).unwrap_or_else(|e| {
                println!("Warning: Failed to remove response file: {}", e);
            });
        }
    }

    #[test]
    fn test_connection_timeout() {
        // Create a temporary directory
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_path = temp_dir.path().join("test.socket");
        let model_path = temp_dir.path().join("dummy.eim");

        // Create the executable
        let script = "#!/bin/sh\nsleep 10\n"; // Sleep long enough for timeout
        std::fs::write(&model_path, script).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&model_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&model_path, perms).unwrap();
        }

        // Test that we get the expected timeout error
        let result = EimModel::new_with_socket(&model_path, &socket_path);
        assert!(
            matches!(result,
                Err(EimError::SocketError(ref msg)) if msg.contains("Timeout waiting for socket")
            ),
            "Expected timeout error, got {:?}",
            result
        );
    }
}
