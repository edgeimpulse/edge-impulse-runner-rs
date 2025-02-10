#[cfg(test)]
mod tests {
    use crate::{EimError, EimModel};
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

        // Create a shell script that simulates an EIM model
        // The script uses socat to:
        // - Create a Unix socket (UNIX-LISTEN)
        // - Accept connections (fork)
        // - Respond with a valid JSON message
        let mock_script = r#"#!/bin/sh
SOCKET_PATH=$1
socat UNIX-LISTEN:$SOCKET_PATH,fork SYSTEM:'echo "{\"success\":true,\"error\":null}\n"'
"#;

        let mut file = File::create(&mock_path).unwrap();
        file.write_all(mock_script.as_bytes()).unwrap();

        // Make the script executable (required for Unix systems)
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&mock_path).unwrap().permissions();
        perms.set_mode(0o755); // rwxr-xr-x permissions
        std::fs::set_permissions(&mock_path, perms).unwrap();

        mock_path
    }

    #[test]
    fn test_missing_file_error() {
        // Verify that attempting to load a non-existent file returns FileError
        let result = EimModel::new("unknown.eim");
        match result {
            Err(EimError::FileError(_)) => (),
            _ => panic!("Expected FileError when file doesn't exist"),
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

        // Create and set up the mock EIM executable
        let mock_path = create_mock_eim();
        let mut mock_path_with_eim = mock_path.clone();
        mock_path_with_eim.set_extension("eim");
        std::fs::rename(&mock_path, &mock_path_with_eim).unwrap();

        // Test the connection
        let result = EimModel::new(&mock_path_with_eim);
        assert!(
            result.is_ok(),
            "Failed to create EIM model: {:?}",
            result.err()
        );

        // Clean up the test file
        std::fs::remove_file(&mock_path_with_eim).unwrap();
    }

    #[test]
    fn test_connection_timeout() {
        // Create a temporary directory
        let temp_dir = tempfile::tempdir().unwrap();
        let socket_dir = temp_dir.path().join("nonexistent");
        let temp_file = socket_dir.join("dummy.eim");

        // Create the directory structure
        std::fs::create_dir_all(socket_dir).unwrap();

        // Create the executable
        let script = "#!/bin/sh\nsleep 10\n";  // Sleep long enough for timeout
        std::fs::write(&temp_file, script).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&temp_file).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&temp_file, perms).unwrap();
        }

        // Test that we get the expected timeout error
        let result = EimModel::new(&temp_file);
        assert!(matches!(result,
            Err(EimError::SocketError(ref msg)) if msg.contains("Timeout waiting for socket")
        ), "Expected timeout error, got {:?}", result);
    }
}
