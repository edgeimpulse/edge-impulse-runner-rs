use crate::error::IngestionError;
use hmac::{Hmac, Mac};
use mime_guess::from_path;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;
const DEFAULT_INGESTION_HOST: &str = "https://ingestion.edgeimpulse.com";

/// Edge Impulse Ingestion API client
///
/// This module provides a client implementation for the Edge Impulse Ingestion API, which allows
/// uploading data samples and files to Edge Impulse for machine learning training, testing, and
/// anomaly detection.
///
/// # API Endpoints
///
/// The client supports two types of endpoints:
///
/// * Data endpoints (legacy):
///   - `/api/training/data`
///   - `/api/testing/data`
///   - `/api/anomaly/data`
///
/// * File endpoints:
///   - `/api/training/files`
///   - `/api/testing/files`
///   - `/api/anomaly/files`
///
/// # Examples
///
/// ```no_run
/// use edge_impulse_runner::ingestion::{Ingestion, Category, Sensor, UploadSampleParams};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a new client
/// let client = Ingestion::new("your-api-key".to_string())
///     .with_hmac("optional-hmac-key".to_string());
///
/// // Upload a file
/// let response = client.upload_file(
///     "data.wav",
///     Category::Training,
///     Some("walking".to_string()),
///     None
/// ).await?;
///
/// // Upload sensor data
/// let sensors = vec![
///     Sensor {
///         name: "accX".to_string(),
///         units: "m/s2".to_string(),
///     }
/// ];
/// let values = vec![vec![1.0, 2.0, 3.0]];
///
/// let params = UploadSampleParams {
///     device_id: "device-id",
///     device_type: "CUSTOM_DEVICE",
///     sensors,
///     values,
///     interval_ms: 100.0,
///     label: Some("walking".to_string()),
///     category: "training",
/// };
///
/// let response = client.upload_sample(params).await?;
/// # Ok(())
/// # }
/// ```

#[derive(Debug, Serialize, Deserialize)]
struct Protected {
    ver: String,
    alg: String,
    iat: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    device_name: String,
    device_type: String,
    interval_ms: f64,
    sensors: Vec<Sensor>,
    values: Vec<Vec<f64>>,
}

/// Represents a sensor in the Edge Impulse data format
///
/// Each sensor has a name and units that describe the type of data being collected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor {
    pub name: String,
    pub units: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataMessage {
    protected: Protected,
    signature: String,
    payload: Payload,
}

/// Data category for Edge Impulse uploads
///
/// Determines which API endpoint will be used for the upload:
/// - Training: Used for gathering training data
/// - Testing: Used for gathering testing data
/// - Anomaly: Used for anomaly detection data
#[derive(Debug, Clone, Copy)]
pub enum Category {
    Training,
    Testing,
    Anomaly,
}

impl Category {
    fn as_str(&self) -> &'static str {
        match self {
            Category::Training => "training",
            Category::Testing => "testing",
            Category::Anomaly => "anomaly",
        }
    }
}

/// Parameters for uploading a sample
#[derive(Debug)]
pub struct UploadSampleParams<'a> {
    /// Device identifier
    pub device_id: &'a str,
    /// Type of device
    pub device_type: &'a str,
    /// List of sensors
    pub sensors: Vec<Sensor>,
    /// Sample values
    pub values: Vec<Vec<f64>>,
    /// Interval in milliseconds
    pub interval_ms: f64,
    /// Optional label for the sample
    pub label: Option<String>,
    /// Category (training, testing, or anomaly)
    pub category: &'a str,
}

/// Edge Impulse Ingestion API client
///
/// This struct provides methods to interact with the Edge Impulse Ingestion API.
/// It supports uploading both raw sensor data and files to Edge Impulse for
/// machine learning training, testing, and anomaly detection.
pub struct Ingestion {
    api_key: String,
    hmac_key: Option<String>,
    host: String,
    debug: bool,
}

impl Ingestion {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            hmac_key: None,
            host: DEFAULT_INGESTION_HOST.to_string(),
            debug: false,
        }
    }

    pub fn with_host(api_key: String, host: String) -> Self {
        Self {
            api_key,
            hmac_key: None,
            host,
            debug: false,
        }
    }

    pub fn with_hmac(mut self, hmac_key: String) -> Self {
        self.hmac_key = Some(hmac_key);
        self
    }

    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    async fn create_signature(&self, data: &[u8]) -> Result<String, IngestionError> {
        if let Some(hmac_key) = &self.hmac_key {
            let mut mac = Hmac::<Sha256>::new_from_slice(hmac_key.as_bytes())
                .map_err(|e| IngestionError::Config(e.to_string()))?;
            mac.update(data);
            let result = mac.finalize();
            Ok(hex::encode(result.into_bytes()))
        } else {
            Ok("0".repeat(64))
        }
    }

    pub async fn upload_sample(
        &self,
        params: UploadSampleParams<'_>,
    ) -> Result<String, IngestionError> {
        if self.debug {
            println!("=== Request Details ===");
            println!("URL: {}/api/{}/data", self.host, params.category);
            println!("Device ID: {}", params.device_id);
            println!("Device Type: {}", params.device_type);
            println!("Sensors: {:?}", params.sensors);
            println!(
                "Data size: {} sensors, {} samples",
                params.sensors.len(),
                params.values.len()
            );
        }

        debug!("Creating data message");
        let payload = Payload {
            device_name: params.device_id.to_string(),
            device_type: params.device_type.to_string(),
            interval_ms: params.interval_ms,
            sensors: params.sensors.clone(),
            values: params.values.iter().map(|v| v.to_vec()).collect(),
        };

        let message = DataMessage {
            protected: Protected {
                ver: "v1".to_string(),
                alg: "HS256".to_string(),
                iat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            signature: "0".repeat(64),
            payload,
        };

        debug!("Serializing data message");
        let json = serde_json::to_string(&message)?;

        if let Some(ref _hmac_key) = self.hmac_key {
            debug!("Creating signature for data");
            let signature = self.create_signature(json.as_bytes()).await?;
            debug!("Generated signature: {}", signature);
        }

        debug!("Creating multipart form");
        let form = reqwest::multipart::Form::new().text("data", json);

        let mut headers = reqwest::header::HeaderMap::new();
        debug!("Setting up headers");
        headers.insert("x-api-key", self.api_key.parse()?);
        headers.insert("x-file-name", format!("{}.json", params.device_id).parse()?);

        if let Some(label) = params.label {
            debug!("Adding label header: {}", label);
            headers.insert("x-label", urlencoding::encode(&label).parse()?);
        }

        if self.debug {
            println!("=== Request Headers ===");
            println!("{:#?}", &headers);
        }

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/api/{}/data", self.host, params.category))
            .headers(headers.clone())
            .multipart(form)
            .send()
            .await?;

        let status = response.status();

        if self.debug {
            println!("=== Response ===");
            println!("Status: {status}");
            println!("Headers: {:#?}", response.headers());
        }

        let body = response.text().await?;

        if self.debug {
            println!("Body: {body}");
        }

        if !status.is_success() {
            return Err(IngestionError::Server {
                status_code: status.as_u16(),
                message: body,
            });
        }

        Ok(body)
    }

    /// Upload a file to Edge Impulse using the /files endpoint
    pub async fn upload_file<P: AsRef<Path>>(
        &self,
        file_path: P,
        category: Category,
        label: Option<String>,
        options: Option<UploadOptions>,
    ) -> Result<String, IngestionError> {
        let path = file_path.as_ref();

        // Verify the file exists
        if !path.exists() {
            return Err(IngestionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {path:?}"),
            )));
        }

        // Get the mime type of the file
        let mime_type = from_path(path).first_or_octet_stream().to_string();

        if self.debug {
            println!("Detected mime type: {mime_type}");
        }

        // Read the file
        let file_data = std::fs::read(path)?;

        // Create the multipart form
        let form = reqwest::multipart::Form::new().part(
            "data",
            reqwest::multipart::Part::bytes(file_data)
                .file_name(
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file")
                        .to_string(),
                )
                .mime_str(&mime_type)?,
        );

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", self.api_key.parse()?);

        if let Some(label) = label {
            headers.insert("x-label", urlencoding::encode(&label).parse()?);
        }

        // Add optional headers from UploadOptions
        if let Some(opts) = options {
            if opts.disallow_duplicates {
                headers.insert("x-disallow-duplicates", "1".parse()?);
            }
            if opts.add_date_id {
                headers.insert("x-add-date-id", "1".parse()?);
            }
        }

        if self.debug {
            println!("=== Request Headers ===");
            println!("{:#?}", &headers);
        }

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/api/{}/files", self.host, category.as_str()))
            .headers(headers.clone())
            .multipart(form)
            .send()
            .await?;

        let status = response.status();

        if self.debug {
            println!("=== Response ===");
            println!("Status: {status}");
            println!("Headers: {:#?}", response.headers());
        }

        let body = response.text().await?;

        if self.debug {
            println!("Body: {body}");
        }

        if !status.is_success() {
            return Err(IngestionError::Server {
                status_code: status.as_u16(),
                message: body,
            });
        }

        Ok(body)
    }
}

/// Options for file uploads to Edge Impulse
///
/// These options correspond to various headers that can be set when uploading files.
#[derive(Debug, Default)]
pub struct UploadOptions {
    /// When set, the server checks the hash of the message against your current dataset
    pub disallow_duplicates: bool,
    /// Add a date ID to the filename
    pub add_date_id: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use tracing::error;
    use tracing_test::traced_test;

    fn create_test_sensors() -> Vec<Sensor> {
        vec![Sensor {
            name: "accelerometer".to_string(),
            units: "m/s2".to_string(),
        }]
    }

    fn create_test_values() -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    }

    #[test]
    #[traced_test]
    fn test_ingestion_creation() {
        let ingestion = Ingestion::new("test_key".to_string());
        assert_eq!(ingestion.api_key, "test_key");
        assert_eq!(ingestion.host, DEFAULT_INGESTION_HOST);
        assert!(ingestion.hmac_key.is_none());

        let ingestion_with_host =
            Ingestion::with_host("test_key".to_string(), "http://custom.host".to_string());
        assert_eq!(ingestion_with_host.host, "http://custom.host");

        let ingestion_with_hmac =
            Ingestion::new("test_key".to_string()).with_hmac("hmac_key".to_string());
        assert!(ingestion_with_hmac.hmac_key.is_some());
        assert_eq!(ingestion_with_hmac.hmac_key.unwrap(), "hmac_key");
    }

    #[test]
    fn test_successful_upload() {
        let mut server = Server::new();

        let mock = server
            .mock("POST", "/api/training/data")
            .with_header("x-api-key", "test_key")
            .with_header("x-file-name", "test_device.json")
            .with_header("content-type", "multipart/form-data")
            .with_status(200)
            .with_body("OK")
            .create();

        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let ingestion = Ingestion::with_host("test_key".to_string(), server.url());

            let params = UploadSampleParams {
                device_id: "test_device",
                device_type: "CUSTOM_DEVICE",
                sensors: create_test_sensors(),
                values: create_test_values(),
                interval_ms: 100.0,
                label: Some("walking".to_string()),
                category: "training",
            };

            let result = ingestion.upload_sample(params).await;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "OK");
        });

        mock.assert();
    }

    #[test]
    #[traced_test]
    fn test_upload_with_hmac() {
        let mut server = Server::new();
        debug!("Mock server created at: {}", server.url());

        let mock = server
            .mock("POST", "/api/training/data")
            .match_header("x-api-key", "test_key")
            .match_header("x-file-name", "test_device.json")
            .match_header(
                "content-type",
                mockito::Matcher::Regex("multipart/form-data.*".to_string()),
            )
            .with_status(200)
            .with_body("OK")
            .expect(1)
            .create();
        debug!("Mock endpoint created");

        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let ingestion = Ingestion::with_host("test_key".to_string(), server.url())
                .with_hmac("test_hmac".to_string());
            debug!("Created ingestion client with HMAC");

            let sensors = create_test_sensors();
            let values = create_test_values();
            debug!(
                "Test data created: sensors={:?}, values={:?}",
                sensors, values
            );

            let params = UploadSampleParams {
                device_id: "test_device",
                device_type: "CUSTOM_DEVICE",
                sensors,
                values,
                interval_ms: 100.0,
                label: None,
                category: "training",
            };

            let result = ingestion.upload_sample(params).await;

            match &result {
                Ok(response) => debug!("Upload successful: {}", response),
                Err(e) => error!("Upload failed: {:?}", e),
            }

            assert!(result.is_ok(), "Upload failed: {:?}", result.err().unwrap());

            mock.assert_async().await;
        });

        debug!("Test completed");
    }

    #[test]
    fn test_upload_error() {
        let mut server = Server::new();

        let mock = server
            .mock("POST", "/api/training/data")
            .with_status(400)
            .with_body("Invalid data")
            .create();

        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let ingestion = Ingestion::with_host("test_key".to_string(), server.url());

            let params = UploadSampleParams {
                device_id: "test_device",
                device_type: "CUSTOM_DEVICE",
                sensors: create_test_sensors(),
                values: create_test_values(),
                interval_ms: 100.0,
                label: None,
                category: "training",
            };

            let result = ingestion.upload_sample(params).await;

            assert!(result.is_err());
            match result {
                Err(IngestionError::Server {
                    status_code,
                    message,
                }) => {
                    assert_eq!(status_code, 400);
                    assert_eq!(message, "Invalid data");
                }
                _ => panic!("Expected Server error"),
            }
        });

        mock.assert();
    }

    #[test]
    fn test_invalid_category() {
        let server = Server::new();
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let ingestion = Ingestion::with_host("test_key".to_string(), server.url());

            let params = UploadSampleParams {
                device_id: "test_device",
                device_type: "CUSTOM_DEVICE",
                sensors: create_test_sensors(),
                values: create_test_values(),
                interval_ms: 100.0,
                label: None,
                category: "invalid_category",
            };

            let result = ingestion.upload_sample(params).await;

            assert!(result.is_err());
        });
    }
}
