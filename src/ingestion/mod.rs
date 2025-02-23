use crate::error::IngestionError;
use hmac::{Hmac, Mac};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error};
const DEFAULT_INGESTION_HOST: &str = "https://ingestion.edgeimpulse.com";

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

pub struct Ingestion {
    api_key: String,
    hmac_key: Option<String>,
    host: String,
}

impl Ingestion {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            hmac_key: None,
            host: DEFAULT_INGESTION_HOST.to_string(),
        }
    }

    pub fn with_host(api_key: String, host: String) -> Self {
        Self {
            api_key,
            hmac_key: None,
            host,
        }
    }

    pub fn with_hmac(mut self, hmac_key: String) -> Self {
        self.hmac_key = Some(hmac_key);
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
        device_id: &str,
        device_type: &str,
        sensors: Vec<Sensor>,
        values: Vec<Vec<f64>>,
        interval_ms: f64,
        label: Option<String>,
        category: &str,
    ) -> Result<String, IngestionError> {
        debug!("Creating data message");
        let data = DataMessage {
            protected: Protected {
                ver: "v1".to_string(),
                alg: "HS256".to_string(),
                iat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            signature: "0".repeat(64),
            payload: Payload {
                device_name: device_id.to_string(),
                device_type: device_type.to_string(),
                interval_ms,
                sensors,
                values,
            },
        };

        debug!("Serializing data message");
        let json_data = serde_json::to_string(&data)?;
        debug!("Creating signature for data");
        let signature = self.create_signature(json_data.as_bytes()).await?;
        debug!("Generated signature: {}", signature);

        let signed_data = DataMessage { signature, ..data };

        debug!("Creating multipart form");
        let form = Form::new().part("message", Part::text(serde_json::to_string(&signed_data)?));

        let client = reqwest::Client::new();
        let mut headers = reqwest::header::HeaderMap::new();
        debug!("Setting up headers");
        headers.insert("x-api-key", self.api_key.parse()?);
        headers.insert("x-file-name", format!("{}.json", device_id).parse()?);

        if let Some(label) = label {
            debug!("Adding label header: {}", label);
            headers.insert("x-label", urlencoding::encode(&label).parse()?);
        }

        let url = format!("{}/api/{}/data", self.host, category);
        debug!("Sending request to: {}", url);
        let response = client
            .post(url)
            .headers(headers)
            .multipart(form)
            .send()
            .await?;

        let status = response.status();
        debug!("Response status: {}", status);

        let response_text = response.text().await?;
        debug!("Response body: {}", response_text);

        if !status.is_success() {
            error!("Request failed: {}", response_text);
            return Err(IngestionError::Server {
                status_code: status.as_u16(),
                message: response_text,
            });
        }

        debug!("Request successful");
        Ok(response_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
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

            let result = ingestion
                .upload_sample(
                    "test_device",
                    "CUSTOM_DEVICE",
                    create_test_sensors(),
                    create_test_values(),
                    100.0,
                    Some("walking".to_string()),
                    "training",
                )
                .await;

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

            let result = ingestion
                .upload_sample(
                    "test_device",
                    "CUSTOM_DEVICE",
                    sensors,
                    values,
                    100.0,
                    None,
                    "training",
                )
                .await;

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

            let result = ingestion
                .upload_sample(
                    "test_device",
                    "CUSTOM_DEVICE",
                    create_test_sensors(),
                    create_test_values(),
                    100.0,
                    None,
                    "training",
                )
                .await;

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

            let result = ingestion
                .upload_sample(
                    "test_device",
                    "CUSTOM_DEVICE",
                    create_test_sensors(),
                    create_test_values(),
                    100.0,
                    None,
                    "invalid_category", // Invalid category
                )
                .await;

            assert!(result.is_err());
        });
    }
}
