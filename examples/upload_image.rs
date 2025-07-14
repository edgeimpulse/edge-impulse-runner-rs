use clap::Parser;
use edge_impulse_runner::ingestion::{Category, Ingestion, UploadOptions};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Edge Impulse API key
    #[arg(short, long)]
    api_key: String,

    /// Path to the file
    #[arg(short, long)]
    file: PathBuf,

    /// Optional HMAC key for signing requests
    #[arg(short = 'k', long)]
    hmac_key: Option<String>,

    /// Optional label for the file
    #[arg(short, long)]
    label: Option<String>,

    /// Category (training, testing, or anomaly)
    #[arg(short, long, default_value = "training")]
    category: String,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Disallow duplicates
    #[arg(long)]
    disallow_duplicates: bool,

    /// Add date ID to filename
    #[arg(long)]
    add_date_id: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging with appropriate level
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", if args.debug { "debug" } else { "info" });
        }
    }
    env_logger::init();

    // Create ingestion client
    let mut ingestion = Ingestion::new(args.api_key);
    if let Some(hmac_key) = args.hmac_key {
        ingestion = ingestion.with_hmac(hmac_key);
    }
    if args.debug {
        ingestion = ingestion.with_debug();
    }

    // Parse category
    let category = match args.category.to_lowercase().as_str() {
        "training" => Category::Training,
        "testing" => Category::Testing,
        "anomaly" => Category::Anomaly,
        _ => return Err("Invalid category. Must be one of: training, testing, anomaly".into()),
    };

    let options = UploadOptions {
        disallow_duplicates: args.disallow_duplicates,
        add_date_id: args.add_date_id,
    };

    println!("Uploading file {:?}...", args.file);

    // Upload the file
    let result = ingestion
        .upload_file(&args.file, category, args.label, Some(options))
        .await?;

    println!("Upload successful! Response: {result}");
    Ok(())
}
