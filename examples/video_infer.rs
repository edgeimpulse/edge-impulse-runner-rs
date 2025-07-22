#![allow(deprecated)]

//! Video Classification Example (macOS only)
//!
//! This example demonstrates how to use the Edge Impulse Runner to perform video classification
//! using a trained model. It captures frames from a webcam using GStreamer and runs inference.
//!
//! The program will:
//! 1. Load an Edge Impulse model
//! 2. Initialize webcam capture using GStreamer
//! 3. Process frames in real-time
//! 4. Output the classification results
//!
//! Usage:
//!   # EIM mode (requires model file)
//!   cargo run --example video_infer -- --model <path_to_model> [--debug]
//!
//!   # FFI mode (no model file needed)
//!   cargo run --example video_infer --features ffi -- [--debug]
//!
//! You may need to set the gstreamer plugins path:
//! export GST_PLUGIN_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/gstreamer-1.0"
//! export DYLD_LIBRARY_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib:$DYLD_LIBRARY_PATH"
//! export DYLD_FRAMEWORK_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib:$DYLD_FRAMEWORK_PATH"
//!
//! Note: This only works on MacOS and requires a webcam connected to the system.

use clap::Parser;
// Removed unused import
#[cfg(feature = "ffi")]
use edge_impulse_runner::ffi::ModelMetadata;
use edge_impulse_runner::types::ModelThreshold;
use edge_impulse_runner::{EdgeImpulseModel, InferenceResult};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gstreamer_video::{self, VideoCapsBuilder, VideoInfo};
// Removed unused import
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Performance tracking structure
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    frame_count: u64,
    total_inference_time: Duration,
    start_time: Instant,
    last_fps_time: Instant,
    fps_samples: Vec<f64>,
    inference_samples: Vec<Duration>,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            frame_count: 0,
            total_inference_time: Duration::ZERO,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_samples: Vec::new(),
            inference_samples: Vec::new(),
        }
    }

    fn update(&mut self, inference_time_ms: u32) {
        self.frame_count += 1;
        let inference_time = Duration::from_millis(inference_time_ms as u64);
        self.total_inference_time += inference_time;
        self.inference_samples.push(inference_time);

        // Keep only last 100 samples for rolling average
        if self.inference_samples.len() > 100 {
            self.inference_samples.remove(0);
        }

        // Calculate FPS every second
        let now = Instant::now();
        if now.duration_since(self.last_fps_time) >= Duration::from_secs(1) {
            let fps = self.frame_count as f64 / now.duration_since(self.start_time).as_secs_f64();
            self.fps_samples.push(fps);

            // Keep only last 10 FPS samples
            if self.fps_samples.len() > 10 {
                self.fps_samples.remove(0);
            }

            self.last_fps_time = now;
        }
    }

    fn print_summary(&self) {
        let total_time = self.start_time.elapsed();
        let avg_fps = if !self.fps_samples.is_empty() {
            self.fps_samples.iter().sum::<f64>() / self.fps_samples.len() as f64
        } else {
            0.0
        };

        let avg_inference_time = if !self.inference_samples.is_empty() {
            self.inference_samples.iter().sum::<Duration>() / self.inference_samples.len() as u32
        } else {
            Duration::ZERO
        };

        let min_inference_time = self
            .inference_samples
            .iter()
            .min()
            .unwrap_or(&Duration::ZERO);
        let max_inference_time = self
            .inference_samples
            .iter()
            .max()
            .unwrap_or(&Duration::ZERO);

        println!("\n📊 PERFORMANCE SUMMARY:");
        println!("   Total frames processed: {}", self.frame_count);
        println!("   Total runtime: {:.2}s", total_time.as_secs_f64());
        println!("   Average FPS: {avg_fps:.2}");
        println!(
            "   Average inference time: {:.2}ms",
            avg_inference_time.as_millis()
        );
        println!(
            "   Min inference time: {:.2}ms",
            min_inference_time.as_millis()
        );
        println!(
            "   Max inference time: {:.2}ms",
            max_inference_time.as_millis()
        );
        println!(
            "   Total inference time: {:.2}s",
            self.total_inference_time.as_secs_f64()
        );
        println!(
            "   Inference efficiency: {:.1}%",
            (self.total_inference_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        );
    }
}

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
struct VideoInferParams {
    /// Path to the Edge Impulse model file (.eim) (not needed for FFI mode)
    #[clap(short, long)]
    model: Option<String>,

    /// Enable debug output
    #[clap(short, long)]
    debug: bool,

    /// Override model thresholds. E.g. --thresholds 4.min_anomaly_score=35 overrides the min. anomaly score for block ID 4 to 35.
    #[clap(long)]
    thresholds: Option<String>,

    /// Set object detection threshold (0.0 to 1.0)
    #[clap(long)]
    threshold: Option<f32>,



    /// Enable performance monitoring
    #[clap(long)]
    perf: bool,
}

// macOS specific run loop handling
#[cfg(not(target_os = "macos"))]
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    main()
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)] // Suppress the warnings from objc macros
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    use std::{
        ffi::c_void,
        sync::mpsc::{Sender, channel},
        thread,
    };

    use cocoa::{
        appkit::{NSApplication, NSWindow},
        base::id,
        delegate,
    };
    use objc::{
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    unsafe {
        let app = cocoa::appkit::NSApp();
        let (send, recv) = channel::<()>();

        extern "C" fn on_finish_launching(this: &Object, _cmd: Sel, _notification: id) {
            let send = unsafe {
                let send_pointer = *this.get_ivar::<*const c_void>("send");
                let boxed = Box::from_raw(send_pointer as *mut Sender<()>);
                *boxed
            };
            send.send(()).unwrap();
        }

        let delegate = delegate!("AppDelegate", {
            app: id = app,
            send: *const c_void = Box::into_raw(Box::new(send)) as *const c_void,
            (applicationDidFinishLaunching:) => on_finish_launching as extern "C" fn(&Object, Sel, id)
        });
        app.setDelegate_(delegate);

        let t = thread::spawn(move || {
            // Wait for the NSApp to launch to avoid possibly calling stop_() too early
            recv.recv().unwrap();

            let res = main();

            let app = cocoa::appkit::NSApp();
            app.stop_(cocoa::base::nil);

            // Stopping the event loop requires an actual event
            let event = cocoa::appkit::NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                cocoa::base::nil,
                cocoa::appkit::NSEventType::NSApplicationDefined,
                cocoa::foundation::NSPoint { x: 0.0, y: 0.0 },
                cocoa::appkit::NSEventModifierFlags::empty(),
                0.0,
                0,
                cocoa::base::nil,
                cocoa::appkit::NSEventSubtype::NSApplicationActivatedEventType,
                0,
                0,
            );
            app.postEvent_atStart_(event, cocoa::base::YES);

            res
        });

        app.run();

        t.join().unwrap()
    }
}

fn create_pipeline(
    params: &edge_impulse_runner::ModelParameters,
) -> Result<gst::Pipeline, Box<dyn Error>> {
    println!(
        "Setting up pipeline for {}x{} input with {} channels",
        params.image_input_width, params.image_input_height, params.image_channel_count
    );

    // Create pipeline elements
    let pipeline = gst::Pipeline::new();
    let src = gst::ElementFactory::make("avfvideosrc")
        .property("device-index", 0i32)
        .build()?;
    let convert = gst::ElementFactory::make("videoconvert").build()?;
    let scale = gst::ElementFactory::make("videoscale").build()?;
    let tee = gst::ElementFactory::make("tee").build()?;

    // Preview branch with optimized queue settings and video scaling
    let queue_preview = gst::ElementFactory::make("queue").build()?;
    queue_preview.set_property_from_str("leaky", "downstream");
    queue_preview.set_property("max-size-buffers", 2u32);
    queue_preview.set_property("max-size-bytes", 0u32);
    queue_preview.set_property("max-size-time", gst::ClockTime::from_seconds(0));

    // Add videoscale element for preview
    let scale_preview = gst::ElementFactory::make("videoscale").build()?;

    // Add capsfilter to set the preview size
    let preview_caps = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("video/x-raw")
                .field("width", 640i32) // Set preview width
                .field("height", 480i32) // Set preview height
                .build(),
        )
        .build()?;

    let autovideosink = gst::ElementFactory::make("autovideosink").build()?;

    // Analysis branch with optimized queue settings
    let queue_analysis = gst::ElementFactory::make("queue").build()?;
    queue_analysis.set_property_from_str("leaky", "downstream");
    queue_analysis.set_property("max-size-buffers", 1u32);
    queue_analysis.set_property("max-size-bytes", 0u32);
    queue_analysis.set_property("max-size-time", gst::ClockTime::from_seconds(0));
    let appsink = gst::ElementFactory::make("appsink")
        .property("emit-signals", true)
        .build()?;

    // Add elements to pipeline
    pipeline.add_many([
        &src,
        &convert,
        &scale,
        &tee,
        &queue_preview,
        &scale_preview,
        &preview_caps,
        &autovideosink,
        &queue_analysis,
        &appsink,
    ])?;

    // Link elements
    gst::Element::link_many([&src, &convert, &scale, &tee])?;
    gst::Element::link_many([
        &queue_preview,
        &scale_preview,
        &preview_caps,
        &autovideosink,
    ])?;
    gst::Element::link_many([&queue_analysis, &appsink])?;

    // Link tee to queues
    let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
    let tee_preview_pad = tee.request_pad(&tee_src_pad_template, None, None).unwrap();
    let queue_preview_sink_pad = queue_preview.static_pad("sink").unwrap();
    tee_preview_pad.link(&queue_preview_sink_pad)?;

    let tee_analysis_pad = tee.request_pad(&tee_src_pad_template, None, None).unwrap();
    let queue_analysis_sink_pad = queue_analysis.static_pad("sink").unwrap();
    tee_analysis_pad.link(&queue_analysis_sink_pad)?;

    // Configure appsink with correct format and size
    let appsink = appsink.dynamic_cast::<gst_app::AppSink>().unwrap();
    let caps = VideoCapsBuilder::new()
        .format(if params.image_channel_count == 1 {
            gstreamer_video::VideoFormat::Gray8
        } else {
            gstreamer_video::VideoFormat::Rgb
        })
        .width(params.image_input_width as i32)
        .height(params.image_input_height as i32)
        .build();
    appsink.set_caps(Some(&caps));

    Ok(pipeline)
}

fn process_image(
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
    debug: bool,
) -> Vec<f32> {
    let mut features = Vec::with_capacity((width * height) as usize);

    // Always pass in full RGB array in the form of pixels like 0xff0000
    // DSP handles normalization and splitting into 1 or 3 channels
    if channels == 1 {
        // For grayscale images, create RGB values by repeating the grayscale value
        for &pixel in image_data.iter() {
            // Create 24-bit RGB value: 0xRRGGBB where R=G=B=pixel
            let feature = ((pixel as u32) << 16) | ((pixel as u32) << 8) | (pixel as u32);
            features.push(feature as f32);
        }
    } else {
        // For RGB images, combine channels into 24-bit RGB values
        for chunk in image_data.chunks(3) {
            if chunk.len() == 3 {
                // Create 24-bit RGB value: 0xRRGGBB
                let feature =
                    ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
                features.push(feature as f32);
            }
        }
    }

    if debug {
        // Calculate some statistics about the features
        let min = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;

        println!(
            "Feature statistics: min={:.3}, max={:.3}, mean={:.3}, count={}",
            min,
            max,
            mean,
            features.len()
        );
    }

    features
}

fn process_sample(
    buffer: &gst::BufferRef,
    _info: &gst_video::VideoInfo,
    model_params: &edge_impulse_runner::ModelParameters,
    debug: bool,
) -> Vec<f32> {
    let width = model_params.image_input_width;
    let height = model_params.image_input_height;

    // Map the buffer for reading
    let map = buffer.map_readable().unwrap();
    let data = map.as_slice();

    // Convert frame data to Vec<u8>
    let image_data: Vec<u8> = data.to_vec();

    // Process the image data using our image processing function
    process_image(
        image_data,
        width,
        height,
        model_params.image_channel_count,
        debug,
    )
}

/// Initialize the Edge Impulse model and extract its parameters
async fn initialize_model(
    _model_path: Option<&str>,
    debug: bool,
    thresholds: Option<&str>,
    threshold: Option<f32>,
) -> Result<
    (
        Arc<Mutex<EdgeImpulseModel>>,
        edge_impulse_runner::ModelParameters,
    ),
    Box<dyn std::error::Error>,
> {
    let mut model_instance = {
        // Auto-detect which backend to use based on available features
        #[cfg(feature = "ffi")]
        {
            // FFI mode - no model file needed (default)
            println!("Using FFI mode (default)");
            let model = if debug {
                EdgeImpulseModel::new_with_debug(true)?
            } else {
                EdgeImpulseModel::new()?
            };

            // Print model metadata for FFI mode
            let metadata = ModelMetadata::get();
            println!("\nModel Metadata:");
            println!("===============");
            println!("Project ID: {}", metadata.project_id);
            println!("Project Owner: {}", metadata.project_owner);
            println!("Project Name: {}", metadata.project_name);
            println!("Deploy Version: {}", metadata.deploy_version);
            println!(
                "Model Type: {}",
                if metadata.has_object_detection {
                    "Object Detection"
                } else {
                    "Classification"
                }
            );
            println!(
                "Input Dimensions: {}x{}x{}",
                metadata.input_width, metadata.input_height, metadata.input_frames
            );
            println!("Input Features: {}", metadata.input_features_count);
            println!("Label Count: {}", metadata.label_count);
            println!(
                "Sensor Type: {}",
                match metadata.sensor {
                    1 => "Microphone",
                    2 => "Accelerometer",
                    3 => "Camera",
                    4 => "Positional",
                    _ => "Unknown",
                }
            );
            println!(
                "Inferencing Engine: {}",
                match metadata.inferencing_engine {
                    1 => "uTensor",
                    2 => "TensorFlow Lite",
                    3 => "CubeAI",
                    4 => "TensorFlow Lite Full",
                    _ => "Other",
                }
            );
            println!("Has Anomaly Detection: {}", metadata.has_anomaly);
            println!("Has Object Tracking: {}", metadata.has_object_tracking);
            println!("===============\n");

            model
        }
        #[cfg(all(feature = "eim", not(feature = "ffi")))]
        {
            // Only EIM feature available, use EIM mode
            println!("Using EIM mode (FFI not available)");
            let model_path = _model_path.ok_or("Model path is required for EIM mode")?;
            if debug {
                EdgeImpulseModel::new_eim_with_debug(model_path, true)?
            } else {
                EdgeImpulseModel::new_eim(model_path)?
            }
        }
        #[cfg(not(any(feature = "ffi", feature = "eim")))]
        {
            return Err("No backend available. Enable either 'ffi' or 'eim' feature.".into());
        }
    };

    // Set thresholds if provided (not supported in current backend abstraction)
    if let Some(thresholds_str) = thresholds {
        println!(
            "Threshold setting not supported in current backend abstraction: {thresholds_str}"
        );
    }

    // Apply object detection threshold if provided
    if let Some(threshold_value) = threshold {
        println!("Setting object detection threshold to {}", threshold_value);

        // Get model parameters to find object detection thresholds
        let model_params_temp = model_instance.parameters()?;

        // Collect object detection thresholds to avoid borrowing issues
        let object_detection_thresholds: Vec<_> = model_params_temp
            .thresholds
            .iter()
            .filter_map(|t| {
                if let edge_impulse_runner::types::ModelThreshold::ObjectDetection { id, .. } = t {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        // Apply the new threshold to each object detection block
        for block_id in object_detection_thresholds {
            let new_threshold = edge_impulse_runner::types::ModelThreshold::ObjectDetection {
                id: block_id,
                min_score: threshold_value,
            };

            match model_instance.set_threshold(new_threshold) {
                Ok(()) => println!(
                    "Successfully set object detection threshold for block ID {} to {}",
                    block_id, threshold_value
                ),
                Err(e) => println!("Failed to set threshold for block ID {}: {}", block_id, e),
            }
        }
    }

    // Get model parameters
    let model_params = model_instance.parameters()?.clone();

    // Print detailed model parameters
    println!("\nModel Parameters:");
    println!("----------------");
    println!("{model_params:#?}");
    println!("----------------\n");

    // Wrap the model in Arc<Mutex>
    let model = Arc::new(Mutex::new(model_instance));

    Ok((model, model_params))
}

#[tokio::main]
async fn example_main() -> Result<(), Box<dyn Error>> {
    // Initialize GStreamer first
    gst::init()?;

    // Parse command line arguments
    let params = VideoInferParams::parse();

    // Initialize performance metrics
    let perf_metrics = Arc::new(Mutex::new(PerformanceMetrics::new()));

    // Initialize Edge Impulse model and get parameters
    let (model, model_params) = initialize_model(
        params.model.as_deref(),
        params.debug,
        params.thresholds.as_deref(),
        params.threshold,

    )
    .await?;

    let input_width = model_params.image_input_width;
    let input_height = model_params.image_input_height;
    let channel_count = model_params.image_channel_count;
    let features_count = model_params.input_features_count;

    println!(
        "Model expects {input_width}x{input_height} input with {channel_count} channels ({features_count} features)"
    );

    // Add the image format details here
    println!("Image format will be {input_width}x{input_height} with {channel_count} channels");

    // Create pipeline with the extracted parameters
    let pipeline = create_pipeline(&edge_impulse_runner::ModelParameters {
        image_input_width: input_width,
        image_input_height: input_height,
        image_channel_count: channel_count,
        input_features_count: features_count,
        ..Default::default() // Fill in other fields with defaults
    })?;

    let sink = pipeline
        .by_name("appsink0")
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    // Clone Arcs for the closure
    let model = Arc::clone(&model);
    let debug = params.debug;
    let perf_metrics_clone = Arc::clone(&perf_metrics);

    // Add timestamp tracking before the sink callbacks
    let last_no_detection = Arc::new(Mutex::new(Instant::now()));
    let last_no_detection_clone = Arc::clone(&last_no_detection);

    // Process frames
    sink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;

                let caps = sample.caps().ok_or(gst::FlowError::Error)?;
                let info = VideoInfo::from_caps(caps).map_err(|_| gst::FlowError::Error)?;

                let features = process_sample(
                    buffer,
                    &info,
                    &edge_impulse_runner::ModelParameters {
                        image_input_width: input_width,
                        image_input_height: input_height,
                        image_channel_count: channel_count,
                        input_features_count: features_count,
                        ..Default::default()
                    },
                    debug,
                );

                // Debug: Print feature count mismatch
                if features.len() != features_count as usize {
                    eprintln!("Feature count mismatch: got {} features, expected {}", features.len(), features_count);
                }

                // Run inference with timing
                if let Ok(mut model) = model.lock() {
                    let inference_start = Instant::now();
                    match model.infer(features, None) {
                        Ok(response) => {
                            let inference_time = inference_start.elapsed();
                            let inference_time_ms = inference_time.as_millis() as u32;

                            // Update performance metrics
                            if let Ok(mut metrics) = perf_metrics_clone.lock() {
                                metrics.update(inference_time_ms);

                                // Print real-time performance info if enabled
                                if params.perf {
                                    let current_fps = if !metrics.fps_samples.is_empty() {
                                        metrics.fps_samples.last().unwrap()
                                    } else {
                                        &0.0
                                    };
                                    let avg_inference = if !metrics.inference_samples.is_empty() {
                                        metrics.inference_samples.iter()
                                            .rev()
                                            .take(10)
                                            .sum::<Duration>() / metrics.inference_samples.iter().rev().take(10).count() as u32
                                    } else {
                                        Duration::ZERO
                                    };

                                    println!("🎯 Frame #{:3} | FPS: {:4.1} | Inference: {:3}ms | Avg: {:4.1}ms",
                                        metrics.frame_count,
                                        current_fps,
                                        inference_time_ms,
                                        avg_inference.as_millis()
                                    );
                                }
                            }

                            match response.result {
                                InferenceResult::Classification { classification } => {
                                    if !classification.is_empty() {
                                        println!("Classification: {classification:?}");
                                    }
                                }
                                InferenceResult::ObjectDetection {
                                    bounding_boxes,
                                    classification,
                                } => {
                                    if !bounding_boxes.is_empty() {
                                        println!("Detected objects: {bounding_boxes:?}");
                                        if let Ok(mut last) = last_no_detection_clone.lock() {
                                            *last = Instant::now();
                                        }
                                    } else {
                                        // Check if 5 seconds have passed since last notification
                                        if let Ok(mut last) = last_no_detection_clone.lock()
                                            && last.elapsed() >= Duration::from_secs(5) {
                                                println!("No objects detected");
                                                *last = Instant::now();
                                            }
                                    }

                                    if !classification.is_empty() {
                                        println!("Classification: {classification:?}");
                                    }
                                }
                                InferenceResult::VisualAnomaly {
                                    visual_anomaly_grid,
                                    visual_anomaly_max,
                                    visual_anomaly_mean,
                                    anomaly,
                                } => {
                                    // Print raw values for debugging
                                    println!("\nRaw anomaly values:");
                                    println!("  Overall: {anomaly:.2}");
                                    println!("  Maximum: {visual_anomaly_max:.2}");
                                    println!("  Mean: {visual_anomaly_mean:.2}");

                                    // Debug output for the grid
                                    if debug {
                                        println!("\nVisual anomaly grid:");
                                        println!("  Number of regions: {}", visual_anomaly_grid.len());
                                        for (i, bbox) in visual_anomaly_grid.iter().enumerate() {
                                            println!("  Region {}: value={:.2}, x={}, y={}, w={}, h={}",
                                                i, bbox.value, bbox.x, bbox.y, bbox.width, bbox.height);
                                        }
                                    }

                                    println!("\nThreshold information:");
                                    let min_anomaly_score = model_params.thresholds.iter()
                                        .find_map(|t| match t {
                                            ModelThreshold::AnomalyGMM { min_anomaly_score, .. } => Some(*min_anomaly_score),
                                            _ => None,
                                        })
                                        .unwrap_or(6.0);
                                    println!("  min_anomaly_score: {min_anomaly_score}");

                                    // Normalize all scores using the model's normalization method
                                    let (normalized_anomaly, normalized_max, normalized_mean, normalized_regions) =
                                        model.normalize_visual_anomaly(
                                            anomaly,
                                            visual_anomaly_max,
                                            visual_anomaly_mean,
                                            &visual_anomaly_grid.iter()
                                                .map(|bbox| (bbox.value, bbox.x as u32, bbox.y as u32, bbox.width as u32, bbox.height as u32))
                                                .collect::<Vec<_>>()
                                        );

                                    println!("\nNormalized scores:");
                                    println!("  Overall score: {:.2}%", normalized_anomaly * 100.0);
                                    println!("  Maximum score: {:.2}%", normalized_max * 100.0);
                                    println!("  Mean score: {:.2}%", normalized_mean * 100.0);

                                    // Show all detected regions (if any exist)
                                    if !normalized_regions.is_empty() {
                                        println!("\nDetected regions:");
                                        for (value, x, y, w, h) in normalized_regions {
                                            println!(
                                                "  Region (normalized: {:.2}%): x={}, y={}, width={}, height={}",
                                                value * 100.0,
                                                x,
                                                y,
                                                w,
                                                h
                                            );
                                        }
                                    } else {
                                        println!("\nNo regions detected");
                                        if debug {
                                            println!("  Note: This could be because:");
                                            println!("  1. The model didn't detect any anomalies above the threshold");
                                            println!("  2. The visual_anomaly_grid is empty");
                                            println!("  3. The normalization process filtered out all regions");
                                            println!("  4. The min_anomaly_score threshold ({min_anomaly_score}) is too high");
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Inference error: {e}");
                        }
                    }
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start playing
    pipeline.set_state(gst::State::Playing)?;
    println!("Playing... (Ctrl+C to stop)");

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                println!(
                    "Error from {:?}: {} ({:?})",
                    err.src().map(|s| s.path_string()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            _ => (),
        }
    }

    // Cleanup
    pipeline.set_state(gst::State::Null)?;
    println!("Pipeline stopped");

    // Print performance summary
    if let Ok(metrics) = perf_metrics.lock() {
        metrics.print_summary();
    }

    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
