//! Video Classification Example
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
//!   cargo run --example video_classify -- --model <path_to_model> [--debug]
//!
//! You may need to set the gstreamer plugins path:
//! export GST_PLUGIN_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/gstreamer-1.0"
//! export DYLD_LIBRARY_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib:$DYLD_LIBRARY_PATH"
//! export DYLD_FRAMEWORK_PATH="/Library/Frameworks/GStreamer.framework/Versions/1.0/lib:$DYLD_FRAMEWORK_PATH"
//!
//! You may also need to install the opencv library:
//! brew install opencv
//!
//! Note: This only works on MacOS and requires a webcam connected to the system.

use clap::Parser;
use edge_impulse_runner::{
    InferenceResult,
    EimModel,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video::{self, VideoCapsBuilder, VideoInfo};
use std::error::Error;
use std::sync::{Arc, Mutex};

/// Command line parameters for the video classification example
#[derive(Parser, Debug)]
struct VideoClassifyParams {
    /// Path to the Edge Impulse model file (.eim)
    #[clap(short, long)]
    model: String,

    /// Enable debug output
    #[clap(short, long)]
    debug: bool,
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
fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    use std::{
        ffi::c_void,
        sync::mpsc::{channel, Sender},
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
            (applicationDidFinishLaunching:) => on_finish_launching as extern fn(&Object, Sel, id)
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

fn create_pipeline(params: &edge_impulse_runner::ModelParameters) -> Result<gst::Pipeline, Box<dyn Error>> {
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

    // Preview branch with optimized queue settings
    let queue_preview = gst::ElementFactory::make("queue").build()?;
    queue_preview.set_property_from_str("leaky", "downstream");
    queue_preview.set_property("max-size-buffers", 2u32);
    queue_preview.set_property("max-size-bytes", 0u32);
    queue_preview.set_property("max-size-time", gst::ClockTime::from_seconds(0));
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
    pipeline.add_many(&[
        &src,
        &convert,
        &scale,
        &tee,
        &queue_preview,
        &autovideosink,
        &queue_analysis,
        &appsink,
    ])?;

    // Link elements
    gst::Element::link_many(&[&src, &convert, &scale, &tee])?;
    gst::Element::link_many(&[&queue_preview, &autovideosink])?;
    gst::Element::link_many(&[&queue_analysis, &appsink])?;

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

fn process_sample(
    buffer: &gst::buffer::BufferRef,
    info: &gstreamer_video::VideoInfo,
    params: &edge_impulse_runner::ModelParameters
) -> Vec<f32> {
    let map = buffer.map_readable().unwrap();
    let data = map.as_slice();

    let width = info.width() as usize;
    let height = info.height() as usize;
    let stride = info.stride()[0] as usize;
    let channels = params.image_channel_count as usize;
    let expected_features = params.input_features_count as usize;

    println!("Model expects {} features with {} channels", expected_features, channels);
    println!("Image is {}x{} with stride {}", width, height, stride);

    // Convert to features based on model requirements
    let mut features = Vec::with_capacity(expected_features);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let offset = y * stride + x * channels + c;
                let value = data[offset] as f32 / 255.0;
                features.push(value);
            }
        }
    }

    // Verify size and truncate if necessary
    if features.len() != expected_features {
        println!("WARNING: Got {} features, expected {}. Truncating.", features.len(), expected_features);
        features.truncate(expected_features);
    }

    assert_eq!(features.len(), expected_features, "Feature count mismatch");
    features
}

fn example_main() -> Result<(), Box<dyn Error>> {
    // Initialize GStreamer first
    gst::init()?;

    // Parse command line arguments
    let params = VideoClassifyParams::parse();

    // Initialize Edge Impulse model first
    let model_instance = EimModel::new_with_debug(&params.model, params.debug)?;

    // Extract just the values we need for the pipeline
    let input_width = model_instance.parameters()?.image_input_width;
    let input_height = model_instance.parameters()?.image_input_height;
    let channel_count = model_instance.parameters()?.image_channel_count;
    let features_count = model_instance.parameters()?.input_features_count;

    // Now wrap the model in Arc<Mutex>
    let model = Arc::new(Mutex::new(model_instance));

    println!(
        "Model expects {}x{} input with {} channels ({} features)",
        input_width, input_height, channel_count, features_count
    );

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

    // Process frames
    sink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or_else(|| {
                    gst::FlowError::Error
                })?;

                let caps = sample.caps().ok_or_else(|| gst::FlowError::Error)?;
                let info = VideoInfo::from_caps(caps).map_err(|_| gst::FlowError::Error)?;

                let features = process_sample(buffer, &info, &edge_impulse_runner::ModelParameters {
                    image_input_width: input_width,
                    image_input_height: input_height,
                    image_channel_count: channel_count,
                    input_features_count: features_count,
                    ..Default::default()
                });

                // Run inference
                if let Ok(mut model) = model.lock() {
                    match model.classify(features, Some(debug)) {
                        Ok(result) => {
                            // Access classification through the InferenceResult enum
                            match result.result {
                                InferenceResult::Classification { classification } => {
                                    println!("Classification: {:?}", classification);
                                },
                                InferenceResult::ObjectDetection { bounding_boxes, classification } => {
                                    println!("Detected objects: {:?}", bounding_boxes);
                                    if !classification.is_empty() {
                                        println!("Classification: {:?}", classification);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Inference error: {}", e);
                        }
                    }
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build()
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

    Ok(())
}

fn main() {
    run(|| example_main().unwrap());
}
