use std::fmt;

/// Result structure for classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub value: f32,
}

/// Result structure for bounding boxes
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub label: String,
    pub value: f32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Result structure for timing information
#[derive(Debug, Clone)]
pub struct TimingResult {
    pub dsp: i32,
    pub classification: i32,
    pub anomaly: i32,
}

/// Result structure for Edge Impulse inference
pub struct InferenceResult {
    result: *mut crate::ffi::bindings::ei_impulse_result_t,
}

impl InferenceResult {
    /// Create a new inference result
    pub fn new() -> Self {
        let result = unsafe {
            let ptr = std::alloc::alloc_zeroed(std::alloc::Layout::new::<crate::ffi::bindings::ei_impulse_result_t>())
                as *mut crate::ffi::bindings::ei_impulse_result_t;
            ptr
        };
        Self { result }
    }

    /// Get a pointer to the result
    pub fn as_ptr(&self) -> *const crate::ffi::bindings::ei_impulse_result_t {
        self.result
    }

    /// Get a mutable pointer to the result
    pub fn as_mut_ptr(&mut self) -> *mut crate::ffi::bindings::ei_impulse_result_t {
        self.result
    }

    /// Get classification results
    pub fn classifications(&self, label_count: usize) -> Vec<ClassificationResult> {
        let mut results = Vec::new();
        unsafe {
            let result = &*self.result;
            for i in 0..label_count {
                if i < result.classification.len() {
                    let classification = &result.classification[i];
                    results.push(ClassificationResult {
                        label: std::ffi::CStr::from_ptr(classification.label)
                            .to_string_lossy()
                            .to_string(),
                        value: classification.value,
                    });
                }
            }
        }
        results
    }

    /// Get bounding box results
    pub fn bounding_boxes(&self) -> Vec<BoundingBox> {
        let mut results = Vec::new();
        unsafe {
            let result = &*self.result;
            for i in 0..result.bounding_boxes_count {
                let bbox = &*result.bounding_boxes.add(i as usize);
                results.push(BoundingBox {
                    label: std::ffi::CStr::from_ptr(bbox.label)
                        .to_string_lossy()
                        .to_string(),
                    value: bbox.value,
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height,
                });
            }
        }
        results
    }

    /// Get timing information
    pub fn timing(&self) -> TimingResult {
        unsafe {
            let result = &*self.result;
            TimingResult {
                dsp: result.timing.dsp,
                classification: result.timing.classification,
                anomaly: result.timing.anomaly,
            }
        }
    }
}

impl Drop for InferenceResult {
    fn drop(&mut self) {
        unsafe {
            if !self.result.is_null() {
                std::alloc::dealloc(
                    self.result as *mut u8,
                    std::alloc::Layout::new::<crate::ffi::bindings::ei_impulse_result_t>(),
                );
            }
        }
    }
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.4}", self.label, self.value)
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.4} (x={}, y={}, w={}, h={})",
            self.label, self.value, self.x, self.y, self.width, self.height
        )
    }
}

impl fmt::Display for TimingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Timing: dsp={} ms, classification={} ms, anomaly={} ms",
            self.dsp, self.classification, self.anomaly
        )
    }
}