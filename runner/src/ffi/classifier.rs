use crate::ffi::bindings;
use crate::ffi::error::{EdgeImpulseError, EdgeImpulseResult};
use crate::ffi::types::InferenceResult;
use crate::ffi::Signal;
use std::ptr;

/// Opaque handle for Edge Impulse operations
pub struct EdgeImpulseHandle {
    handle: *mut bindings::ei_impulse_handle_t,
}

impl EdgeImpulseHandle {
    /// Create a new Edge Impulse handle
    pub fn new() -> EdgeImpulseResult<Self> {
        let handle = ptr::null_mut();
        let result = unsafe { bindings::ei_ffi_init_impulse(handle) };
        let error = EdgeImpulseError::from(result);
        if error != EdgeImpulseError::Ok {
            return Err(error);
        }

        Ok(Self { handle })
    }
}

impl Drop for EdgeImpulseHandle {
    fn drop(&mut self) {
        // Cleanup if needed
    }
}

/// Main classifier for Edge Impulse inference
pub struct EdgeImpulseClassifier {
    initialized: bool,
}

impl EdgeImpulseClassifier {
    /// Create a new classifier instance
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Initialize the classifier
    pub fn init(&mut self) -> EdgeImpulseResult<()> {
        unsafe {
            bindings::ei_ffi_run_classifier_init();
        }
        self.initialized = true;
        Ok(())
    }

    /// Deinitialize the classifier
    pub fn deinit(&mut self) -> EdgeImpulseResult<()> {
        if self.initialized {
            unsafe {
                bindings::ei_ffi_run_classifier_deinit();
            }
            self.initialized = false;
        }
        Ok(())
    }

    /// Run classifier on a signal
    pub fn run_classifier(
        &self,
        signal: &Signal,
        debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        let mut result = InferenceResult::new();
        let debug_int = if debug { 1 } else { 0 };

        let error = unsafe {
            bindings::ei_ffi_run_classifier(signal.as_ptr(), result.as_mut_ptr(), debug_int)
        };

        if error == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(result)
        } else {
            Err(EdgeImpulseError::from(error))
        }
    }

    /// Run classifier in continuous mode
    pub fn run_classifier_continuous(
        &self,
        signal: &Signal,
        debug: bool,
        enable_maf: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        let mut result = InferenceResult::new();
        let debug_int = if debug { 1 } else { 0 };
        let maf_int = if enable_maf { 1 } else { 0 };

        let error = unsafe {
            bindings::ei_ffi_run_classifier_continuous(
                signal.as_ptr(),
                result.as_mut_ptr(),
                debug_int,
                maf_int,
            )
        };

        if error == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(result)
        } else {
            Err(EdgeImpulseError::from(error))
        }
    }

    /// Run inference with a handle and feature matrix
    pub fn run_inference(
        &self,
        handle: &mut EdgeImpulseHandle,
        fmatrix: *mut bindings::ei_feature_t,
        debug: bool,
    ) -> EdgeImpulseResult<InferenceResult> {
        let mut result = InferenceResult::new();
        let debug_int = if debug { 1 } else { 0 };

        let error = unsafe {
            bindings::ei_ffi_run_inference(handle.handle, fmatrix, result.as_mut_ptr(), debug_int)
        };

        if error == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(result)
        } else {
            Err(EdgeImpulseError::from(error))
        }
    }
}

impl Drop for EdgeImpulseClassifier {
    fn drop(&mut self) {
        let _ = self.deinit();
    }
}