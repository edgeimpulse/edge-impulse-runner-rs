use crate::ffi::bindings;
use crate::ffi::error::{EdgeImpulseError, EdgeImpulseResult};

/// Signal structure for holding audio or sensor data
pub struct Signal {
    c_signal: Box<bindings::ei_signal_t>,
}

impl Signal {
    /// Create a new signal from raw data (f32 slice) using the SDK's signal_from_buffer
    pub fn from_raw_data(data: &[f32]) -> EdgeImpulseResult<Self> {
        let mut c_signal = Box::new(bindings::ei_signal_t {
            get_data: [0u64; 4], // Initialize with zeros for std::function
            total_length: 0,
        });

        let result =
            unsafe { bindings::ei_ffi_signal_from_buffer(data.as_ptr(), data.len(), &mut *c_signal) };

        if result == bindings::EI_IMPULSE_ERROR::EI_IMPULSE_OK {
            Ok(Self { c_signal })
        } else {
            Err(EdgeImpulseError::from(result))
        }
    }

    pub fn as_ptr(&self) -> *mut bindings::ei_signal_t {
        Box::as_ref(&self.c_signal) as *const bindings::ei_signal_t as *mut bindings::ei_signal_t
    }
}