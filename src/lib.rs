#![deny(clippy::correctness)]
#![deny(clippy::perf)]
#![deny(clippy::complexity)]
#![warn(clippy::style)]
#![allow(clippy::cast_precision_loss)]
#![feature(portable_simd)]
use pyo3::prelude::*;
// Declare the modules
pub mod core;
pub mod engine;

// Bring the structs into scope
use crate::engine::HadronisEngine;

#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HadronisEngine>()?;
    Ok(())
}
