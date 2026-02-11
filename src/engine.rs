use crate::core::connectivity::build_batched_neighbors;
use crate::core::model::GNNModel;
use memmap2::Mmap;
use nalgebra::DMatrix;
use numpy::ndarray::ArrayView2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;

type PyNeighborResult = PyResult<(Vec<usize>, Vec<usize>, Vec<[f32; 3]>)>;

/// Python wrapper for GNNModel
#[pyclass]
pub struct HadronisEngine {
    inner: GNNModel,
    // Keep mmap alive if loaded from file for zero-copy
    #[allow(dead_code)]
    _mmap: Option<Mmap>,
}

#[pymethods]
impl HadronisEngine {
    /// Load weights from safetensors file using mmap (zero-copy)
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        // Open and mmap
        let file = File::open(path)
            .map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Deserialize safetensors
        let safetensors =
            SafeTensors::deserialize(&mmap).map_err(|e: safetensors::SafeTensorError| {
                pyo3::exceptions::PyIOError::new_err(e.to_string())
            })?;

        let tensor = safetensors
            .tensor("weights")
            .map_err(|_| pyo3::exceptions::PyKeyError::new_err("No tensor 'weights' found"))?;

        if tensor.shape().len() != 2 || tensor.shape()[0] != tensor.shape()[1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weights must be square [feat_dim, feat_dim]",
            ));
        }

        let weights = DMatrix::from_row_slice(
            tensor.shape()[0],
            tensor.shape()[1],
            bytemuck::cast_slice(tensor.data()),
        );

        Ok(HadronisEngine {
            inner: GNNModel { weights },
            _mmap: Some(mmap), // keep mmap alive
        })
    }

    /// Run batched model
    /// Build batched neighbor lists for molecules
    #[staticmethod]
    pub fn build_batched_neighbors(
        py_positions: PyReadonlyArray2<f32>,
        mol_ptrs: Vec<i32>,
        cutoff: f32,
        k: usize,
    ) -> PyNeighborResult {
        let positions = py_positions.as_array();
        build_batched_neighbors(&positions, &mol_ptrs, cutoff, k)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_batched(
        &self,
        py_atomic_numbers: Vec<i32>,
        py_positions: PyReadonlyArray2<f32>,   // [n_atoms,3]
        py_features: PyReadonlyArray2<f32>,    // [n_atoms, feat_dim]
        py_edge_relpos: PyReadonlyArray2<f32>, // [n_edges,3]
        edge_src: Vec<usize>,
        edge_dst: Vec<usize>,
        mol_ptrs: Vec<i32>,
        cutoff: f32,
        num_rbf: usize,
    ) -> PyResult<Py<PyArray2<f32>>> {
        // Zero-copy views
        let positions: ArrayView2<f32> = py_positions.as_array();
        let features: ArrayView2<f32> = py_features.as_array();
        let edge_relpos: ArrayView2<f32> = py_edge_relpos.as_array();
        let edge_relpos_owned = edge_relpos.to_owned();

        // Run Rust kernel
        let result = self.inner.run_batched(
            &py_atomic_numbers,
            &positions,
            &features,
            &edge_src,
            &edge_dst,
            &edge_relpos_owned,
            &mol_ptrs,
            cutoff,
            num_rbf,
        );

        // Convert back to PyArray2 and return as Py<PyArray2<f32>>
        Python::attach(|py| Ok(PyArray2::from_array(py, &result).into()))
    }
}
