//! Python bindings for `pgtoken-core`.
//!
//! Ids cross the boundary as little-endian `u32` bytes, so the Python wrapper can hand back a
//! numpy array with `np.frombuffer` and never builds a Python int per token.

use pgtoken_core::header::Codec;
use pgtoken_core::RankTable;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A trained `freq` ranking, loaded from the `<id>.tntt` file the extension writes.
#[pyclass(frozen)]
struct Table(RankTable);

#[pymethods]
impl Table {
    #[staticmethod]
    fn from_bytes(buf: &[u8]) -> PyResult<Self> {
        RankTable::from_bytes(buf)
            .map(Table)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn k(&self) -> u32 {
        self.0.k()
    }
}

fn codec_of(name: &str) -> PyResult<Codec> {
    Ok(match name {
        "raw8" => Codec::Raw8,
        "raw16" => Codec::Raw16,
        "raw24" => Codec::Raw24,
        "freq" => Codec::Freq,
        _ => return Err(PyValueError::new_err(format!("unknown codec {name:?}"))),
    })
}

fn ids_as_bytes(ids: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and any byte pattern is a valid u8. The wire format is
    // little-endian, which is what this cast yields on the little-endian targets we build for.
    unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), std::mem::size_of_val(ids)) }
}

/// Decode a stored value into little-endian u32 id bytes.
#[pyfunction]
#[pyo3(signature = (value, table=None))]
fn decode<'py>(
    py: Python<'py>,
    value: &[u8],
    table: Option<&Table>,
) -> PyResult<Bound<'py, PyBytes>> {
    let ids = pgtoken_core::decode(value, table.map(|t| &t.0))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, ids_as_bytes(&ids)))
}

/// Encode a contiguous u32 buffer (e.g. a numpy uint32 array) into a stored value.
#[pyfunction]
#[pyo3(signature = (ids, codec, vocabulary_id, table=None))]
fn encode<'py>(
    py: Python<'py>,
    ids: PyBuffer<u32>,
    codec: &str,
    vocabulary_id: u16,
    table: Option<&Table>,
) -> PyResult<Bound<'py, PyBytes>> {
    let ids = ids.to_vec(py)?;
    let out = pgtoken_core::encode(&ids, codec_of(codec)?, vocabulary_id, table.map(|t| &t.0))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Table>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    Ok(())
}
