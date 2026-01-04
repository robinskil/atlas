use ndarray::{ArrayView, IxDyn};

use crate::dtype::DataType;

#[derive(Debug)]
pub struct Array<'s, T> {
    data: bytes::Bytes,
    shape: &'s [usize],
    dtype: DataType,
    _marker: std::marker::PhantomData<T>,
}

impl<'s, T> Array<'s, T> {
    pub fn new(data: bytes::Bytes, shape: &'s [usize], dtype: DataType) -> Self {
        Self {
            data,
            shape,
            dtype,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_nd_array<'l>(&'l self) -> ArrayView<'l, T, IxDyn> {
        let len: usize = self.shape.iter().product();
        let ptr = self.data.as_ptr() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        ArrayView::from_shape(IxDyn(self.shape), slice).unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}

pub struct ArrayMut<'s, T> {
    data: bytes::BytesMut,
    shape: &'s [usize],
    dtype: DataType,
    _marker: std::marker::PhantomData<T>,
}

impl<'s, T> ArrayMut<'s, T> {
    pub fn new(data: bytes::BytesMut, shape: &'s [usize], dtype: DataType) -> Self {
        Self {
            data,
            shape,
            dtype,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn as_nd_array_mut<'l>(&'l mut self) -> ndarray::ArrayViewMut<'l, T, IxDyn> {
        let len: usize = self.shape.iter().product();
        let ptr = self.data.as_mut_ptr() as *mut T;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        ndarray::ArrayViewMut::from_shape(IxDyn(self.shape), slice).unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}
