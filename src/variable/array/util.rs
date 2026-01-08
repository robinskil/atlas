use crate::variable::array::error::ArrayValidationError;

pub fn num_elements(shape: &[usize]) -> Result<usize, ArrayValidationError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ArrayValidationError::ShapeOverflow)
}
