#[inline(always)]
pub(crate) fn find_range_index(
    ranges: &[std::ops::RangeInclusive<usize>],
    value: usize,
) -> Option<usize> {
    let idx = ranges.partition_point(|r| *r.end() < value);
    ranges.get(idx).filter(|r| r.contains(&value))?;
    Some(idx)
}
