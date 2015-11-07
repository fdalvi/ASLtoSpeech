import scipy.signal

def to_fixed_length(data, series_length, axis=0):
    """Converts list of arbitrary length np arrays to 2D array of fixed length.

    Args:
        data: list of 2D array-like objects of arbitrary length
        series_length: desired length for all arrays in data
        axis: axis across which the signal should be stretched/shrink the 
            signal. default: 0 (row-wise)

    Returns:
        A len(data) array with 2D array-like objects of the specified length.
    """
    assert len(data) > 0

    fixed_length_data = []
    for series in data:
        fixed_length_series = scipy.signal.resample(series, series_length, axis=axis)
        fixed_length_data.append(fixed_length_series)

    return fixed_length_data