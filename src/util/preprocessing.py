import numpy as np
import scipy.signal

def to_fixed_length(data, series_length, axis=0):
    """
    Converts list of arbitrary length np arrays to 2D array of fixed length.

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

def create_data_tensor(data, series_length=57):
    """
    Converts data dictionary created by io.load_data into a 3D tensor.

    Args:
        data: Dictionary with signs as keys and values as list of sign 
            instances.
        series_length: desired length for all signals in the data
            default: 57

    Returns:
        X: Tensor of data, where axis 0 corresponds to each data point,
            axis 1 corresponds to features (signals), and axis 2 corresponds
            to signal data across time.
        y: Array of length(num data points), each element corresponding to 
            a class
        class_names: dictionary where each key is a class (0, 1, 2...) and each value
            is the class label
    """
    class_names = {}
    class_labels = {}
    num_classes = len(data)
    num_features = data.itervalues().next()[0].shape[1]

    num_samples = 0
    for i, sign in enumerate(data.keys()):
        class_labels[sign] = i
        class_names[i] = sign
        num_samples += len(data[sign])

    X = np.zeros((num_samples, num_features, series_length))
    y = np.zeros((num_samples))

    sample_idx = 0
    for i, sign in enumerate(data.keys()):
        fixed_length_data = to_fixed_length(data[sign], series_length)
        for d in fixed_length_data:
            X[sample_idx, :, :] = d.T
            y[sample_idx] = class_labels[sign]
            sample_idx += 1

    return X, y, class_names

