import numpy as np
import scipy.signal

def flatten_matrix(X): 
    """
    Flattens a tensor matrix X to two dimensions. 

    Args:
        X: matrix with dimensions (x,y,z)

    Returns: 
        flattened matrix of (x,y*z)
    """
    return X.swapaxes(1,2).reshape((X.shape[0], X.shape[1]*X.shape[2]))

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
    y = np.zeros((num_samples), dtype=np.uint)

    sample_idx = 0
    for i, sign in enumerate(data.keys()):
        fixed_length_data = to_fixed_length(data[sign], series_length)
        for d in fixed_length_data:
            X[sample_idx, :, :] = d.T
            y[sample_idx] = class_labels[sign]
            sample_idx += 1

    return X, y, class_names

# def get_ablated_metrix(X, feature_keyword): 
#     'pos'

def create_train_test_split(X, y, test_size=0.3, shuffle=False, debug=False):
    """
    Splits data into a training and a testing set.

    Args:
        X: data tensor created by create_data_tensor
        y: labels array created by create_data_tensor
        test_size: size of test set. default: 30%
        shuffle: boolean defining if the sets should be shuffled
            before being split

    Returns:
        train_X:
            data tensor containing training samples
        train_y:
            array containing class labels for training samples
        test_X:
            data tensor containing testing samples
        test_y:
            array containing class labels for testing samples
    """
    class_labels = np.unique(y)
    train_samples_idx = []
    test_samples_idx = []

    # Shuffle the data
    if shuffle:
        if debug: print 'Shuffling data...',
        shuffled_idx = np.arange(y.size)
        np.random.shuffle(shuffled_idx)
        X = X[shuffled_idx, :, :]
        y = y[shuffled_idx]

        if debug: print 'done'
    
    # Collect all the indices for the training and testing set
    if debug: print 'Extracting sample indices...',
    for label in class_labels:
        idx = np.where(y==label)[0]
        train_samples = int(len(idx) * (1-test_size))
        test_samples = len(idx) - train_samples

        train_samples_idx.append(idx[:train_samples])
        test_samples_idx.append(idx[train_samples:])
    if debug: print 'done'

    # Concatenate all indices arrays into a single array of indices
    if debug: print 'Collecting indices...',
    train_samples_idx = np.concatenate(tuple(train_samples_idx))
    test_samples_idx = np.concatenate(tuple(test_samples_idx))
    if debug: print 'done'

    if shuffle:
        if debug: print 'Shuffling collecting indices...',
        np.random.shuffle(train_samples_idx)
        np.random.shuffle(test_samples_idx)
        if debug: print 'done'

    # Index and return the correct sets
    if debug: print 'Returning sets...',
    return X[train_samples_idx, :, :], y[train_samples_idx], X[test_samples_idx, :, :], y[test_samples_idx]
        
def test():
    print 'Running tests for create_train_test_split'
    X = np.random.rand(10,40,60)
    y = np.array([0,1,2,3,1,4,4,2,3,4], dtype=np.uint)
    tests = []
    for i in xrange(0,5):
        idx = np.where(y == i)[0]
        test = np.sort(X[idx, :, :], axis=0)
        tests.append(test)

    X_train, y_train, X_test, y_test = create_train_test_split(X, y, test_size=0.5, shuffle=True)
    for i in xrange(0,5):
        test2 = np.sort(np.concatenate((X_train[np.where(y_train == i)[0], :, :], X_test[np.where(y_test == i)[0], :, :])), axis=0)
        assert(np.sum(tests[i] == test2) == np.size(tests[i]))

if __name__ == '__main__':
    test()