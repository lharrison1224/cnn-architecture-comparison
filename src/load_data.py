import pickle
import numpy as np


def load():
    """
    Returns:
        np.array: train data, shape (50000, 32, 32, 3)
        np.array: train labels, shape (50000,)
        np.array: test data, shape (10000, 32, 32, 3)
        np.array: test labels, shape (10000,)
    """
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None

    # load train data
    for i in range(1, 6):
        with open('data/data_batch_' + str(i), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = d[b'data']
            data = np.reshape(data, (10000, 3, 32, 32))
            data = data.transpose((0, 2, 3, 1))
            labels = d[b'labels']
            if i == 1:
                train_data = data
                train_labels = labels
            else:
                train_data = np.concatenate([train_data, data])
                train_labels = np.concatenate([train_labels, labels])

    # load test data
    with open('data/test_batch', 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        data = d[b'data']
        data = np.reshape(data, (10000, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))
        test_data = data
        test_labels = d[b'labels']

    return (train_data, train_labels, test_data, test_labels)
