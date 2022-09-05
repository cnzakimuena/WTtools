
import numpy as np
import os
import pywt


# --------------------------------------------------------------------------------
# Loading the UCI-HAR time-series dataset
# --------------------------------------------------------------------------------


def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data


def read_labels_ucihar(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


def load_ucihar_data(folder):
    train_folder = folder + 'train/InertialSignals/'
    test_folder = folder + 'test/InertialSignals/'
    labelfile_train = folder + 'train/y_train.txt'
    labelfile_test = folder + 'test/y_test.txt'
    train_signals, test_signals = [], []
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels


folder_ucihar = './data/UCI_HAR/'
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)

# --------------------------------------------------------------------------------
# Applying the CWT on the dataset and transforming the data to the right format
# --------------------------------------------------------------------------------

scales = range(1, 128)
waveletname = 'morl'
train_size = 5000
test_size = 500

train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))

for ii in range(0, train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0, 9):
        signal = train_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :127]
        train_data_cwt[ii, :, :, jj] = coeff_

test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
for ii in range(0, test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0, 9):
        signal = test_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:, :127]
        test_data_cwt[ii, :, :, jj] = coeff_

uci_har_labels_train = list(map(lambda x: int(x) - 1, train_labels_ucihar))
uci_har_labels_test = list(map(lambda x: int(x) - 1, test_labels_ucihar))

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])

# np.save('x_train_file.npy', x_train)
# np.save('x_test_file.npy', x_test)
# np.save('y_train_file.npy', y_train)
# np.save('y_test_file.npy', y_test)
