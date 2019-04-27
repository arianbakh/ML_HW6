import math
import numpy as np
import os
import sys
from matplotlib import pyplot as plt


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
V_PATH = os.path.join(BASE_DIR, 'v')
W_PATH = os.path.join(BASE_DIR, 'w')


# Algorithm Parameters
ALPHA = 0.02
ERROR_THRESHOLD = 0.05
HIDDEN_LAYER_SIZE = 5  # extra one for bias


BINARY_MODE = 'binary'
BIPOLAR_MODE = 'bipolar'
MODIFIED_MODE = 'modified'
MODES = [BINARY_MODE, BIPOLAR_MODE, MODIFIED_MODE]


_binary_sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
_binary_sigmoid_derivative = np.vectorize(lambda x: _binary_sigmoid(x) * (1 - _binary_sigmoid(x)))
_bipolar_sigmoid = np.vectorize(lambda x: 2 * _binary_sigmoid(x) - 1)
_bipolar_sigmoid_derivative = np.vectorize(lambda x: 0.5 * (1 + _bipolar_sigmoid(x)) * (1 - _bipolar_sigmoid(x)))


def _get_pairs(mode):
    input_vectors = []
    target_vectors = []
    for i in [0, 1]:
        for j in [0, 1]:
            input_vectors.append([1, i, j])  # including bias
            if i != j:  # XOR is True
                if mode == BINARY_MODE or mode == BIPOLAR_MODE:
                    target_vectors.append([1])
                elif mode == MODIFIED_MODE:
                    target_vectors.append([0.8])
            else:  # XOR is False
                if mode == BINARY_MODE:
                    target_vectors.append([0])
                elif mode == BIPOLAR_MODE:
                    target_vectors.append([-1])
                elif mode == MODIFIED_MODE:
                    target_vectors.append([-0.8])
    return np.array(input_vectors), np.array(target_vectors)


def _init_v(input_vectors):
    if os.path.exists(V_PATH):
        return np.load(V_PATH)
    else:
        v = np.random.rand(input_vectors.shape[1], HIDDEN_LAYER_SIZE) - 0.5
        np.save(V_PATH, v)
        return v


def _init_w(target_vectors):
    if os.path.exists(W_PATH):
        return np.load(W_PATH)
    else:
        w = np.random.rand(HIDDEN_LAYER_SIZE, target_vectors.shape[1]) - 0.5
        np.save(W_PATH, w)
        return w


def _train(input_vectors, target_vectors, mode):
    errors = []

    v = _init_v(input_vectors)
    w = _init_w(target_vectors)

    epoch = 0
    total_squared_error = ERROR_THRESHOLD + 1
    while total_squared_error > ERROR_THRESHOLD:
        epoch += 1
        total_squared_error = 0

        for i in range(input_vectors.shape[0]):
            # feed forward
            x = np.reshape(input_vectors[i], (1, input_vectors[i].shape[0]))
            z_in = np.matmul(
                x,
                v
            )
            if mode == BINARY_MODE:
                z = _binary_sigmoid(z_in)
            else:
                z = _bipolar_sigmoid(z_in)
            z[0, 0] = 1  # bias
            y_in = np.matmul(
                z,
                w
            )
            if mode == BINARY_MODE:
                y = _binary_sigmoid(y_in)
            else:
                y = _bipolar_sigmoid(y_in)

            # error calculation
            for j in range(y.shape[0]):
                total_squared_error += (y[j] - target_vectors[i, j]) ** 2

            # back propagation
            if mode == BINARY_MODE:
                output_sigma = (target_vectors[i] - y) * _binary_sigmoid_derivative(y_in)
            else:
                output_sigma = (target_vectors[i] - y) * _bipolar_sigmoid_derivative(y_in)
            delta_w = ALPHA * np.matmul(
                z.T,
                output_sigma
            )
            hidden_sigma_in = np.matmul(
                output_sigma,
                w.T
            )
            if mode == BINARY_MODE:
                hidden_sigma = hidden_sigma_in * _binary_sigmoid_derivative(z_in)
            else:
                hidden_sigma = hidden_sigma_in * _bipolar_sigmoid_derivative(z_in)
            delta_v = ALPHA * np.matmul(
                x.T,
                hidden_sigma
            )

            # update weights
            w += delta_w
            v += delta_v

        if epoch % 10 == 0:
            sys.stdout.write('\rEpoch %d: Error %f' % (epoch, total_squared_error))
            sys.stdout.flush()

        errors.append(total_squared_error)

    print()  # newline

    return errors


def run(mode):
    input_vectors, target_vectors = _get_pairs(mode)
    errors = _train(input_vectors, target_vectors, mode)
    plt.plot(errors)
    plt.savefig(os.path.join(BASE_DIR, '%s.png' % mode))


if __name__ == '__main__':
    exec_mode = sys.argv[1]
    if exec_mode not in MODES:
        print('Invalid Argument %s' % exec_mode)
        exit()
    run(exec_mode)
