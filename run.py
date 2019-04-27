import math
import numpy as np
import os
import sys
from matplotlib import pyplot as plt


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'output.png')


# Algorithm Parameters
ALPHA = 0.02
ERROR_THRESHOLD = 0.05
HIDDEN_LAYER_SIZE = 4


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
            input_vectors.append([i, j])
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


def _init_hidden_weight_matrix(input_vectors):
    return np.random.rand(input_vectors.shape[1] + 1, HIDDEN_LAYER_SIZE) - 0.5  # +1 is for bias


def _init_output_weight_matrix(target_vectors):
    return np.random.rand(HIDDEN_LAYER_SIZE, target_vectors.shape[1]) - 0.5


def _train(input_vectors, target_vectors, mode):
    errors = []

    hidden_weight_matrix = _init_hidden_weight_matrix(input_vectors)
    output_weight_matrix = _init_output_weight_matrix(target_vectors)

    total_squared_error = ERROR_THRESHOLD + 1
    while total_squared_error > ERROR_THRESHOLD:
        total_squared_error = 0
        for i in range(len(input_vectors)):
            # feedforward
            biased_input = np.insert(input_vectors[i], 0, 1)
            z_in = np.matmul(
                np.reshape(biased_input, (1, biased_input.shape[0])),
                hidden_weight_matrix
            )
            if mode == BINARY_MODE:
                z = _binary_sigmoid(z_in)
            else:
                z = _bipolar_sigmoid(z_in)
            y_in = np.matmul(z, output_weight_matrix)
            if mode == BINARY_MODE:
                y = _binary_sigmoid(y_in)
            else:
                y = _bipolar_sigmoid(y_in)

            # add error
            for j in range(y.shape[0]):
                total_squared_error += (y[j] - target_vectors[i, j]) ** 2

            # backpropagation
            if mode == BINARY_MODE:
                output_sigma = (target_vectors[i] - y) * _binary_sigmoid_derivative(y_in)
            else:
                output_sigma = (target_vectors[i] - y) * _bipolar_sigmoid_derivative(y_in)
            output_weight_delta = ALPHA * np.matmul(
                z.T,
                output_sigma
            )
            hidden_sigma_in = np.matmul(
                output_sigma,
                output_weight_matrix.T
            )
            if mode == BINARY_MODE:
                hidden_sigma = hidden_sigma_in * _binary_sigmoid_derivative(z_in)
            else:
                hidden_sigma = hidden_sigma_in * _bipolar_sigmoid_derivative(z_in)
            hidden_weight_delta = np.matmul(
                np.reshape(biased_input, (biased_input.shape[0], 1)),
                hidden_sigma
            )

            # update weights
            output_weight_matrix += output_weight_delta
            hidden_weight_matrix += hidden_weight_delta

        print(total_squared_error)  # TODO remove
        errors.append(total_squared_error)

    return errors


def run(mode):
    input_vectors, target_vectors = _get_pairs(mode)
    errors = _train(input_vectors, target_vectors, mode)
    print(len(errors))  # TODO


if __name__ == '__main__':
    exec_mode = sys.argv[1]
    if exec_mode not in MODES:
        print('Invalid Argument %s' % exec_mode)
        exit()
    run(exec_mode)
