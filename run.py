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


BINARY_MODE = 'binary'
BIPOLAR_MODE = 'bipolar'
MODIFIED_MODE = 'modified'
MODES = [BINARY_MODE, BIPOLAR_MODE, MODIFIED_MODE]


_binary_sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
_binary_sigmoid_derivative = np.vectorize(lambda x: _binary_sigmoid(x) * (1 - _binary_sigmoid(x)))
_bipolar_sigmoid = np.vectorize(lambda x: 2 * _binary_sigmoid(x) - 1)
_bipolar_sigmoid_derivative = np.vectorize(lambda x: 0.5 * (1 + _bipolar_sigmoid(x)) * (1 - _bipolar_sigmoid(x)))


def _get_pairs(mode):
    input_vector = []
    target_vector = []
    for i in [0, 1]:
        for j in [0, 1]:
            input_vector.append([i, j])
            if i != j:  # XOR is True
                if mode == BINARY_MODE or mode == BIPOLAR_MODE:
                    target_vector.append(1)
                elif mode == MODIFIED_MODE:
                    target_vector.append(0.8)
            else:  # XOR is False
                if mode == BINARY_MODE:
                    target_vector.append(0)
                elif mode == BIPOLAR_MODE:
                    target_vector.append(-1)
                elif mode == MODIFIED_MODE:
                    target_vector.append(-0.8)
    return input_vector, target_vector


def _train(mode):
    pass  # TODO


def run(mode):
    print(_get_pairs(mode))  # TODO


if __name__ == '__main__':
    exec_mode = sys.argv[1]
    if exec_mode not in MODES:
        print('Invalid Argument %s' % exec_mode)
        exit()
    run(exec_mode)
