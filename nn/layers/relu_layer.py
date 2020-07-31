import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)

    def forward(self, data):
        # TODO
        self.data = data  # save data for backward function
        return np.maximum(data, 0)

    def backward(self, previous_partial_gradient):
        # TODO
        cur_grad = np.array(previous_partial_gradient, copy=True)
        # print(cur_grad.shape)
        # print(self.data.shape)
        cur_grad[self.data <= 0] = 0
        return cur_grad


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO
        flatten_data = data.flatten()
        data_len = flatten_data.shape[0]
        for i in prange(data_len):
            if flatten_data[i] < 0:
                flatten_data[i] = 0
        return flatten_data.reshape(data.shape)

    def forward(self, data):
        # Modify if you want
        self.data = data
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO
        flatten_grad = grad.flatten()
        flatten_data = data.flatten()
        for i in prange(flatten_grad.shape[0]):
            if flatten_data[i] < 0:
                flatten_grad[i] = 0
        return flatten_grad.reshape(grad.shape)

    def backward(self, previous_partial_gradient):
        # TODO
        self.backward_numba(self.data, previous_partial_gradient)
        return None
