from numba import njit, prange
import numpy as np

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope

    def forward(self, data):
        # TODO
        self.data = data  # save data for backward function
        cur_data = np.array(data, copy=True)
        cur_data[data <= 0] *= self.slope
        return cur_data

    def backward(self, previous_partial_gradient):
        # TODO
        cur_grad = np.array(previous_partial_gradient, copy=True)
        cur_grad[self.data <= 0] *= self.slope
        return cur_grad

    """
    def forward(self, data):
        # use numba
        self.data = data
        flatten_data = data.flatten()
        for i in prange(flatten_data.shape[0]):
            if flatten_data[i] <= 0:
                flatten_data[i] = self.slope * flatten_data[i]
        return flatten_data.reshape(data.shape)


    def backward(self, previous_partial_gradient):
        # TODO
        # use numba
        flatten_grad = previous_partial_gradient.flatten()
        flatten_data = self.data.flatten()
        for i in prange(flatten_grad.shape[0]):
            if flatten_data[i] < 0:
                flatten_grad[i] = self.slope
        return flatten_grad.reshape(previous_partial_gradient.shape)
        """