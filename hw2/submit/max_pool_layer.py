import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

    def Im2col(self, img, k_size, s):
        """
        :param img: input image(after padded), shape is (N, C, H, W)
        :param k_size: kernerl size
        :param s: stride
        :return: col trasnformed from input image, shape is (C * k_size * k_size, N * out_h * out_W)
        """
        N, C, H, W = img.shape
        out_height = (H - k_size) // s + 1
        out_width = (W - k_size) // s + 1

        # convert img to column
        col = np.zeros((N * out_width * out_height, k_size * k_size * C), dtype=np.float32)
        out_size = out_height * out_width
        for y in prange(out_height):
            y_min = y * s
            y_max = y_min + k_size
            y_start = y * out_width
            for x in prange(out_width):
                x_min = x * s
                x_max = x_min + k_size
                col[y_start + x::out_size, :] = img[:, :, y_min:y_max, x_min:x_max].reshape(N, -1)
        return col.transpose()

    def Col2im(self, col, padded_shape, k_size, s):
        """
        :param col: input col
        :param padded_shape: the shape of data
        :param k_size: kernel size
        :param s: stride
        :param p: padding
        :return: image transformed from input col, the shape is out_shape
        """
        col = col.transpose()
        N, C, H, W = padded_shape
        out_h = (H - k_size) // s + 1
        out_w = (W - k_size) // s + 1
        img = np.zeros(padded_shape, dtype=np.float32)
        out_size = out_h * out_w
        for y in prange(out_h):
            y_min = y * s
            y_max = y_min + k_size
            y_start = y * out_w
            for x in prange(out_w):
                x_min = x * s
                x_max = x_min + k_size
                img[:, :, y_min:y_max, x_min:x_max] += col[y_start + x::out_size, :].reshape(N, C, k_size, k_size)
        return img

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # data is N x C x H x W
        # TODO
        return None

    def forward(self, data):
        # TODO
        # define parameter
        k_size = self.kernel_size
        p = self.padding
        s = self.stride
        N, C, H, W = data.shape

        # pad
        data_pad = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        out_height = (H + 2 * p - k_size) // s + 1
        out_width = (W + 2 * p - k_size) // s + 1
        x_col = self.Im2col(data_pad, k_size, s)
        x_col = x_col.reshape(C, k_size * k_size, -1)
        arg_max = np.argmax(x_col, axis=1)
        out_col = np.max(x_col, axis=1)
        self.padded_shape = data_pad.shape
        self.arg_max = arg_max
        out = np.moveaxis(out_col.reshape(C, N, out_height, out_width), 1, 0)
        return out

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data):
        # data is N x C x H x W
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        N, C, H, W = previous_partial_gradient.shape
        p = self.padding
        pool_size = self.kernel_size * self.kernel_size
        dx_col = np.zeros((pool_size, previous_partial_gradient.size))

        # we need to transform arg_max to direct sequence
        changed_arg_max = np.moveaxis(self.arg_max.reshape(C, N, -1), 1, 0)
        dx_col[changed_arg_max.flatten(), np.arange(self.arg_max.size)] = previous_partial_gradient.flatten()
        dx = self.Col2im(dx_col, self.padded_shape, self.kernel_size, self.stride)

        # remove padded part
        dx_h = dx.shape[2]
        dx_w = dx.shape[3]
        dx_depad = dx[:, :, p:dx_h - p, p:dx_w - p]
        return dx_depad

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
