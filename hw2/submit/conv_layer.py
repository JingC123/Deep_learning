from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()

    # @staticmethod
    # @njit(parallel=True, cache=True)
    def Im2col(self, img, k_size, s):
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
        # print(img.shape)
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
    def forward_numba(data, weights, bias):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        shape1 = np.dot(weights, data).shape
        output = np.zeros(shape1)
        for i in prange(weights.shape[0]):
            for j in prange(data.shape[1]):
                for k in prange(weights.shape[1]):
                    output[i][j] += weights[i][k] * data[k][j]
        return output

    def forward(self, data):
        # TODO
        # define parameter
        k_size = self.kernel_size
        p = self.padding
        s = self.stride
        CNew = self.weight.data.shape[1]
        N, C, H, W = data.shape

        # pad
        data_pad = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        out_height = (H + 2 * p - k_size) // s + 1
        out_width = (W + 2 * p - k_size) // s + 1
        x_col = self.Im2col(data_pad, k_size, s)
        self.padded_shape = data_pad.shape
        self.x_col = x_col
        w = self.weight.data
        b = self.bias.data
        w_col = np.moveaxis(w, 1, 0).reshape(CNew, -1)
        out_col = np.dot(w_col, x_col) + b.reshape(b.shape[0], -1)
        # out_col = self.forward_numba(x_col, w_col, b)
        output = np.moveaxis(out_col.reshape(CNew, N, out_height, out_width), 1, 0)
        # print(output.shape)
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K

        return None

    def backward(self, previous_partial_gradient):
        # TODO
        CNew = self.weight.data.shape[1]
        COld = self.weight.data.shape[0]
        k_size = self.weight.data.shape[2]
        grad_col = np.moveaxis(previous_partial_gradient, 1, 0).reshape(CNew, -1)
        p = self.padding

        # compute dw, db
        dw_col = np.dot(grad_col, self.x_col.T)
        dw = np.moveaxis(dw_col.reshape((CNew, COld, k_size, k_size)), 0, 1)
        db = np.sum(previous_partial_gradient, axis=(0, 2, 3)).reshape(CNew, -1).flatten()
        self.weight.grad = dw
        self.bias.grad = db

        # compute dx
        w = self.weight.data
        w_col = np.moveaxis(w, 1, 0).reshape(CNew, -1)
        dx_col = np.dot(w_col.T, grad_col)
        dx = self.Col2im(dx_col, self.padded_shape, self.kernel_size, self.stride)
        dx_h = dx.shape[2]
        dx_w = dx.shape[3]
        dx_depad = dx[:, :, p:dx_h - p, p:dx_w - p]
        return dx_depad

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
