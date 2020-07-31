import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))

    def forward(self, data):
        # TODO
        dataMove = np.moveaxis(data, 1, -1)
        batch = np.prod(dataMove.shape, axis=0) / data.shape[1]
        dataReshape = dataMove.reshape(int(batch), data.shape[1])
        self.dataReshape = dataReshape
        output = np.maximum(dataReshape, 0) + np.minimum(dataReshape, 0) * self.slope.data
        outputReshape = output.reshape(dataMove.shape)
        outputMove = np.moveaxis(outputReshape, -1, 1)
        return outputMove

    def backward(self, previous_partial_gradient):
        # TODO
        # dL/dY and dL/dX
        # change dimension
        dyMove = np.moveaxis(previous_partial_gradient, 1, -1)
        batch = np.prod(dyMove.shape, axis=0) / previous_partial_gradient.shape[1]
        dyReshape = dyMove.reshape(int(batch), previous_partial_gradient.shape[1])

        # compute
        self.slope.zero_grad()
        slopeGrad = np.sum(np.array(self.dataReshape < 0, dtype=int) * self.dataReshape * dyReshape, axis=0)
        if self.slope.grad.shape[0]>1:
            self.slope.grad = slopeGrad
        if self.slope.grad.shape[0]==1:
            self.slope.grad = np.sum(slopeGrad)
        dx = np.array(self.dataReshape > 0, dtype=int) * dyReshape + np.array(self.dataReshape < 0, dtype=int) * dyReshape * self.slope.data

        # change dimension
        dxReshape = dx.reshape(dyMove.shape)
        dxMove=np.moveaxis(dxReshape,-1,1)
        print(self.slope.data)
        return dxMove
