from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        self.data_shape = data.shape
        # print(data.reshape(data.shape[0], -1).shape)
        return data.reshape(data.shape[0], -1)

    def backward(self, previous_partial_gradient):
        # TODO
        return previous_partial_gradient.reshape(self.data_shape)
