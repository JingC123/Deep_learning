import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: N-Dimensional non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets.
            Example: inputs might be (4 x 10), targets (4) and axis 1.
        :param targets: (N-1)-Dimensional class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        # compute softmax
        maxValue = np.max(logits, axis = axis, keepdims= True)
        exps = np.exp(logits-maxValue)
        softmax = exps / np.sum(exps, axis= axis, keepdims = True)
        logSoftmax = logits - maxValue - np.log(np.sum(np.exp(logits-maxValue), axis = axis, keepdims=True))

        # change dimension for softmax
        batch = int(np.prod(logSoftmax.shape,axis=0) / logSoftmax.shape[axis])
        softmaxReshape = np.moveaxis(logSoftmax,axis, -1).reshape(batch, logSoftmax.shape[axis])

        # change dimension for targets
        targetsReshape = targets.reshape((batch))

        # save data
        # self.softmax = softmax
        # self.targets = targets
        self.shapeAftermove = np.moveaxis(logSoftmax,axis, -1).shape
        self.softmaxReshape = softmaxReshape
        self.targetsReshape = targetsReshape
        self.axis= axis

        # cross_entropy
        log_likelihood = -softmaxReshape[range(batch), targetsReshape]
        if self.reduction == "mean":
            loss = np.sum(log_likelihood) / batch
        elif self.reduction == "sum":
            loss = np.sum(log_likelihood)
        return loss


    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        """
        #####  for 2 dimension input
        grad = self.softmax
        n = self.targets.shape[0]
        grad[range(n), self.targets] -= 1
        if self.reduction == "mean":
            grad = grad/n
        if self.reduction == "sum":
            grad = grad
        return grad       
        
        """
        # compute gradient
        grad =np.exp(self.softmaxReshape)
        batch = grad.shape[0]
        grad[range(batch), self.targetsReshape] -=1
        if self.reduction == "mean":
            grad = grad/batch
        if self.reduction == "sum":
            grad = grad

        # change dimension
        gradReshape = np.moveaxis(grad.reshape(self.shapeAftermove), -1, self.axis)
        return gradReshape
