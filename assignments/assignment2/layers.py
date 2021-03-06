import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if predictions.ndim == 1:
        predictions_normalized = predictions.copy() - predictions.max()
        predictions_exp = np.exp(predictions_normalized)

        exp_sum = predictions_exp.sum()
        results = predictions_exp / exp_sum
    else:
        predictions_normalized = predictions.copy() - predictions.max(axis=1).reshape((-1, 1))
        predictions_exp = np.exp(predictions_normalized)

        exp_sum = predictions_exp.sum(axis=1)
        results = predictions_exp / exp_sum.reshape((-1, 1))

    return results


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    loss = 0.0
    for i in range(probs.shape[0]):
        loss -= np.log(probs[i][target_index[i]])
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.power(W, 2).sum()
    grad = reg_strength * 2 * W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    if preds.ndim == 1:
        dprediction[target_index] -= 1
        return loss, dprediction
    else:
        for ind, value in enumerate(target_index):
            dprediction[ind, value] -= 1
        return loss / probs.shape[0], dprediction / probs.shape[0]
    

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        result = np.maximum(X, 0)
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = (self.X >= 0) * 1
        d_result = np.multiply(d_out, d_result)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        result = X.dot(self.W.value) + self.B.value
        return result
    
    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        dW = self.X.value.T.dot(d_out)
        dX = d_out.dot(self.W.value.T)
        dB = d_out.sum(axis=0).reshape((1, -1))
        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.B.grad += dB
        self.W.grad += dW
        d_input = dX
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
