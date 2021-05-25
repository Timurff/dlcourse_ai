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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.power(W, 2).sum()
    grad = reg_strength * 2 * W
    
    return loss, grad


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


def softmax_with_cross_entropy(preds, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        result = np.maximum(X, 0)
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = (self.X >= 0) * 1
        d_result = np.multiply(d_out, d_result)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = Param(X)
        result = X.dot(self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
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
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding

    def forward(self, X):
        X_padded = np.pad(X, pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), constant_values=0)
        batch_size, height, width, channels = X.shape
        self.X = Param(X)
        self.X_padded = Param(X_padded)
        
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                receptive_field = X_padded[:, y: y + self.filter_size, x: x + self.filter_size, :].reshape((batch_size, self.filter_size * self.filter_size * channels))
                W_reshaped = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels) 
                result[:, y, x, :] = receptive_field @ W_reshaped + self.B.value
        
        return result
                

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X_padded.value.shape
        _, out_height, out_width, out_channels = d_out.shape
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        fs = self.filter_size
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                dd_out = d_out[:, y, x, :] # (bs, out_chan)
                
                receptive_field = self.X_padded.value[:, y: y + fs, x: x + fs, :] # (bs, fs, fs, in_chan)
                receptive_field_reshaped = receptive_field.reshape((batch_size, fs * fs * channels)) # (bs, fs*fs*in_chan)
                
                W_grad_reshaped = receptive_field_reshaped.T.dot(dd_out) # (fs*fs*in_chan, out_chan)
                self.W.grad += W_grad_reshaped.reshape((fs, fs, channels, out_channels)) # (fs, fs, in_chan, out_chan)
                
                X_grad_reshaped = dd_out.dot(self.W.value.reshape(fs*fs*self.in_channels, self.out_channels).T) # (bs, fs*fs*in_chan)
                
                self.X_padded.grad[:, y:y + fs, x:x + fs, :] += X_grad_reshaped.reshape((batch_size, fs, fs, channels))
                self.B.grad += dd_out.sum(axis=0)
        
        if self.padding != 0:
            d_input = self.X_padded.grad[:, self.padding: -self.padding, self.padding: -self.padding, :]
        else:
            d_input = self.X_padded.grad
        
        self.X.grad = d_input
         
        return d_input

    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = Param(X)
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
    
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                result[:, y, x, :] = np.max(X[:, y*self.stride: y*self.stride + self.pool_size, x*self.stride: x*self.stride + self.pool_size, :], axis=(1, 2))
        
        return result
    
    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        ps = self.pool_size
        stride = self.stride
        
        batch_size, height, width, channels = self.X.value.shape
        d_input = np.zeros_like(self.X.value)
        
        out_height = int((height - ps) / stride + 1)
        out_width = int((width - ps) / stride + 1)
        
        for y in range(out_height):
            for x in range(out_width):
                window = self.X.value[:, y*stride: y*stride + ps, x*stride: x*stride + ps, :]

                d_input[:, y*stride: y*stride + ps, x*stride: x*stride + ps, :] = (np.max(window, axis=(1, 2), keepdims=True) == window) * d_out[:, y, x, :].reshape((batch_size, 1, 1, channels))
        
        return d_input
                
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape((batch_size, height*width*channels))
        
    def backward(self, d_out):
        # TODO: Implement backward pass        
        return d_out.reshape((self.X_shape))
        
    def params(self):
        # No params!
        return {}
