import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size

        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu_layer = ReLULayer()
        self.layers = [self.first_layer, self.relu_layer, self.output_layer]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in self.layers:
            for key in layer.params().keys():
                layer.params()[key].grad = 0

        result = X.copy()
        for layer in self.layers:
            result = layer.forward(result)

        loss, d_out = softmax_with_cross_entropy(result, y)

        for layer in self.layers:
            for key in layer.params().keys():
                loss_l2, grad_l2 = l2_regularization(layer.params()[key].value, self.reg)
                loss += loss_l2
                layer.params()[key].grad += grad_l2

        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")
        return loss


    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        pred = np.argmax(softmax(pred), axis=1)
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {'W_f': self.first_layer.W, 'B_f': self.first_layer.B, 'W_o': self.output_layer.W,
                  'B_o': self.output_layer.B}
        return result
