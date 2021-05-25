import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax, cross_entropy_loss
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        
        in_w, in_h, in_ch = input_shape
        max_pool1_size = 4
        max_pool2_size = 4
        fc_layer_input_size = int(in_w * in_h/ (max_pool1_size * max_pool2_size)**2)
        
        self.first_conv_layer = ConvolutionalLayer(in_channels=in_ch,
                                                   out_channels=conv1_channels,
                                                   filter_size=3, 
                                                   padding=1)

        self.ReLu_first = ReLULayer()
        
        self.first_max_pool = MaxPoolingLayer(pool_size=max_pool1_size, 
                                              stride=max_pool1_size)

        self.second_conv_layer = ConvolutionalLayer(in_channels=conv1_channels,
                                                   out_channels=conv2_channels,
                                                   filter_size=3, 
                                                   padding=1)
        
        self.ReLu_second = ReLULayer()
        
        self.second_max_pool = MaxPoolingLayer(pool_size=max_pool2_size, 
                                              stride=max_pool2_size)
        
        self.flattener = Flattener()
        
        self.fully_connected = FullyConnectedLayer(n_input=fc_layer_input_size * conv2_channels,
                                                   n_output=n_output_classes)
        
        self.layers = [self.first_conv_layer, 
                       self.ReLu_first, 
                       self.first_max_pool,
                       
                       self.second_conv_layer,
                       self.ReLu_second, 
                       self.second_max_pool, 
                       
                       self.flattener, 
                       self.fully_connected]
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        for layer in self.layers:
            for key in layer.params().keys():
                layer.params()[key].grad = 0
                
        result = X.copy()
        for layer in self.layers:
            result = layer.forward(result)

        loss, d_out = softmax_with_cross_entropy(result, y)

        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)

        return loss
        
        
        
    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)

        pred = np.argmax(softmax(pred), axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for index, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result[f'{index}_{str(layer)}_{name}'] = param

        return result
