#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of layers used within neural networks

from __future__ import print_function
from __future__ import division
import numpy as np


class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.shape = shape
        self.num_inputs = shape[0]
        self.num_outputs = shape[1]
        self.activation_func = afun
        self.use_bias = use_bias

    def get_params_number(self):
        num_params = self.num_outputs * self.num_inputs

        if self.use_bias:
            num_params += self.num_outputs

        return num_params

    def get_weights(self):
        return np.ravel(self.W)

    def set_weights(self, w):
        num_params = self.get_params_number()

        if w.shape[0] != num_params:
            raise ValueError('Invalid number of the layer parameters')

        if self.use_bias:
            self.W = w.reshape((self.num_outputs, self.num_inputs + 1))
        else:
            self.W = w.reshape((self.num_outputs, self.num_inputs))

    def set_direction(self, p):
        num_params = self.get_params_number()

        if p.shape[0] != num_params:
            raise ValueError('Invalid number of the layer parameters')

        if self.use_bias:
            self.P = p.reshape((self.num_outputs, self.num_inputs + 1))
        else:
            self.P = p.reshape((self.num_outputs, self.num_inputs))

    def forward(self, inputs):
        if inputs.shape[0] != self.num_inputs:
            raise ValueError('Size of the batch does not correspond to size of the layer')

        # W --> (num_outputs x num_inputs), b --> (num_outputs x 1)
        # logits --> (num_outputs x num_objects)
        # inputs --> (num_inputs x num_objects)
        if self.use_bias:
            self.inputs = np.vstack((inputs, np.ones(inputs.shape[1])))
        else:
            self.inputs = inputs

        self.logits = np.dot(self.W, self.inputs)

        return self.activation_func.val(self.logits)

    def backward(self, derivs):
        if derivs.shape[0] != self.num_outputs:
            raise ValueError('Size of the derivatives does not correspond to output size of the layer')

        # derivs --> (num_outputs x num_objects)
        self.derivs = derivs
        self.derivs_wrt_logits = self.derivs * self.activation_func.deriv(self.logits)

        self.derivs_wrt_inputs = np.dot(self.W.T, self.derivs_wrt_logits)

        if self.use_bias:
            self.derivs_wrt_inputs = self.derivs_wrt_inputs[:-1, :]

        self.derivs_wrt_W = np.dot(self.derivs_wrt_logits, self.inputs.T) / self.inputs.shape[1]

        return self.derivs_wrt_inputs, np.ravel(self.derivs_wrt_W)

    def Rp_forward(self, Rp_inputs):
        if Rp_inputs.shape[0] != self.num_inputs:
            raise ValueError('Size of `Rp_inputs` does not correspond to the number of layer inputs')

        # Rp_inputs --> (num_inputs x num_objects)
        if self.use_bias:
            self.Rp_inputs = np.vstack((Rp_inputs, np.zeros(Rp_inputs.shape[1])))
        else:
            self.Rp_inputs = Rp_inputs

        self.Rp_logits = np.dot(self.W, self.Rp_inputs) + np.dot(self.P, self.inputs)

        return self.activation_func.deriv(self.logits) * self.Rp_logits

    def Rp_backward(self, Rp_derivs):
        if Rp_derivs.shape[0] != self.num_outputs:
            raise ValueError('Size of `Rp_outputs` does not correspond to the number of layer outputs')

        Rp_derivs_wrt_logits = Rp_derivs * self.activation_func.deriv(self.logits) + \
                               self.derivs * self.activation_func.second_deriv(self.logits) * self.Rp_logits

        Rp_derivs_wrt_inputs = np.dot(self.P.T, self.derivs_wrt_logits) + np.dot(self.W.T, Rp_derivs_wrt_logits)

        if self.use_bias:
            Rp_derivs_wrt_inputs = Rp_derivs_wrt_inputs[:-1, :]

        Rp_derivs_wrt_W = np.dot(Rp_derivs_wrt_logits, self.inputs.T) + np.dot(self.derivs_wrt_logits, self.Rp_inputs.T)

        Rp_derivs_wrt_W /= self.inputs.shape[1]

        return Rp_derivs_wrt_inputs, np.ravel(Rp_derivs_wrt_W)

    def get_activations(self):
        return self.activation_func.val(self.logits)


if __name__ == '__main__':
    from activations import SigmoidActivationFunction

    print('>>> Testing basic functionality...')

    num_inputs = 1000
    num_outputs = 100
    num_objects = 50

    layer = FCLayer(shape=(num_inputs, num_outputs), afun=SigmoidActivationFunction(), use_bias=True)

    layer.set_weights(np.random.normal(size=(num_inputs + 1) * num_outputs))
    layer.set_direction(np.random.normal(size=(num_inputs + 1) * num_outputs))

    batch = np.random.normal(size=(num_inputs, num_objects))
    outputs = layer.forward(batch)

    derivs_wrt_outputs = np.random.normal(size=(num_outputs, num_objects))
    derivs_wrt_inputs, derivs_wrt_weights = layer.backward(derivs_wrt_outputs)

    Rp_inputs = np.random.normal(size=(num_inputs, num_objects))
    Rp_ouputs = layer.Rp_forward(Rp_inputs)

    Rp_derivs_wrt_outputs = np.random.normal(size=(num_outputs, num_objects))
    Rp_derivs_wrt_inputs, Rp_derivs_wrt_weights = layer.Rp_backward(Rp_derivs_wrt_outputs)
