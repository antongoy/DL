#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of general feed-forward neural network
# THIS IMPLEMENTATION SHOULD NOT BE MODIFIED!!!

from __future__ import print_function
from __future__ import division
import numpy as np


class FFNet:

    def __init__(self, layers):
        """
        :param layers: a list of layers in network, each layer is an instance of BaseActivationFunction
        """
        # Calculates total number of net parameters
        params_number = 0
        for layer in layers:
            params_number += layer.get_params_number()

        self.layers = layers
        self.params_number = params_number

    def get_weights(self):
        """
        :return w: all net parameters in one-dimensional vector
        """
        w = np.array([])
        for layer in self.layers:
            w = np.r_[w, layer.get_weights()]

        return w

    def set_weights(self, w):
        """
        Takes a one-dimensional numpy vector of all net parameters and assign them for each layer
        :param w: a one-dimensional numpy vector
        """
        start = 0
        for layer in self.layers:
            curr_params_number = layer.get_params_number()
            curr_w = w[start:start+curr_params_number]
            layer.set_weights(curr_w)
            start += curr_params_number

    def set_direction(self, p):
        """
        Takes a one-dimensional direction vector for all net parameters and assign them for each layer
        :param p: a one-dimensional numpy vector
        """
        start = 0
        for layer in self.layers:
            curr_params_number = layer.get_params_number()
            curr_p = p[start:start+curr_params_number]
            layer.set_direction(curr_p)
            start += curr_params_number

    def compute_outputs(self, inputs):
        """
        Computes network outputs for given batch of data
        :param inputs: input data, numpy matrix of size num_inputs x num_objects
        :return outputs: network outputs, numpy matrix of size num_outputs x num_objects
        """
        if not isinstance(inputs, np.ndarray):
            raise TypeError('Input data must be given by a numpy array!')

        if inputs.shape[0] != self.layers[0].shape[0]:
            raise ValueError('Input data size does not correspond to input size of the first layer!')

        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = outputs

        return outputs

    def compute_loss_grad(self, loss_derivs):
        """
        Computes loss derivatives w.r.t. all network parameters using loss derivatives w.r.t network outputs.
        Presumes accomplished forward pass.
        :param loss_derivs: loss derivatives w.r.t. network outputs, numpy array of size num_outputs x num_objects
        :return loss_grad: loss derivatives w.r.t. network parameters, numpy array of length num_params
        """
        if not isinstance(loss_derivs, np.ndarray):
            raise TypeError('Loss derivatives must be given by a numpy array!')

        if loss_derivs.shape[0] != self.layers[-1].shape[1]:
            raise ValueError('Loss derivatives size does not correspond to output size of the network!')

        loss_grad = np.array([])
        derivs = loss_derivs.copy()
        for layer in self.layers[::-1]:
            output_derivs, w_derivs = layer.backward(derivs)
            loss_grad = np.r_[w_derivs, loss_grad]
            derivs = output_derivs

        return loss_grad

    def compute_Rp_outputs(self):
        """
        Computes network Rp outputs for batch of data, given for forward pass
        :return Rp_outputs: Rp network outputs, numpy matrix of size num_outputs x num_objects
        """
        Rp_inputs = np.zeros((self.layers[0].shape[0], self.layers[0].inputs.shape[1]))
        for layer in self.layers:
            Rp_outputs = layer.Rp_forward(Rp_inputs)
            Rp_inputs = Rp_outputs

        return Rp_outputs

    def compute_loss_Rp_grad(self, loss_Rp_derivs):
        """
        Computes loss Rp derivatives w.r.t. all network parameters using loss Rp derivatives w.r.t network outputs
        :param loss_Rp_derivs: loss Rp derivatives w.r.t. network outputs, numpy array of size num_outputs x num_objects
        :return loss_Rp_grad: loss derivatives w.r.t. all network parameters, numpy array of length num_params
        """
        if not isinstance(loss_Rp_derivs, np.ndarray):
            raise TypeError('Loss directional derivatives must be given by a numpy array!')

        if loss_Rp_derivs.shape[0] != self.layers[-1].shape[1]:
            raise Exception('Loss directional derivatives size does not correspond to output size of the network!')

        res = np.array([])
        Rp_derivs = loss_Rp_derivs.copy()
        for layer in self.layers[::-1]:
            output_Rp_derivs, Rp_w_derivs = layer.Rp_backward(Rp_derivs)
            res = np.r_[Rp_w_derivs, res]
            Rp_derivs = output_Rp_derivs

        return res

    def get_activations(self, layer_number):
        """
        :param layer_number: particular layer number
        :return a: activations for given layer on data, given for forward pass, numpy matrix of size num_outputs x num_objects
        """
        return self.layers[layer_number].get_activations()

if __name__ == '__main__':
    from layers import FCLayer
    from activations import SigmoidActivationFunction
    from approx_gradient import compute_approx_grad

    def loss_function(net, w, inputs):
        net.set_weights(w)

        outputs = net.compute_outputs(inputs)

        return 0.5 * np.sum((inputs - outputs) ** 2, axis=0).mean()

    def loss_function_grad(net, w, inputs):
        net.set_weights(w)
        outputs = net.compute_outputs(inputs)

        exact_grad = net.compute_loss_grad(outputs - inputs)

        return exact_grad

    print('>>> Testing basic functionality...')

    inputs = np.random.normal(size=(5, 50))

    hidden_layer_1 = FCLayer(shape=(5, 2),
                             afun=SigmoidActivationFunction(),
                             use_bias=True)

    hidden_layer_2 = FCLayer(shape=(2, 5),
                             afun=SigmoidActivationFunction(),
                             use_bias=True)

    num_params = hidden_layer_1.get_params_number() + hidden_layer_2.get_params_number()

    w0 = np.random.normal(size=num_params)

    net = FFNet([hidden_layer_1, hidden_layer_2])

    approx_grad = np.zeros_like(w0)
    p = np.zeros_like(w0)

    for i in range(num_params):
        p[:] = 0
        p[i] = 1

        approx_grad[i] = compute_approx_grad(lambda x: loss_function(net, x, inputs), w0, p)

    exact_grad = loss_function_grad(net, w0, inputs)

    p = np.random.normal(size=num_params)
    approx_Hp = compute_approx_grad(lambda x: loss_function_grad(net, x, inputs), w0, p)

    net.set_direction(p)

    Rp_outputs = net.compute_Rp_outputs()
    exact_Hp = net.compute_loss_Rp_grad(Rp_outputs)

    print('>>> Approximate gradient')
    print(approx_grad)

    print('>>> Exact gradient')
    print(exact_grad)

    print('>>> Is gradients close? (eps = 1e-05)')
    print(np.allclose(approx_grad, exact_grad, rtol=0, atol=1e-05))

    print('>>> Approximate Hp')
    print(approx_Hp)

    print('>>> Exact Hp')
    print(exact_Hp)

    print('>>> Is Hp close? (eps = 1e-05)')
    print(np.allclose(approx_Hp, exact_Hp, rtol=0, atol=1e-05))
