#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

from __future__ import division
from __future__ import print_function
import numpy as np

import ffnet


class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and outputs is different!')

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of data
        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """

        outputs = self.net.compute_outputs(inputs)
        derivs_wrt_W = self.net.compute_loss_grad(outputs - inputs)

        loss_value = 0.5 * np.sum((inputs - outputs) ** 2, axis=0).mean()

        return loss_value, derivs_wrt_W

    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """

        self.net.set_direction(p)

        Rp_outputs = self.net.compute_Rp_outputs()
        Hp = self.net.compute_loss_Rp_grad(Rp_outputs)

        return Hp

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """

        self.net.set_direction(p)
        qs = self.net.compute_Rp_outputs()

        return self.net.compute_loss_grad(qs)

    def run_adam(self, inputs, step_size=0.1, max_epoch=300, minibatch_size=20, l2_coef=1e-5, test_inputs=None):
        """
        ADAM stochastic optimization method with fixed stepsizes
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param max_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numby matrix of size num_features x num_test_objects
        """


    def run_hfn(self, inputs):
        """
        Hessian-free Newton optimization method
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        """
        raise NotImplementedError('Implementation will be provided')

if __name__ == '__main__':
    from layers import FCLayer
    from activations import SigmoidActivationFunction

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
    p = np.random.normal(size=num_params)

    autoencoder = Autoencoder([hidden_layer_1, hidden_layer_2])
    autoencoder.net.set_weights(w0)

    print('>>> Loss and gradient of the net')
    print(autoencoder.compute_loss(inputs))

    print('>>> Hessian-Vector product')
    print(autoencoder.compute_hessvec(p))

    print('>>> GaussNewtonApproximation-Vector product')
    print(autoencoder.compute_gaussnewtonvec(p))