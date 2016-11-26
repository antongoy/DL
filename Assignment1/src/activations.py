#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of activation functions used within neural networks

from __future__ import print_function
from __future__ import division

import numpy as np


class BaseActivationFunction(object):

    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')


class LinearActivationFunction(BaseActivationFunction):

    def val(self, inputs):
        return inputs

    def deriv(self, inputs):
        return np.ones_like(inputs)

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)


class SigmoidActivationFunction(BaseActivationFunction):

    def val(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def deriv(self, inputs):
        val = self.val(inputs)
        return val * (1 - val)

    def second_deriv(self, inputs):
        val = self.val(inputs)
        return val * (1 - 1 * val) * (1 - 2 * val)


class ReluActivationFunction(BaseActivationFunction):

    def val(self, inputs):
        return np.maximum(0, inputs)

    def deriv(self, inputs):
        derivatives = np.zeros_like(inputs)
        derivatives[inputs > 0] = 1
        return derivatives

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)


if __name__ == '__main__':
    from approx_gradient import compute_approx_grad

    def log(activ_func):
        exact_grad = activ_func.deriv(u)
        approx_grad = compute_approx_grad(activ_func.val, u, np.ones_like(u))

        print('\tActivations:', activ_func.val(u))
        print('\tGradient checking')
        print('\t\tExact first derivatives:', exact_grad)
        print('\t\tFirst derivative approximations:', approx_grad)
        print('\t\tErrors of approximation:', np.abs(exact_grad - approx_grad))

    print('>>> Testing basic functionality...')

    linear_activ_func = LinearActivationFunction()
    sigmoid_activ_func = SigmoidActivationFunction()
    relu_activ_func = ReluActivationFunction()

    u = np.random.normal(scale=10, size=3)

    print('>>> Points:', u)

    print('>>> Linear Activation')
    log(linear_activ_func)

    print('>>> Sigmoid Activation')
    log(sigmoid_activ_func)

    print('>>> ReLU Activation')
    log(relu_activ_func)