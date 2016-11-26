#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of gradient checking

from __future__ import print_function
from __future__ import division


def compute_approx_grad(func, theta, p, eps=1e-08):
    """
    Compute approximation of the gradient for function `func` in the point `theta` along the vector `p`
    :param func: callable
    :param theta: numpy array
    :param p: numpy array
    :param eps: float
    :rtype: float
    """
    return (func(theta + eps * p) - func(theta)) / eps
