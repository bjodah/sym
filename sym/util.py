# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np


def banded_jacobian(y, x, ml, mu):
    """ Calculates a banded version of the jacobian

    Compatible with the format requested by
    :func:`scipy.integrate.ode` (for SciPy >= v0.15).

    Parameters
    ----------
    y: array_like of expressions
    x: array_like of symbols
    ml: int
        number of lower bands
    mu: int
        number of upper bands

    Returns
    -------
    2D array of shape ``(1+ml+mu, len(y))``

    """
    ny = len(y)
    nx = len(x)
    packed = np.zeros((mu+ml+1, nx), dtype=object)

    def set(ri, ci, val):
        packed[ri-ci+mu, ci] = val

    for ri in range(ny):
        for ci in range(max(0, ri-ml), min(nx, ri+mu+1)):
            set(ri, ci, y[ri].diff(x[ci]))
    return packed


def check_transforms(fw, bw, symbs):
    """ Verify validity of a pair of forward and backward transformations

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    symbs: iterable of symbols
        the variables that are transformed
    """
    for f, b, y in zip(fw, bw, symbs):
        if f.subs(y, b) - y != 0:
            raise ValueError('Cannot prove correctness (did you set real=True?) fw: %s'
                             % str(f))
        if b.subs(y, f) - y != 0:
            raise ValueError('Cannot prove correctness (did you set real=True?) bw: %s'
                             % str(b))
