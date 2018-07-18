# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest
from .. import Backend
from . import AVAILABLE_BACKENDS


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_Matrix(key):
    be = Backend(key)
    x = be.Symbol('x')
    mat = be.Matrix(2, 2, [x, 1, x**2, 3])
    assert mat[0, 0] == x
    assert mat[0, 1] == 1
    assert mat[1, 0] == x**2
    assert mat[1, 1] == 3


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_Matrix_jacobian(key):
    be = Backend(key)
    x = be.Symbol('x')
    y = be.Symbol('y')
    a = be.Matrix(2, 1, [x+y, y*x**2])
    b = be.Matrix(2, 1, [x, y])
    J = a.jacobian(b)
    assert J[0, 0] == 1
    assert J[0, 1] == 1
    lmb = be.Lambdify([x, y], [J[1, 0], J[1, 1]])
    assert np.allclose(lmb(np.array([3, 5])), [2*3*5, 9])
