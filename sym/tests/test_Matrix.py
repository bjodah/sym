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


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_Matrix_jacobian__2(key):
    be = Backend(key)
    sin, cos = be.sin, be.cos
    x = be.Symbol('x', real=True)
    y = be.Symbol('y', real=True)
    e0 = x*y - 5
    e1 = ((y**4*sin(5*x) - 1)/y)/y**3
    #e1 = y**-3 * be.UnevaluatedExpr(be.Mul(y**-1, y**4 * sin(5*x) -1, evaluate=False))
    a = be.Matrix(2, 1, [e0, e1])
    b = be.Matrix(2, 1, [x, y])
    J = a.jacobian(b) #.doit()
    assert J[0, 0] == y
    assert J[0, 1] == x
    ref10 = 5*cos(5*x)
    ref11 = (4*y**2*sin(5*x) - (y**4*sin(5*x) - 1)/y**2)/y**3 - 3*(y**4*sin(5*x) - 1)/y**5
    # assert J[1, 0] == ref10
    # assert J[1, 1] == ref11
    lmb = be.Lambdify([x, y], [J[1, 0], J[1, 1]])
    xvec = np.array([3, 5])
    subsd = {x: 3, y: 5}
    ref_row = list(map(float, [ref10.xreplace(subsd), ref11.xreplace(subsd)]))
    assert np.allclose(lmb(xvec), ref_row)
