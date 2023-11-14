# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from .._sympy_Lambdify import _callback_factory

from sympy import symbols, atan
import numpy as np
import pytest

try:
    import numba
except ImportError:
    numba = None


def test_callback_factory():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C')
    inp = np.array([17, 1])
    ref = 17 + np.arctan(1)
    assert np.allclose(cb(inp), ref)


def test_callback_factory__broadcast():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C')
    inp = np.array([[17, 1], [18, 2]])
    ref = [17 + np.arctan(1), 18 + np.arctan(2)]
    assert np.allclose(cb(inp), ref)

    inp2 = np.array([
        [[17, 1], [18, 2]],
        [[27, 21], [28, 22]]
    ])
    ref2 = [
        [17 + np.arctan(1), 18 + np.arctan(2)],
        [27 + np.arctan(21), 28 + np.arctan(22)]
    ]
    assert np.allclose(cb(inp2), ref2)


@pytest.mark.skipif(numba is None, reason='numba not available')
def test_callback_factory__numba():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C', use_numba=True)
    n = 500
    inp = np.empty((n, 2))
    inp[:, 0] = np.linspace(0, 1, n)
    inp[:, 1] = np.linspace(-10, 10, n)
    assert np.allclose(cb(inp), inp[:, 0] + np.arctan(inp[:, 1]))


def test_callback_factory__user_defined_function():
    from sympy import Function
    class logsumexp(Function):
        nargs = None # sy.S.Naturals0

        def __new__(cls, *args):
            if len(args) % 2:
                raise ValueError("even number of args expected")
            elif len(args) == 0:
                raise ValueError("expected finite number of arguments")
            return Function.__new__(cls, *args)

        def _numpycode(self, printer):
            return "%s([%s], b=numpy.array([%s]))" % (
                printer._module_format("scipy.special.logsumexp"),
                ', '.join(map(printer._print, self.args[::2])),
                ', '.join(map(printer._print, self.args[1::2])),
            )

    import scipy.special

    def _test1():
        args = x, y, z, u, v, w = symbols('x y z u v w')
        expr = x+y+z + logsumexp(x, u, y, v, z, w)
        cb = _callback_factory(args, [expr, expr-z], 'numpy', np.float64, 'C', use_numba=False)
        x_, y_, z_, u_, v_, w_ = inp = np.array([-8, -1, 2, .13, 17, 42.])
        ref = [x_+y_+z_+scipy.special.logsumexp([x_,y_,z_], b=[u_,v_,w_])]
        ref += [ref[0]-z_]
        ref2 = [x_+y_+z_+np.log(u_*np.exp(x_) + v_*np.exp(y_) + w_*np.exp(z_))]
        ref2 += [ref2[0] - z_]
        assert np.allclose(ref, ref2)
        res = cb(inp)
        assert np.allclose(res, ref)

    def _test2():
        args = x, y, u, v = symbols('x y u v')
        expr = x + logsumexp(x, u, y, v)
        cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C', use_numba=False)
        n = 500
        inp = np.empty((n, 4))
        inp[:, 0] = np.linspace(0, 1, n)
        inp[:, 1] = np.linspace(-10, 10, n)
        inp[:, 2] = np.logspace(-10, 2, n)
        inp[:, 3] = np.logspace(3, -7, n)
        x_ = inp[:, 0]
        ref = x_ + scipy.special.logsumexp(inp[:, :2], b=inp[:, 2:])
        res = cb(inp)
        assert np.allclose(res, ref)

    # def _test3():
    #     args = x, y = symbols('x y')
    #     expr = x + logsumexp(x, 2, y, 3)
    #     cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C', use_numba=False)
    #     n = 500
    #     inp = np.empty((n, 2))
    #     inp[:, 0] = np.linspace(0, 1, n)
    #     inp[:, 1] = np.linspace(-10, 10, n)
    #     x_ = inp[:, 0]
    #     ref = x_ + scipy.special.logsumexp(inp, b=[2,3])
    #     res = cb(inp)
    #     assert np.allclose(res, ref)

    _test1()
    _test2()
    #_test3()
