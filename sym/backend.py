# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import sys

from .util import banded_jacobian


def _DenseMatrix(be, *args, **kwargs):
        if len(args) == 1:
            return be.Matrix(len(args[0]), 1, args[0], **kwargs)
        else:
            nr, nc, elems = args
            return be.Matrix(nr, nc, elems, **kwargs)


class _Base(object):

    def __getattr__(self, key):
        return getattr(self.__sym_backend__, key)

    def banded_jacobian(self, exprs, dep, ml, mu):
        """ Wraps Matrix around result of .util.banded_jacobian """
        exprs = banded_jacobian(exprs, dep, ml, mu)
        return self.Matrix(ml+mu+1, len(dep), list(exprs.flat))


class _SymPy(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')
        from ._sympy_Lambdify import _Lambdify
        self.Lambdify = _Lambdify

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape, real=True)

    DenseMatrix = _DenseMatrix


class _SymPySymEngine(_SymPy):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')
        from symengine import Lambdify, symarray
        self.Lambdify = Lambdify
        #self.real_symarray = symarray

class _Diofant(_SymPy):

    def __init__(self):
        self.__sym_backend__ = __import__('diofant')
        from ._sympy_Lambdify import _Lambdify

        class DiofantLambdify(_Lambdify):
            def __init__(self, args, *exprs, backend='diofant', **kwargs):
                super().__init__(args, *exprs, backend=backend, **kwargs)

        self.Lambdify = DiofantLambdify

    DenseMatrix = _DenseMatrix

    __sym_backend_name__ = 'diofant'


class _SymEngine(_Base):

    _dummy_counter = [0]

    def __init__(self):
        self.__sym_backend__ = __import__('symengine')

    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    def Dummy(self):
        self._dummy_counter[0] += 1
        return self.Symbol('Dummy_'+str(self._dummy_counter[0] - 1))


class _PySym(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('pysym')

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    DenseMatrix = _DenseMatrix


class _SymCXX(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('symcxx').NameSpace()

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    DenseMatrix = _DenseMatrix


def Backend(name=None, envvar='SYM_BACKEND', default='sympy'):
    """ Backend for the underlying symbolic manipulation packages

    Parameters
    ----------
    name: str (default: None)
        Name of package e.g. 'sympy'
    envvar: str (default: 'SYM_BACKEND')
        name of environment variable to read name from (when name is ``None``)
    default: str
        name to use when the environment variable described by ``envvar`` is
        unset or empty (default: 'sympy')

    Examples
    --------
    >>> be = Backend('sympy')  # or e.g. 'symengine'
    >>> x, y = map(be.Symbol, 'xy')
    >>> exprs = [x + y + 1, x*y**2]
    >>> lmb = be.Lambdify([x, y], exprs)
    >>> import numpy as np
    >>> lmb(np.arange(6.0).reshape((3, 2)))
    array([[   2.,    0.],
           [   6.,   18.],
           [  10.,  100.]])

    """
    if name is None:
        name = os.environ.get(envvar, '') or default
    if isinstance(name, _Base):
        return name
    else:
        return Backend.backends[name]()


Backend.backends = {
    'sympy': _SymPy,
    'symengine': _SymEngine,
    'sympysymengine': _SymPySymEngine,  # uses selected parts from SymEngine to augment SymPy
    'pysym': _PySym,
    'symcxx': _SymCXX,
}

if sys.version_info[0] > 2:
    Backend.backends['diofant'] = _Diofant
