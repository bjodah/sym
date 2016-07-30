# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os


class _Base(object):

    def __getattr__(self, key):
        return getattr(self.__sym_backend__, key)


class _SymPy(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')
        from ._sympy_Lambdify import _Lambdify
        self.Lambdify = _Lambdify

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape, real=True)


class _SymEngine(_Base):

    _dummy_counter = [0]

    def __init__(self):
        self.__sym_backend__ = __import__('symengine')

    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)  # MutableDenseMatrix in SymPy

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


class _SymCXX(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('symcxx').NameSpace()

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)


def Backend(name=None, envvar='SYM_BACKEND', default='sympy'):
    """ Backend for the underlying symbolic manipulation pacages

    Parameters
    ----------
    name: str (default: None)
        Name of package e.g. 'sympy'
    envvar: str (default: 'SYM_BACKEND')
        name of environmentvariable to read name from (when name is ``None``)
    default: str
        name to use when the environment variable described by ``envvar`` is
        unset or empty (default: 'sympy')

    Examples
    --------
    >>> be = Backend('sympy')
    >>> x, y = map(be.Symbol, 'xy')
    >>> exprs = [x + y + 1, x*y**2]
    >>> lmb = be.Lambdify([x, y], exprs)
    >>> import numpy as np
    >>> lmb(np.array([2.0, 3.0]))
    array([  6.,  18.])

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
    'pysym': _PySym,
    'symcxx': _SymCXX,
}
