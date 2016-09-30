# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import sys

from .util import banded_jacobian
from ._sympy_Lambdify import create_Lambdify


class _Base(object):

    __sym_backend_name__ = None

    def __init__(self):
        self.__sym_backend__ = __import__(self.__sym_backend_name__)
        self._post_init()

    def _post_init(self):
        pass

    def __getattr__(self, key):
        return getattr(self.__sym_backend__, key)

    def banded_jacobian(self, exprs, dep, ml, mu):
        """ Wraps Matrix around result of .util.banded_jacobian """
        exprs = banded_jacobian(exprs, dep, ml, mu)
        return self.Matrix(ml+mu+1, len(dep), list(exprs.flat))


class _SymPyBase(_Base):

    def _post_init(self):
        _trans = __import__(self.__sym_backend_name__ + '.utilities.lambdify',
                            fromlist=['NUMPY_TRANSLATIONS'])
        _print = __import__(self.__sym_backend_name__ + '.printing.lambdarepr',
                            fromlist=['NumPyPrinter'])
        self.Lambdify = create_Lambdify(self.MatrixBase, self.sympify,
                                        _print.NumPyPrinter,
                                        self.IndexedBase, self.Symbol,
                                        _trans.NUMPY_TRANSLATIONS)[0]

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape, real=True)


class _SymPy(_SymPyBase):

    __sym_backend_name__ = 'sympy'


class _Diofant(_SymPyBase):

    __sym_backend_name__ = 'diofant'


class _SymEngine(_Base):

    __sym_backend_name__ = 'symengine'

    _dummy_counter = [0]

    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    def Dummy(self):
        self._dummy_counter[0] += 1
        return self.Symbol('Dummy_'+str(self._dummy_counter[0] - 1))


class _PySym(_Base):

    __sym_backend_name__ = 'pysym'

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)


class _SymCXX(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('symcxx').NameSpace()
        self._post_init()

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
    'pysym': _PySym,
    'symcxx': _SymCXX,
}

if sys.version_info[0] > 2:
    Backend.backends['diofant'] = _Diofant
