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
        from symengine import Lambdify
        self.Lambdify = Lambdify


class _Diofant(_SymPy):

    def __init__(self):
        self.__sym_backend__ = __import__('diofant')
        from ._sympy_Lambdify import _Lambdify

        class DiofantLambdify(_Lambdify):
            def __init__(self, args, *exprs, **kwargs):
                kwargs['backend'] = 'diofant'
                super().__init__(args, *exprs, **kwargs)

        self.Lambdify = DiofantLambdify

    DenseMatrix = _DenseMatrix

    __sym_backend_name__ = 'diofant'


class _SymEngine(_Base):

    _dummy_counter = [0]

    def __init__(self):
        self.__sym_backend__ = __import__('symengine')
        # cse isn't in any symengine release yet; only in dev version
        # this will allow backend use with older symengine versions,
        # failing gracefully only if cse is invoked
        self._cse = getattr(self.__sym_backend__, 'cse', None)
        self._ccode = self.__sym_backend__.lib.symengine_wrapper.ccode

    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    def Dummy(self):
        self._dummy_counter[0] += 1
        return self.Symbol('Dummy_'+str(self._dummy_counter[0] - 1))

    def ccode(self, *args, **kwargs):
        # need to wrap, as ccode does not exist within root
        # symengine namespace
        return self.__sym_backend__.lib.symengine_wrapper.ccode(*args, **kwargs)

    def numbered_symbols(self, prefix='x', cls=None, start=0, exclude=None, *args):
        exclude = set(exclude or [])
        if cls is None:
            cls = self.Symbol

        while True:
            name = '%s%s' % (prefix, start)
            s = cls(name, *args)
            if s not in exclude:
                yield s
            start += 1

    def cse(self, exprs, symbols=None):
        # symengine's cse, but augmented with custom cse symbols ala sympy
        if self._cse is None:
            raise NotImplementedError("CSE not yet supported in symengine version %s" %
                                      self.__sym_backend__.__version__)
        if symbols is None:
            symbols = self.numbered_symbols()
        else:
            symbols = iter(symbols)

        old_repl, old_reduced = self._cse(exprs)
        if not old_repl:
            return old_repl, old_reduced
        old_cse_symbols, old_cses = zip(*old_repl)
        cse_symbols = [next(symbols) for _ in range(len(old_cse_symbols))]
        subsd = dict(zip(old_cse_symbols, cse_symbols))

        cses = [c.xreplace(subsd) for c in old_cses]
        reduced = [e.xreplace(subsd) for e in old_reduced]
        return list(zip(cse_symbols, cses)), reduced


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
