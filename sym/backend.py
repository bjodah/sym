# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import numpy as np


# ONLY_WHITELIST_ATTRS = os.environ.get('SYM_ONLY_WHITELIST_ATTRS', '0') == '1'
# WHITE_LIST = 'Symbol Matrix symarray Dummy Lambdify '

class _Base(object):

    def __getattr__(self, key):
        if key == '__sym_backend__':
            return self.__dict__[key]
        # if ONLY_WHITELIST_ATTRS and (
        #         key not in self.__dict__ and
        #         key not in WHITE_LIST):
        #     raise AttributeError
        return self.__dict__.get(key, getattr(self.__sym_backend__, key))

    def __setattr__(self, key, value):
        if key == '__sym_backend__':
            self.__dict__[key] = value
        # elif key in self.__dict__:
        #     self.__dict__[key] = value
        else:
            setattr(self.__sym_backend__, key, value)


class _SymPy(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')

    def Lambdify(self, args, exprs):
        # Lambdify not in SymPy
        _lmb = self.lambdify(args, exprs, modules=[
            {'ImmutableMatrix': np.array}, 'numpy'])
        def cb(args, out=None):
            result = _lmb(*args)
            if out is not None:
                out[...] = result[...]
                return out
            else:
                return result
        return cb

    def symarray(self, prefix, shape, Symbol=None, real=True):
        """ Creates an nd-array of symbols

        Parameters
        ----------
        prefix: str
        shape: tuple
        Symbol: callable
            (defualt :func:`Symbol`)
        """
        # see https://github.com/sympy/sympy/pull/9939
        # will be in SymPy > 0.7.6.1
        arr = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            arr[index] = self.Symbol('%s_%s' % (prefix, '_'.join(map(str, index))))
        return arr


class _SymEngine(_Base):

    _dummy_counter = [0]

    def __init__(self):
        self.__sym_backend__ = __import__('symengine')


    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)  # MutableDenseMatrix in SymPy

    symarray = _SymPy.__dict__['symarray']

    def Dummy(self):
        self._dummy_counter[0] += 1
        return self.Symbol('Dummy_'+str(self._dummy_counter[0] - 1))


class _PySym(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('pysym')


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

    """
    if name is None:
        name = os.environ.get(envvar, '') or default
    return Backend.backends[name]()

Backend.backends = {
    'sympy': _SymPy,
    'symengine': _SymEngine,
    'pysym': _PySym,
}
