# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import array
from functools import reduce
from itertools import product
from operator import mul

# Note that this is a reimplementation of symengine.Lambdify


def _size(n):
    try:
        return n.size
    except AttributeError:
        return len(n)  # e.g. array.array


def _get_shape_nested(ndarr):
    # no checking of shape consistency is done
    if isinstance(ndarr, (list, tuple)):
        return (len(ndarr),) + _get_shape_nested(ndarr[0])
    else:
        try:
            return (len(ndarr),)
        except TypeError:
            return ()


def _get_shape(ndarr):
    try:
        return ndarr.shape
    except AttributeError:
        return _get_shape_nested(ndarr)


def _nested_getitem(ndarr, indices):
    if len(indices) == 0:
        return ndarr
    else:
        return _nested_getitem(ndarr[indices[0]], indices[1:])


def _all_indices_from_shape(shape):
    return product(*(range(dim) for dim in shape))


def _ravel_nested(ndarr):
    return [_nested_getitem(ndarr, indices) for indices in
            _all_indices_from_shape(_get_shape(ndarr))]


def _ravel(ndarr):
    try:
        return ndarr.ravel()
    except AttributeError:
        return _ravel_nested(ndarr)


def _flatten(mat, backend):
    if isinstance(mat, backend.MatrixBase):
        _mat = []
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                _mat.append(mat[ri, ci])
        return _mat  # flattened
    else:
        return _ravel(mat)


class _Lambdify(object):
    """ See docstring of symengine.Lambdify """
    # Note that this is a reimplementation of symengine.Lambdify.
    # If any modifications are to be made, they need to be implemented
    # in symengine.Lambdify *first*, and then reimplemented here.

    def __init__(self, args, exprs, real=True, module='numpy',
                 use_numba=None, backend='sympy'):
        self._backend = __import__(backend)
        self.out_shape = _get_shape(exprs)
        self.args_size = _size(args)
        self.out_size = reduce(mul, self.out_shape)
        self.args = _flatten(args, self._backend)
        self.exprs = [self._backend.sympify(expr)
                      for expr in _flatten(exprs, self._backend)]
        if self.out_size != len(self.exprs):
            raise ValueError("Sanity-check failed: bug in %s" % self.__class__)
        self.real = real
        if use_numba is None and module == 'numpy':
            _true = ('1', 't', 'true')
            use_numba = os.environ.get('SYM_USE_NUMBA', '0').lower() in _true
        elif use_numba and module != 'numpy':
            raise ValueError("Numba only available when using numpy as module.")
        self.use_numba = use_numba
        self._callbacks = [callback_factory(self.args, expr, module,
                                            self.use_numba, backend)
                           for expr in self.exprs]

    def _evaluate_xreplace(self, inp, out, out_offset):
        for idx in range(self.out_size):
            subsd = dict(zip(self.args, inp))
            out[out_offset + idx] = self.exprs[idx].xreplace(subsd)

    def __call__(self, inp, out=None, use_numpy=None):
        if hasattr(inp, 'shape'):
            inp_shape = inp.shape
        else:
            inp = list(inp)
            inp_shape = _get_shape(inp)
        inp_size = reduce(mul, inp_shape)
        if inp_size % self.args_size != 0:
            raise ValueError("Broadcasting failed")
        nbroadcast = inp_size // self.args_size
        if nbroadcast > 1 and self.args_size == 1 and inp_shape[-1] != 1:
            inp_shape = inp_shape + (1,)  # Implicit reshape
        new_out_shape = inp_shape[:-1] + self.out_shape
        new_out_size = nbroadcast * self.out_size

        if use_numpy is None:
            try:
                import numpy as np
            except ImportError:
                use_numpy = False  # we will use array.array instead
            else:
                use_numpy = True
        elif use_numpy is True:
            import numpy as np

        if out is None:
            # allocate output container
            if use_numpy:
                out = np.empty(new_out_size, dtype=np.float64 if
                               self.real else np.complex128)
                reshape_out = len(new_out_shape) > 1
            else:
                if self.real:
                    out = array.array('d', [0]*new_out_size)
                else:
                    raise NotImplementedError("Zd unsupported in array.array")
                reshape_out = False
        else:
            if use_numpy:
                if out.dtype != (np.float64 if self.real else np.complex128):
                    raise TypeError("Output array is of incorrect type")
                if out.size < new_out_size:
                    raise ValueError("Incompatible size of output argument")
                for idx, ln in enumerate(out.shape[-len(self.out_shape)::-1]):
                    if ln < self.out_shape[-idx]:
                        raise ValueError("Incompatible shape of output array")
                if not out.flags['WRITEABLE']:
                    raise ValueError("Output argument needs to be writeable")
                if out.ndim > 1:
                    out = out.ravel()
                    reshape_out = True
                else:
                    # The user passed a 1-dimensional output argument,
                    # we trust the user to do the right thing.
                    reshape_out = False
            else:
                reshape_out = False

        if use_numpy:
            if reshape_out:
                out = out.reshape(new_out_shape)
            for idx, callback in zip(_all_indices_from_shape(self.out_shape), self._callbacks):
                out[(Ellipsis,) + idx] = callback(inp)
        else:
            flat_inp = _flatten(inp)
            for idx in range(nbroadcast):
                out_offset = idx*self.out_size
                local_inp = flat_inp[idx*self.args_size:(idx+1)*self.args_size]
                self._evaluate_xreplace(local_inp, out, out_offset)  # slow, consider lambdastr

        if not use_numpy and reshape_out:
            raise NotImplementedError("array.array lacks shape, use NumPy")
        return out


def callback_factory(args, expr, module, use_numba=False, backend='sympy'):
    if module == 'numpy':
        TRANSLATIONS = __import__(backend + '.utilities.lambdify',
                                  fromlist=['NUMPY_TRANSLATIONS']).NUMPY_TRANSLATIONS
        Printer = __import__(backend + '.printing.lambdarepr',
                             fromlist=['NumPyPrinter']).NumPyPrinter

        def lambdarepr(_x):
            return Printer().doprint(_x)
    else:
        lambdarepr = __import__(backend + '.printing.lambdarepr',
                                fromlist=['lambdarepr']).lambdarepr
        if module == 'mpmath':
            TRANSLATIONS = __import__(backend + '.utilities.lambdify',
                                      fromlist=['MPMATH_TRANSLATIONS']).MPMATH_TRANSLATIONS
        elif module == 'sympy':
            TRANSLATIONS = {}
        else:
            raise NotImplementedError("Lambdify does not yet support %s" % module)

    mod = __import__(backend)
    x = mod.IndexedBase('x')
    indices = [mod.Symbol('..., %d' % i) for i in range(len(args))]
    dummy_subs = dict(zip(args, [x[i] for i in indices]))
    dummified = expr.xreplace(dummy_subs)
    estr = lambdarepr(dummified)

    mod = __import__(module)
    namespace = mod.__dict__.copy()

    # e.g. NumPyPrinter incomplete: https://github.com/sympy/sympy/issues/11023
    # we need to read translations from lambdify
    for k, v in TRANSLATIONS.items():
        namespace[k] = namespace[v]

    if module != 'mpmath':
        namespace['Abs'] = abs

    func = eval('lambda x: %s' % estr, namespace)
    if use_numba:
        from numba import njit
        func = njit(func)
    if module == 'numpy':
        def wrapper(x):
            return func(mod.asarray(x, dtype=mod.float64))
    else:
        wrapper = func
    wrapper.__doc__ = estr

    return wrapper
