# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import math
import os
from functools import reduce
from operator import mul

import numpy as np  # Lambdify requires numpy


class _Lambdify(object):
    """ See docstring of symengine.Lambdify """
    # Note that this is a reimplementation of symengine.Lambdify.
    # If any modifications are to be made, they need to be implemented
    # in symengine.Lambdify *first*, and then reimplemented here.

    def __init__(self, args, *exprs, **kwargs):
        real = kwargs.pop('real', True)
        order = kwargs.pop('order', 'C')
        module = kwargs.pop('module', 'numpy')
        use_numba = kwargs.pop('use_numba', None)
        backend = kwargs.pop('backend', 'sympy')
        self._backend = __import__(backend)
        self.args = np.asanyarray(args)
        self.args_size = self.args.size
        self.exprs = tuple(np.asanyarray(expr) for expr in exprs)
        self.out_shapes = [expr.shape for expr in self.exprs]
        self.n_exprs = len(self.exprs)

        out_sizes, self.accum_out_sizes = [], []
        self.tot_out_size = 0
        for idx, shape in enumerate(self.out_shapes):
            out_sizes.append(reduce(mul, shape or (1,)))
            self.tot_out_size += out_sizes[idx]
        for i in range(self.n_exprs + 1):
            self.accum_out_sizes.append(0)
            for j in range(i):
                self.accum_out_sizes[i] += out_sizes[j]

        args_, outs_ = [], []
        self.order = order
        for arg in np.ravel(self.args, order=self.order):
            args_.append(self._backend.sympify(arg))

        for curr_expr in self.exprs:
            if curr_expr.ndim == 0:
                outs_.append(self._backend.sympify(curr_expr.item()))
            else:
                for e in np.ravel(curr_expr, order=self.order):
                    outs_.append(self._backend.sympify(e))

        self.real = real
        self.numpy_dtype = np.float64 if self.real else np.complex128
        if use_numba is None and module == 'numpy':
            _true = ('1', 't', 'true')
            use_numba = os.environ.get('SYM_USE_NUMBA', '0').lower() in _true
        elif use_numba and module != 'numpy':
            raise ValueError("Numba only available when using numpy as module.")
        self.use_numba = use_numba
        self._callback = _callback_factory(args_, outs_, module, self.numpy_dtype,
                                           self.order, self.use_numba, backend)

    def __call__(self, inp, out=None):
        try:
            inp = np.asanyarray(inp, dtype=self.numpy_dtype)
        except TypeError:
            inp = np.fromiter(inp, dtype=self.numpy_dtype)

        if inp.size < self.args_size or inp.size % self.args_size != 0:
            raise ValueError("Broadcasting failed (input/arg size mismatch)")
        nbroadcast = inp.size // self.args_size

        if inp.ndim > 1:
            if self.args_size > 1:
                if self.order == 'C':
                    if inp.shape[inp.ndim-1] != self.args_size:
                        raise ValueError(("C order implies last dim (%d) == len(args)"
                                          " (%d)") % (inp.shape[inp.ndim-1], self.args_size))
                    extra_dim = inp.shape[:inp.ndim-1]
                elif self.order == 'F':
                    if inp.shape[0] != self.args_size:
                        raise ValueError("F order implies first dim (%d) == len(args) (%d)"
                                         % (inp.shape[0], self.args_size))
                    extra_dim = inp.shape[1:]
            else:
                extra_dim = inp.shape
        else:
            if nbroadcast > 1 and inp.ndim == 1:
                extra_dim = (nbroadcast,)  # special case
            else:
                extra_dim = ()
        extra_left = extra_dim if self.order == 'C' else ()
        extra_right = () if self.order == 'C' else extra_dim
        new_out_shapes = [extra_left + out_shape + extra_right
                          for out_shape in self.out_shapes]

        new_tot_out_size = nbroadcast * self.tot_out_size
        if out is None:
            out = np.empty(new_tot_out_size, dtype=self.numpy_dtype, order=self.order)
        else:
            if out.size < new_tot_out_size:
                raise ValueError("Incompatible size of output argument")
            if out.ndim > 1:
                if len(self.out_shapes) > 1:
                    raise ValueError("output array with ndim > 1 assumes one output")
                out_shape, = self.out_shapes
                if self.order == 'C':
                    if not out.flags['C_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be C-contiguous")
                    if out.shape[-len(out_shape):] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
                elif self.order == 'F':
                    if not out.flags['F_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be F-contiguous")
                    if out.shape[:len(out_shape)] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
            else:
                if not out.flags['F_CONTIGUOUS']:  # or C_CONTIGUOUS (ndim <= 1)
                    raise ValueError("Output array need to be contiguous")
            if not out.flags['WRITEABLE']:
                raise ValueError("Output argument needs to be writeable")
            out = out.ravel(order=self.order)

        inp = np.ascontiguousarray(inp.ravel(order=self.order))
        res_exprs = self._callback(inp if nbroadcast == 1 else inp.reshape(
            (nbroadcast, inp.size//nbroadcast)
        ))
        assert len(res_exprs) == self.tot_out_size
        for idx, res in enumerate(res_exprs):
            out.flat[idx::self.tot_out_size] = res

        if self.order == 'C':
            out = out.reshape((nbroadcast, self.tot_out_size), order='C')
            result = [
                out[:, self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1]].reshape(
                    new_out_shapes[idx], order='C') for idx in range(self.n_exprs)
            ]
        elif self.order == 'F':
            out = out.reshape((self.tot_out_size, nbroadcast), order='F')
            result = [
                out[self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1], :].reshape(
                    new_out_shapes[idx], order='F') for idx in range(self.n_exprs)
            ]
        if self.n_exprs == 1:
            return result[0]
        else:
            return result


def _callback_factory(args, flat_exprs, module, dtype, order, use_numba=False, backend='sympy'):
    if module == 'numpy':
        TRANSLATIONS = {
            "acos": "arccos",
            "acosh": "arccosh",
            "arg": "angle",
            "asin": "arcsin",
            "asinh": "arcsinh",
            "atan": "arctan",
            "atan2": "arctan2",
            "atanh": "arctanh",
            "ceiling": "ceil",
            "E": "e",
            "im": "imag",
            "ln": "log",
            "Mod": "mod",
            "oo": "inf",
            "re": "real",
            "SparseMatrix": "array",
            "ImmutableSparseMatrix": "array",
            "Matrix": "array",
            "MutableDenseMatrix": "array",
            "ImmutableDenseMatrix": "array",
        }
        Printer = __import__(backend + '.printing.lambdarepr',
                             fromlist=['NumPyPrinter']).NumPyPrinter

        def lambdarepr(_x):
            return Printer().doprint(_x)
    else:
        lambdarepr = __import__(backend + '.printing.lambdarepr',
                                fromlist=['lambdarepr']).lambdarepr
        if module == 'mpmath':
            TRANSLATIONS = {
                "Abs": "fabs",
                "elliptic_k": "ellipk",
                "elliptic_f": "ellipf",
                "elliptic_e": "ellipe",
                "elliptic_pi": "ellippi",
                "ceiling": "ceil",
                "chebyshevt": "chebyt",
                "chebyshevu": "chebyu",
                "E": "e",
                "I": "j",
                "ln": "log",
                # "lowergamma":"lower_gamma",
                "oo": "inf",
                # "uppergamma":"upper_gamma",
                "LambertW": "lambertw",
                "MutableDenseMatrix": "matrix",
                "ImmutableDenseMatrix": "matrix",
                "conjugate": "conj",
                "dirichlet_eta": "altzeta",
                "Ei": "ei",
                "Shi": "shi",
                "Chi": "chi",
                "Si": "si",
                "Ci": "ci"
            }

        elif module == 'sympy':
            TRANSLATIONS = {}
        else:
            raise NotImplementedError("Lambdify does not yet support %s" % module)

    mod = __import__(backend)
    ordering = '..., %d'  # if order == 'C' else '%d, ...'
    indices = [mod.Symbol(ordering % i) for i in range(len(args))]
    dummy_subs = dict(zip(args, [mod.Symbol('x[%s]' % i) for i in indices]))
    dummified = [expr.xreplace(dummy_subs) for expr in flat_exprs]
    estr = lambdarepr(dummified)

    mod = __import__(module)
    namespace = mod.__dict__.copy()

    # e.g. NumPyPrinter incomplete: https://github.com/sympy/sympy/issues/11023
    # we need to read translations from lambdify
    for k, v in TRANSLATIONS.items():
        namespace[k] = namespace[v]

    if module != 'mpmath':
        namespace['Abs'] = abs

    namespace['numpy'] = np
    namespace['math'] = math
    # namespace['_transpose'] = _transpose

    funcstr = 'lambda x: %s' % estr
    func = eval(funcstr, namespace)
    if use_numba:
        from numba import jit
        func = jit(func)
    if module == 'numpy':
        def wrapper(x):
            arg = np.atleast_1d(np.asanyarray(x, dtype=dtype))
            res = func(arg)
            return res
    else:
        wrapper = func
    wrapper.__doc__ = estr

    return wrapper
