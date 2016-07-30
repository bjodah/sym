# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import array
from functools import reduce
from itertools import product
from operator import mul

from sympy import MatrixBase, sympify

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


def _flatten(mat):
    if isinstance(mat, MatrixBase):
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

    def __init__(self, args, exprs, real=True):
        self.out_shape = _get_shape(exprs)
        self.args_size = _size(args)
        self.out_size = reduce(mul, self.out_shape)
        self.args = _flatten(args)
        self.exprs = [sympify(expr) for expr in _flatten(exprs)]
        self.real = real

    def _evaluate_numerically(self, inp, out, out_offset):
        for idx in range(self.out_size):
            subsd = dict(zip(self.args, inp))
            out[out_offset + idx] = self.exprs[idx].xreplace(subsd)

    def __call__(self, inp, out=None, use_numpy=None):
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
                    out = array.array('Zd', [0j]*new_out_size)  # fails
                reshape_out = False
        else:
            if use_numpy:
                if out.dtype != (np.float64 if self.real else np.complex128):
                    raise TypeError("Output array is of incorrect type")
                if out.size < new_out_size:
                    raise ValueError("Incompatible size of output argument")
                if not out.flags['C_CONTIGUOUS']:
                    raise ValueError("Output array needs to be C-contiguous")
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

        flat_inp = _flatten(inp)
        for idx in range(nbroadcast):
            out_offset = idx*self.out_size
            local_inp = flat_inp[idx*self.args_size:(idx+1)*self.args_size]
            self._evaluate_numerically(local_inp, out, out_offset)

        if use_numpy and reshape_out:
            out = out.reshape(new_out_shape)
        elif reshape_out:
            raise NotImplementedError("array.array lacks shape, use NumPy")
        return out
