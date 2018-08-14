# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np


def banded_jacobian(y, x, ml, mu):
    """ Calculates a banded version of the jacobian

    Compatible with the format requested by
    :func:`scipy.integrate.ode` (for SciPy >= v0.15).

    Parameters
    ----------
    y: array_like of expressions
    x: array_like of symbols
    ml: int
        number of lower bands
    mu: int
        number of upper bands

    Returns
    -------
    2D array of shape ``(1+ml+mu, len(y))``

    """
    ny = len(y)
    nx = len(x)
    packed = np.zeros((mu+ml+1, nx), dtype=object)

    def set(ri, ci, val):
        packed[ri-ci+mu, ci] = val

    for ri in range(ny):
        for ci in range(max(0, ri-ml), min(nx, ri+mu+1)):
            set(ri, ci, y[ri].diff(x[ci]))
    return packed


def sparse_jacobian_csc(y, x):
    """ Calculates a compressed sparse column (CSC)
        version of the jacobian

    Parameters
    ----------
    y: array_like of expressions
    x: array_like of symbols

    Returns
    -------
    jac_exprs: flattened list of expressions for nonzero entries of dy/dx in column-major order
    colptrs: list of length ``len(y) + 1``, where ``jac_exprs[colptrs[i]:colptrs[i+1]]`` are
             the nonzero entries of column ``i`` in ``dy/dx``
    rowvals: list of length ``len(jac_exprs``, denoting the row index in ``dy/dx`` for each
             entry in ``jac_exprs``
    """
    n = len(x)
    try:
        # backends with free_symbols and hashable symbols
        idx = dict(zip(x, range(n)))
        cols = [[] for _ in range(n)]
        fs = [yi.free_symbols for yi in y]
        for i, fi in enumerate(fs):
            for j in sorted(list(map(idx.get, fi))):
                cols[j].append(i)
    except (AttributeError, TypeError):
        # backends without free_symbols or with unhashable symbols
        cols = [[i for i, yi in enumerate(y) if yi.has(xj)] for xj in x]

    rowvals = [i for col in cols for i in col]
    colptrs = np.cumsum([0] + list(map(len, cols))).astype(int)
    jac_exprs = [y[i].diff(xj) for j, xj in enumerate(x) for i in cols[j]]
    return jac_exprs, colptrs, rowvals


def sparse_jacobian_csr(y, x):
    """ Calculates a compressed sparse row (CSR)
        version of the jacobian

    Parameters
    ----------
    y: array_like of expressions
    x: array_like of symbols

    Returns
    -------
    jac_exprs: flattened list of expressions for nonzero entries of dy/dx in row-major order
    rowptrs: list of length ``len(y) + 1``, where ``jac_exprs[colptrs[i]:colptrs[i+1]]`` are
             the nonzero entries of row ``i`` in ``dy/dx``
    colvals: list of length ``len(jac_exprs``, denoting the column index in ``dy/dx`` for each
             entry in ``jac_exprs``
    """
    n = len(x)
    try:
        # backends with free_symbols and hashable symbols
        idx = dict(zip(x, range(n)))
        rows = [sorted(list(map(idx.get, yi.free_symbols))) for yi in y]
    except (AttributeError, TypeError):
        # backends without free_symbols or with unhashable symbols
        rows = [[j for j, xj in enumerate(x) if yi.has(xj)] for yi in y]

    colvals = [j for row in rows for j in row]
    rowptrs = np.cumsum([0] + list(map(len, rows))).astype(int)
    jac_exprs = [yi.diff(x[j]) for i, yi in enumerate(y) for j in rows[i]]
    return jac_exprs, rowptrs, colvals


def check_transforms(fw, bw, symbs):
    """ Verify validity of a pair of forward and backward transformations

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    symbs: iterable of symbols
        the variables that are transformed
    """
    for f, b, y in zip(fw, bw, symbs):
        if f.subs(y, b) - y != 0:
            raise ValueError('Cannot prove correctness (did you set real=True?) fw: %s'
                             % str(f))
        if b.subs(y, f) - y != 0:
            raise ValueError('Cannot prove correctness (did you set real=True?) bw: %s'
                             % str(b))


def _map2(cb, iterable):
    if cb is None:  # identity function is assumed
        return iterable
    else:
        return map(cb, iterable)


def _map2l(cb, iterable):  # Py2 type of map in Py3
    return list(_map2(cb, iterable))


def linear_rref(A, b, backend):
    """ Transform a linear system to reduced row-echelon form

    Transforms both the matrix and right-hand side of a linear
    system of equations to reduced row echelon form

    Parameters
    ----------
    A: Matrix-like
        iterable of rows
    b: iterable

    Returns
    -------
    A', b' - transformed versions

    """
    try:
        b = b.as_mutable()
    except:
        b = backend.MutableMatrix(b)

    Aug = A.col_insert(A.cols, backend.eye(A.rows))
    rAug, pivots = Aug.rref()
    colidxs = [i for i in pivots if i < A.cols]
    b = backend.Matrix(rAug[:, A.cols:]*b)
    return rAug[:, :A.cols], b, colidxs
