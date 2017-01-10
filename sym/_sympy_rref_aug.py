"""
The rref method in matrices in SymPy do not yet support ``aug`` kwarg. This is a "backport".
See https://github.com/sympy/sympy/pull/12035
"""

from sympy.matrices.matrices import _iszero, _find_reasonable_pivot
from sympy.simplify import simplify as _simplify
from types import FunctionType


def rref_aug(self, iszerofunc=_iszero, simplify=False, aug=None):
    """Return reduced row-echelon form of matrix and indices of pivot vars.

    To simplify elements before finding nonzero pivots set simplify=True
    (to use the default SymPy simplify function) or pass a custom
    simplify function.

    Parameters
    ==========
    iszerofunc : callable
    simplify : bool or callable
    aug : MutableMatrix (optional)
        Optional extra columns to form an augmented matrix.
        The row operations performed will also be performed
        on ``aug`` if given.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), [0, 1])
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    [0, 1]
    """
    simpfunc = simplify if isinstance(
        simplify, FunctionType) else _simplify
    # pivot: index of next row to contain a pivot
    pivot, r = 0, self.as_mutable()
    # pivotlist: indices of pivot variables (non-free)
    pivotlist = []
    for i in range(r.cols):
        if pivot == r.rows:
            break
        if simplify:
            r[pivot, i] = simpfunc(r[pivot, i])

        pivot_offset, pivot_val, assumed_nonzero, newly_determined = _find_reasonable_pivot(
            r[pivot:, i], iszerofunc, simpfunc)
        # `_find_reasonable_pivot` may have simplified
        # some elements along the way.  If they were simplified
        # and then determined to be either zero or non-zero for
        # sure, they are stored in the `newly_determined` list
        for (offset, val) in newly_determined:
            r[pivot + offset, i] = val

        # if `pivot_offset` is None, this column has no
        # pivot
        if pivot_offset is None:
            continue

        # swap the pivot column into place
        pivot_pos = pivot + pivot_offset
        r.row_swap(pivot, pivot_pos)
        if aug is not None:
            aug.row_swap(pivot, pivot_pos)

        r.row_op(pivot, lambda x, _: x / pivot_val)
        if aug is not None:
            aug.row_op(pivot, lambda x, _: x / pivot_val)

        for j in range(r.rows):
            if j == pivot:
                continue
            pivot_val = r[j, i]
            r.zip_row_op(j, pivot, lambda x, y: x - pivot_val * y)
            if aug is not None:
                aug.zip_row_op(j, pivot, lambda x, y: x - pivot_val * y)
        pivotlist.append(i)
        pivot += 1
    return self._new(r), pivotlist
