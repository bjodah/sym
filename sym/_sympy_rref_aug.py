"""
The rref method in matrices in SymPy do not yet support ``aug`` kwarg. This is a "backport".
See https://github.com/sympy/sympy/pull/12035
"""

from sympy.matrices.matrices import _iszero
from sympy.simplify import simplify as _simplify
from types import FunctionType
from sympy.core.numbers import Integer, Float
from sympy.core.singleton import S


def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    """ Find the lowest index of an item in `col` that is
    suitable for a pivot.  If `col` consists only of
    Floats, the pivot with the largest norm is returned.
    Otherwise, the first element where `iszerofunc` returns
    False is used.  If `iszerofunc` doesn't return false,
    items are simplified and retested until a suitable
    pivot is found.

    Returns a 4-tuple
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    where pivot_offset is the index of the pivot, pivot_val is
    the (possibly simplified) value of the pivot, assumed_nonzero
    is True if an assumption that the pivot was non-zero
    was made without being probed, and newly_determined are
    elements that were simplified during the process of pivot
    finding."""

    newly_determined = []
    col = list(col)
    # a column that contains a mix of floats and integers
    # but at least one float is considered a numerical
    # column, and so we do partial pivoting
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            # just because iszerofunc returned True, doesn't
            # mean the value is numerically zero.  Make sure
            # to replace all entries with numerical zeros
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (iszerofunc directly)
    possible_zeros = []
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        # is someone wrote a custom iszerofunc, it may return
        # BooleanFalse or BooleanTrue instead of True or False,
        # so use == for comparison instead of `is`
        if is_zero is False:
            # we found something that is definitely not zero
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    # by this point, we've found no certain non-zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc after simplify)
    # we haven't found any for-sure non-zeros, so
    # go through the elements iszerofunc couldn't
    # make a determination about and opportunistically
    # simplify to see if we find something
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero is True or is_zero is False:
            newly_determined.append((i, simped))
        if is_zero is False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # after simplifying, some things that were recognized
    # as zeros might be zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # some expressions fail to simplify to zero, but
    # `.equals(0)` evaluates to True.  As a last-ditch
    # attempt, apply `.equals` to these expressions
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            # `.iszero` may return False with
            # an implicit assumption (e.g., `x.equals(0)`
            # when `x` is a symbol), so only treat it
            # as proved when `.equals(0)` returns True
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)

    # at this point there is nothing that could definitely
    # be a pivot.  To maintain compatibility with existing
    # behavior, we'll assume that an illdetermined thing is
    # non-zero.  We should probably raise a warning in this case
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)


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
