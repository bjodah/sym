# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
from ..util import linear_rref
from .. import Backend


@pytest.mark.parametrize('key', ['sympy'])
def test_linear_rref(key):
    be = Backend(key)
    A1 = be.Matrix([[8, 8, 4], [8, 8, 5], [2, 4, 4]])
    b1 = [80, 88, 52]
    A2, b2, colidxs = linear_rref(A1, b1, backend=be)
    A2ref = be.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert A2 - A2ref == A2*0
    assert (b2[0], b2[1], b2[2]) == (2, 4, 8)
    assert colidxs == [0, 1, 2]


@pytest.mark.parametrize('key', ['sympy'])
def test_linear_rref_aug(key):
    be = Backend(key)
    x, y, z = map(be.Symbol, 'x y z'.split())
    A1 = be.Matrix([[1, 2, 3], [3, 4, 7], [6, 5, 9]])
    B1a = [[0], [2], [11]]
    R1a, B1a, pivots1a = linear_rref(A1, B1a, backend=be)
    assert R1a == be.eye(3) and pivots1a == [0, 1, 2] and B1a == be.Matrix([[4], [1], [-2]])

    B1b = [[x], [y], [z]]
    R1b, B1b, pivots1b = linear_rref(A1, B1b, backend=be)
    delta1b = B1b - be.Matrix([
        [(x - 3*y + 2*z)/4],
        [(15*x - 9*y + 2*z)/4],
        [(7*y - 2*z - 9*x)/4]
    ])
    delta1b.simplify()
    assert R1b == be.eye(3) and pivots1b == [0, 1, 2] and delta1b == be.Matrix([[0], [0], [0]])
