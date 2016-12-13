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
