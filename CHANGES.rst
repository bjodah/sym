v0.3.5
======
- Lambdify accepts kwarg 'cse'
- Updates to interface to e.g. symengine.py

v0.3.4
======
- DenseMatrix got 2 new methods:
  - sparse_jacobian_csc
  - sparse_jacobian_csr

v0.3.3
======
- Backend now support ``cse`` & ``ccode`` for symengine

v0.3.2
======
- Lambdify now supports ``sign``

v0.3.0
======
- Lambdify now supports multiple outputs

v0.2.0
======
- linear_rref now handles symbolic entries in the augmented part.

v0.1.8
======
- Provisional support for mpmath, numpy now coerces to float64

v0.1.7
======
- Fix sdist / conda package.

v0.1.6
======
- Added ``.util.check_transforms``

v0.1
====
- Support differentiation
- Support for numerical evaluation
