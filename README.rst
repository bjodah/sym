sym
===

.. image:: http://hera.physchem.kth.se:9090/api/badges/bjodah/sym/status.svg
   :target: http://hera.physchem.kth.se:9090/bjodah/sym
   :alt: Build status
.. image:: https://img.shields.io/pypi/v/sym.svg
   :target: https://pypi.python.org/pypi/sym
   :alt: PyPI version
.. image:: https://img.shields.io/badge/python-2.7,3.5-blue.svg
   :target: https://www.python.org/
   :alt: Python version
.. image:: https://img.shields.io/pypi/l/sym.svg
   :target: https://github.com/bjodah/sym/blob/master/LICENSE
   :alt: License
.. image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
   :target: http://hera.physchem.kth.se/~sym/benchmarks
   :alt: airspeedvelocity
.. image:: http://hera.physchem.kth.se/~sym/branches/master/htmlcov/coverage.svg
   :target: http://hera.physchem.kth.se/~sym/branches/master/htmlcov
   :alt: coverage

`sym <https://github.com/bjodah/sym>`_ provides a unified wrapper to some
symbolic manipulation libraries in Python. It makes it easy for library authors
to test their packages against several symbolic manipulation libraries.

Currently the following Python pacakges are available as "backends":

- `SymPy <https://github.com/sympy/sympy>`_
- `SymEngine <https://github.com/symengine/symengine.py>`_
- `PySym <https://github.com/bjodah/pysym>`_
- `SymCXX <https://github.com/bjodah/symcxx>`_

The capabilities exposed here are those needed by 

- `pyodesys <https://pypi.python.org/pypi/pyodesys>`_
- `pyneqsys <https://pypi.python.org/pypi/pyneqsys>`_

and include:

- Differentiation
- Numerical evaluation (including "lambdify" support)

see `tests <https://github.com/bjodah/sym/tree/master/sym/tests/>`_ for examples.
Note that ``pyodesys`` and ``pyneqsys`` also act as test suits for this package.


Documentation
-------------
Auto-generated API documentation for the latest stable release is found here:
`<https://bjodah.github.io/sym/latest>`_
(and the development version for the current master branch is found here:
`<http://hera.physchem.kth.se/~sym/branches/master/html>`_).

Installation
------------
Simplest way to install sym and its (optional) dependencies is to use pip:

::

   $ pip install --user sym pytest
   $ python -m pytest --pyargs sym

or the `conda package manager <http://conda.pydata.org/docs/>`_:

::

   $ conda install -c bjodah sym pytest
   $ python -m pytest --pyargs sym

Source distribution is available here:
`<https://pypi.python.org/pypi/sym>`_

Example
-------
Differentiation

.. code:: python

   >>> from sym import Backend
   >>> be = Backend('pysym')  # just an example, use SymPy rather than pysym
   >>> x, y = map(be.Symbol, 'x y'.split())
   >>> expr = x*y**2 - be.tan(2*x)
   >>> print(expr.diff(x))
   ((y**2) - ((1 + (tan((2*x))**2))*2))


for more examples, see `examples/ <https://github.com/bjodah/sym/tree/master/examples>`_, and rendered jupyter notebooks here:
`<http://hera.physchem.kth.se/~sym/master/examples>`_

License
-------
The source code is Open Source and is released under the simplified 2-clause BSD license. See `LICENSE <LICENSE>`_ for further details.
Contributors are welcome to suggest improvements at https://github.com/bjodah/sym

Author
------
Björn I. Dahlgren, contact:

- gmail address: bjodah
- kth.se address: bda
