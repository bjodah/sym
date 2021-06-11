#!/bin/bash -ex
# Usage
#   $ ./scripts/run_tests.sh
# or
#   $ ./scripts/run_tests.sh --cov sym --cov-report html
SYM_USE_CSE=1 ${PYTHON:-python3} -m pytest sym/tests/test_Lambdify.py
SYM_USE_NUMBA=1 ${PYTHON:-python3} -m pytest sym/tests/test_Lambdify.py
SYM_USE_CSE=1 SYM_USE_NUMBA=1 ${PYTHON:-python3} -m pytest sym/tests/test_Lambdify.py
${PYTHON:-python3} -m pytest --doctest-modules --pep8 --flakes $@
${PYTHON:-python3} -m doctest README.rst
