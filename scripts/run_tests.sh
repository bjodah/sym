#!/bin/bash -e
# Usage
#   $ ./scripts/run_tests.sh
# or
#   $ ./scripts/run_tests.sh --cov sym --cov-report html
${PYTHON:-python3} -m pytest --doctest-modules --pep8 --flakes $@
${PYTHON:-python3} -m doctest README.rst
