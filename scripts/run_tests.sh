#!/bin/bash -ex
# Usage
#   $ ./scripts/run_tests.sh
# or
#   $ ./scripts/run_tests.sh --cov sym --cov-report html
${PYTHON:-python3} -m pytest --doctest-modules "$@"  # --flake8
${PYTHON:-python3} -m doctest $(dirname $BASH_SOURCE)/../README.rst
