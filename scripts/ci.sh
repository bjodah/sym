#!/bin/bash -xeu
if [[ "$CI_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${1^^}_RELEASE_VERSION=\$CI_BRANCH
    echo ${CI_BRANCH} | tail -c +2 > __conda_version__.txt
fi

# Py3
conda create -q -n test3 python=3.5 python-symengine sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8
source activate test3
./build_and_install_python-symengine.sh 97c5a21d0b5acf743c59ebce5d925d658698b322
python setup.py install
# (cd /; python -m pytest --pyargs $1)
PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $1 --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
#source deactivate

# Py2
conda create -q -n test2 python=2.7 python-symengine sympy pysym symcxx pip pytest
source activate test2
python setup.py sdist
pip install dist/*.tar.gz
(cd /; python -m pytest --pyargs $1)

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
