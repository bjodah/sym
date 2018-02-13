#!/bin/bash -xe
if [[ "$CI_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${1^^}_RELEASE_VERSION=\$CI_BRANCH
    echo ${CI_BRANCH} | tail -c +2 > __conda_version__.txt
fi

# Py2
conda create -q -n test2 python=2.7 sympy pysym symcxx pip pytest python-symengine numba
source activate test2
python -m pip install --upgrade https://github.com/sympy/sympy/archive/sympy-1.1rc1.tar.gz
python2 setup.py sdist
ORIPATH=$(pwd)
(cd /; python2 -m pip install $ORIPATH/dist/*.tar.gz)
(cd /; python2 -m pytest --pyargs $1)
source deactivate

# Py3
conda create -q -n test3 python=3.5 notebook sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8 python-symengine numba
source activate test3
python -m pip install --upgrade https://github.com/sympy/sympy/archive/sympy-1.1rc1.tar.gz
python setup.py install
python -m pip install diofant
PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $1 --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg

python -m pip install git+https://github.com/sympy/sympy@master
PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $1 --cov-report html



! grep "DO-NOT-MERGE!" -R . --exclude ci.sh

python3 -m pip install --user .[all]
PYTHONPATH=$(pwd) ./scripts/render_examples.sh
./scripts/generate_docs.sh
