#!/bin/bash -xe
if [[ "$CI_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${1^^}_RELEASE_VERSION=\$CI_BRANCH
    echo ${CI_BRANCH} | tail -c +2 > __conda_version__.txt
fi

# Py2
conda create -q -n test2 python=2.7 sympy pysym symcxx pip pytest python-symengine
source activate test2
python -m pip install --upgrade https://github.com/sympy/sympy/archive/sympy-1.1rc1.tar.gz
python2 setup.py sdist
ORIPATH=$(pwd)
(cd /; python2 -m pip install $ORIPATH/dist/*.tar.gz)
(cd /; python2 -m pytest --pyargs $1)
source deactivate

# Py3
conda create -q -n test3 python=3.5 notebook sympy pysym symcxx pip pytest pytest-cov pytest-flakes pytest-pep8 python-symengine
source activate test3
python -m pip install --upgrade https://github.com/sympy/sympy/archive/sympy-1.1rc1.tar.gz
python setup.py install
# (cd /; python -m pytest --pyargs $1)
PYTHONPATH=$(pwd) ./scripts/run_tests.sh --cov $1 --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg


! grep "DO-NOT-MERGE!" -R . --exclude ci.sh

python3 -m pip install --user .[all]
./scripts/render_examples.sh
python3 -m pip install --user --force-reinstall docutils==0.12  # see https://github.com/sphinx-doc/sphinx/pull/3217
./scripts/generate_docs.sh
