#!/bin/bash -xe
if [[ "$DRONE_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${1^^}_RELEASE_VERSION=\$CI_BRANCH
fi

python3 -m pip install symengine
python3 -m pip install --no-build-isolation \
        "git+https://github.com/bjodah/symcxx#egg=symcxx" \
        "git+https://github.com/bjodah/pysym#egg=pysym"  # unofficial backends
python3 -m pip install ${INSTALL_FLAGS_FOR_PIP} .[all]
./scripts/run_tests.sh -k "not diofant"
./scripts/render_notebooks.sh examples/
./scripts/generate_docs.sh

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
