#!/bin/bash -xe
if [[ "$CI_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${1^^}_RELEASE_VERSION=\$CI_BRANCH
fi

python3 -m pip install --user .[all]
PYTHONPATH=$(pwd) ./scripts/run_tests.sh
PYTHONPATH=$(pwd) ./scripts/render_notebooks.sh examples/
./scripts/generate_docs.sh

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
