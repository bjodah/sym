#!/bin/bash -xe

PKG_NAME=${1:-${CI_REPO_NAME##*/}}

if [[ "$CI_COMMIT_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${PKG_NAME^^}_RELEASE_VERSION=\$CI_COMMIT_BRANCH
fi

REPO_TMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$REPO_TMP_DIR"' EXIT
cp -ra . "$REPO_TMP_DIR/."
cd "$REPO_TMP_DIR"

python3 -m pip install symengine
python3 -m pip install --no-build-isolation \
        "git+https://github.com/bjodah/symcxx#egg=symcxx" \
        "git+https://github.com/bjodah/pysym#egg=pysym"  # unofficial backends
python3 -m pip install ${INSTALL_FLAGS_FOR_PIP:-} .[all]
./scripts/run_tests.sh -k "not diofant"
./scripts/render_notebooks.sh examples/
./scripts/generate_docs.sh

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
