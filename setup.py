#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
from itertools import chain
import os
import shutil
import warnings

from setuptools import setup


pkg_name = 'sym'
url = 'https://github.com/bjodah/' + pkg_name
license = 'BSD'

SYM_RELEASE_VERSION = os.environ.get('SYM_RELEASE_VERSION', '')  # v*

# http://conda.pydata.org/docs/build.html#environment-variables-set-during-the-build-process
if os.environ.get('CONDA_BUILD', '0') == '1':
    try:
        SYM_RELEASE_VERSION = 'v' + open(
            '__conda_version__.txt', 'rt').readline().rstrip()
    except IOError:
        pass


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)

release_py_path = _path_under_setup(pkg_name, '_release.py')

if len(SYM_RELEASE_VERSION) > 0:
    if SYM_RELEASE_VERSION[0] == 'v':
        TAGGED_RELEASE = True
        __version__ = SYM_RELEASE_VERSION[1:]
    else:
        raise ValueError("Ill formated version")
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(open(release_py_path).read())


classifiers = [
    "Development Status :: 3 - Alpha",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

tests = [
    'sym.tests',
]

with open(_path_under_setup(pkg_name, '__init__.py'), 'rt') as f:
    short_description = f.read().split('"""')[1].split('\n')[1]
if not 10 < len(short_description) < 255:
    warnings.warn("Short description from __init__.py proably not read correctly")
long_descr = io.open(_path_under_setup('README.rst'), encoding='utf-8').read()
if not len(long_descr) > 100:
    warnings.warn("Long description from README.rst probably not read correctly.")
_author, _author_email = open(_path_under_setup('AUTHORS'), 'rt').readline().split('<')

extras_req = {
    'symbolic': ['sympy>=1.0', 'pysym', 'symcxx'],  # use conda for symengine
    'docs': ['Sphinx', 'sphinx_rtd_theme', 'numpydoc'],
    'testing': ['pytest', 'pytest-cov', 'pytest-flakes', 'pytest-pep8']
}
if sys.version_info[0] > 2:
    extras_req['symbolic'].append('diofant')
extras_req['all'] = list(chain(extras_req.values()))


setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=short_description,
    long_description=long_descr,
    classifiers=classifiers,
    author=_author,
    author_email=_author_email.split('>')[0].strip(),
    url=url,
    license=license,
    packages=[pkg_name] + tests,
    install_requires=['numpy'],
    extras_require=extras_req
)

if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            # Same commit should generate different sdist
            # depending on tagged version (set $SYM_RELEASE_VERSION)
            # e.g.:  $ SYM_RELEASE_VERSION=v1.2.3 python setup.py sdist
            # this will ensure source distributions contain the correct version
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__', release_py_path)
