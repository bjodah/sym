# -*- coding: utf-8 -*-

import glob
import os
import subprocess
import sys

import pytest


tests = glob.glob(os.path.join(os.path.dirname(__file__), '../*.py'))


@pytest.mark.parametrize('pypath', tests)
def test_examples(pypath):

    for _ in range(2):
        p = subprocess.Popen([sys.executable, pypath])
        assert p.wait() == os.EX_OK  # SUCCESS==0
