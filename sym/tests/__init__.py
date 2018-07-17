# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
from .. import Backend

if os.environ.get('SYM_STRICT_TESTING', '0')[:1].lower() in ('1', 't'):
    AVAILABLE_BACKENDS = list(Backend.backends)
else:
    AVAILABLE_BACKENDS = []

    for k in Backend.backends:
        try:
            __import__(k)
        except ImportError:
            pass
        else:
            AVAILABLE_BACKENDS.append(k)
