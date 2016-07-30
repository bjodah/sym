#!/bin/bash
git clone git://github.com/symengine/symengine.py
cd symengine.py
python setup.py install
cd ..
rm -r symengine.py
