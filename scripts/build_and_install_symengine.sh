#!/bin/bash
git clone git://github.com/symengine/symengine
cd symengine
git checkout $(../symengine.py/symengine_version.txt)
mkdir build-symengine
cd build-symengine
cmake ..
make
make install
cd ../..
rm -r symengine
