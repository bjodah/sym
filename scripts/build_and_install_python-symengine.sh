#!/bin/bash
git clone git://github.com/symengine/symengine.py
git clone git://github.com/symengine/symengine
cd symengine
git checkout $(../symengine.py/symengine_version.txt)
cmake
make
make install
cd ../symengine.py
python setup.py install
cd ..
rm -r symengine.py symengine 
