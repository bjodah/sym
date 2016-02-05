#!/bin/bash
source activate test2
(cd examples/; ipython2 nbconvert --to=html --debug --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 *.ipynb)
(cd examples/; ../scripts/render_index.sh *.html)
