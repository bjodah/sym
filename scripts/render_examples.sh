#!/bin/bash
jupyter nbconvert --to=html --debug --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 examples/*.ipynb
./scripts/render_index.sh *.html
