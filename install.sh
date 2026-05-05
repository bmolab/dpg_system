#!/bin/bash

conda env create --file environment_2026.yml
conda run -n dpg_system_2026 pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
conda run -n dpg_system_2026 python -m spacy download en_core_web_lg

