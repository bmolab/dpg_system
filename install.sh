#!/bin/bash

conda env create --file environment_2026.yml
conda run -n dpg_system python -m spacy download en_core_web_lg

