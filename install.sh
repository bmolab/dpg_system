#!/bin/bash

conda env create --file environment.yml
conda run -n dpg_system python -m spacy download en_core_web_lg

