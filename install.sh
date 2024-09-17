#!/bin/bash

conda env create --file environment.yml python=3.10
conda run -n dpg_system python -m spacy download en_core_web_lg

