#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
elif [[ "$OSTYPE" == "darwin"* ]]; then
      conda install pytorch::pytorch torchvision torchaudio -c pytorch
elif [[ "$OSTYPE" == "cygwin" ]]; then
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
elif [[ "$OSTYPE" == "msys" ]]; then
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
elif [[ "$OSTYPE" == "win32" ]]; then
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
fi

pip install ssqueezepy
pip install dearpygui
pip install pyquaternion
pip install fuzzywuzzy
pip install python-Levenshtein
pip install spacy
python -m spacy download en_core_web_lg
pip install python-osc
conda install pyopengl
conda install pyglfw -c conda-forge
pip install matplotlib
pip install numpy-quaternion
pip install scipy
conda install freetype
pip install freetype-py
pip install kornia
pip install opencv-python
pip install transformers