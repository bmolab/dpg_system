"""Paths for the ragdoll test scripts: the repo root on sys.path, and
where the capture files live.  Every script here imports this first."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
