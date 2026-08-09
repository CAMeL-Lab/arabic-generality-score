# Thin shim so `pip install -e .` works on pip < 21.3 (no PEP 660 support).
# All real configuration lives in pyproject.toml.
from setuptools import setup

setup()
