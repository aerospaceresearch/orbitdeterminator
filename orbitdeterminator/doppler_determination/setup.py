#!/usr/bin/env python
""" A setuptools based setup module.
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name="orbdet",
    version="0.0.1",
    description="Orbit Determination Test",
    packages=find_packages(),
    python_requires=">= 3.7",
)