# Orbitdeterminator: Automated satellite orbit determination

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9c770ba2dd9d48fa8ba3ac207b9f5c85)](https://www.codacy.com/app/201452004/orbitdeterminator?utm_source=github.com&utm_medium=referral&utm_content=aerospaceresearch/orbitdeterminator&utm_campaign=badger)
[![Build Status](https://travis-ci.org/aerospaceresearch/orbitdeterminator.svg?branch=master)](https://travis-ci.org/aerospaceresearch/orbitdeterminator)
[![Documentation Status](https://readthedocs.org/projects/orbit-determinator/badge/?version=latest)](http://orbit-determinator.readthedocs.io/en/latest/?badge=latest)

## Quick Start

__orbitdeterminator__ is a package written in python3 for determining orbit of satellites based on positional data. Various filtering and determination algorithms are available for satellite operators to choose from.  

## Installation

Instructions for debian/ubuntu:

1. Install pip for python3. `sudo apt-get install python3-pip`
2. In the cloned folder, run `pip3 install -r requirements.txt` to install dependencies
3. Install the python interface to the Tk GUI toolkit with `sudo apt-get install python3-tk`
4. Run tests with `pytest`
5. Go to the `orbitdeterminator/` subdirectory and run `python3 main.py` to run the sample program. See `python3 main.py --help` to know how to supply your own data
