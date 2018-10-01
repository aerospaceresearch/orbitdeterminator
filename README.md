# Orbitdeterminator: Automated satellite orbit determination

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9c770ba2dd9d48fa8ba3ac207b9f5c85)](https://www.codacy.com/app/201452004/orbitdeterminator?utm_source=github.com&utm_medium=referral&utm_content=aerospaceresearch/orbitdeterminator&utm_campaign=badger)
[![Build Status](https://travis-ci.org/aerospaceresearch/orbitdeterminator.svg?branch=master)](https://travis-ci.org/aerospaceresearch/orbitdeterminator)
[![Documentation Status](https://readthedocs.org/projects/orbit-determinator/badge/?version=latest)](http://orbit-determinator.readthedocs.io/en/latest/?badge=latest)

## Quick Start

__orbitdeterminator__ is a package written in python3 for determining orbit of satellites based on positional data. Various filtering and determination algorithms are available for satellite operators to choose from.  

### Installation Instructions
Run the following commands to install orbitdeterminator:

```
# clone the repo
git clone https://github.com/aerospaceresearch/orbitdeterminator/

# cd into the repo
cd orbitdeterminator

# make a virtual environment
virtualenv .

# activate the virtual environment
source bin/activate

# install the module
python setup.py install
```

See [here](http://orbit-determinator.readthedocs.io/en/latest/) for further documentation.
