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

# pip install -r requirements.txt

#if you are using ubuntu, install python3-tk
sudo apt-get install python3-tk


# install the module
python3 setup.py install


#if you are still getting an error explicitly mention python3 instead of python wherever required during installation
#if your system doesn't have python3 then first install python3

# use pytest  in -orbitdeterminator to verify if everything has been setup correctly

#you can also do this by creating a virtual environment as follows(if you don't want to mention python3 explicitly everytime):
Install prerequisites:
sudo apt-get install python3 python3-pip virtualenvwrapper

#Create a Python3 based virtual environment. Optionally enable --system-site-packages flag

Create a virtual environment(ubuntu):
mkvirtualenv -p /usr/bin/python3 <venv-name>

Create a virtual environment(windows-using cmd):
pip3 install virtualenv
cd to directory where you want to install virtual environment
use- virtualenv -p path\to\where\python3\has\been\installed\python.exe environmentname
use- "path\to\new\virtualenvironment\Scripts\activate.bat"
you are inside the virtual environment that you just created(skip to- Install other requirements...)

Use the created environement(ubuntu continued):
workon <venv-name>

Install other requirements using pip package manager:
pip install -r requirements.txt
install python3-tk(for ubuntu users):sudo apt-get install python3-tk
pip install <package_name(any package you want to install in this virtual environment e.g. orbitdeterminator) >

# use pytest to verify if everything has been setup correctly.


```
