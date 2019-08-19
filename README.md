
# Orbitdeterminator: Automated satellite orbit determination

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9c770ba2dd9d48fa8ba3ac207b9f5c85)](https://www.codacy.com/app/201452004/orbitdeterminator?utm_source=github.com&utm_medium=referral&utm_content=aerospaceresearch/orbitdeterminator&utm_campaign=badger)
[![Build Status](https://travis-ci.org/aerospaceresearch/orbitdeterminator.svg?branch=master)](https://travis-ci.org/aerospaceresearch/orbitdeterminator)
[![Documentation Status](https://readthedocs.org/projects/orbit-determinator/badge/?version=latest)](http://orbit-determinator.readthedocs.io/en/latest/?badge=latest)

## Quick Start

__orbitdeterminator__ is a package written in python3 for determining orbit of satellites based on positional data. Various filtering and determination algorithms are available for satellite operators to choose from.  

## Installation Instructions

### Virtual Environment Setup
For this guide we will be using **miniconda**, you are free to use any virtual environment setup

**Step 1**: Install miniconda using script and instruction [here](https://docs.conda.io/en/latest/miniconda.html)

**Step 2**: After installation, create a virtualenv with python3.5 as
```
conda create -n env_name python=3.5.2
```
**Step 3**: Activate your virtualenv
```
conda activate env_name
```
*Rest of the guide will assume that virtualenv is activated
### Linux Users
Run the following commands to install orbitdeterminator:

**Step 1**: Clone the repository
```
git clone https://github.com/aerospaceresearch/orbitdeterminator/
```
**Step 2**: Change directory to orbitdeterminator
```
cd orbitdeterminator
```
**Step 3**: Install required dependencies
```
sudo apt-get install python-tk
```
and
```
python setup.py install
```
or manually install with
```
pip install -r requirements.txt
```
**Step 4**: Test your setup
```
pytest
```
If guide is followed correctly then pytest will not show any failed test case. After this, program can be used (to learn how to use the program check orbitdeterminator tutorials [here](https://orbit-determinator.readthedocs.io/en/latest/examples.html))
### Windows/macOS Users
All the steps are same except for **Step 3** from Linux Users guide

For **Step 3**, instead of using pip use conda to install **pykep** and **matplotlib**
* Add conda-forge source channel to conda
```
conda config --add channels conda-forge
```
* Now install pykep and matplotlib
```
conda install pykep==2.1
conda install matplotlib
```
Now, remove these two dependencies from requirements.txt and then run
```
pip install -r requirements.txt
```
#### Alternate Method for Windows/macOS Users

If you don't want to use conda to install **pykep** and **matplotlib**, then you can just build pykep from source with instructions from their official website [here](https://esa.github.io/pykep/installation.html), and remove pykep as dependency from requirement file. After this you can follow guide for Linux users above.

### Installation using Install Script
Install scripts is available for Unix (macOS and linux) users, which can be used with following instructions:
#### Guide for macOS Users
**Step 1**: Follow "Virtual Environment Setup" guide above and use conda for virtualenv and activate your conda env
**Step 2**: Add conda-forge to source channels
```
conda config --add channels conda-forge
```
**Step 3**: Install pykep and matplotlib using conda
```
conda install pykep==2.1
conda install matplotlib
```
**Step 4**: Make the install script for macOS executable
```
chmod +x ./install_macos.sh
```
**Step 4**: Run the install script with sudo
```
sudo ./install_macos.sh
```
#### Guide for macOS Users
Linux users only needs to make the script executable and then run the script using sudo
```
chmod +x ./install_linux.sh
sudo ./install_linux.sh
```
<hr/>
After successfully running the install script, user can now use the program with 'orbitdeterminator' command from terminal. Example -

```
orbitdeterminator -f orbit.csv
```
Keep in mind that observational data is used from "$HOME/orbitdeterminator/orbitdeterminator/obs_data" directory, where $HOME is home directory of the system. So, user is required to move the data in the above directory inside appropriate sub-directory i.e SourceCSV for .csv data file and SourceTXT for .txt data file

## Contribute:

[![GitHub pull requests](https://img.shields.io/github/issues-pr/aerospaceresearch/orbitdeterminator.svg?style=for-the-badge)](https://github.com/aerospaceresearch/orbitdeterminator/pulls)
[![GitHub issues](https://img.shields.io/github/issues/aerospaceresearch/orbitdeterminator.svg?style=for-the-badge)](https://github.com/aerospaceresearch/orbitdeterminator/issues)
[![Zulip](https://img.shields.io/badge/Chat-on%20Zulip-17C789.svg?style=for-the-badge)](https://aerospaceresearch.zulipchat.com/#narrow/stream/147024-OrbitDeterminator)

PRs are welcomed. For contributing to **orbitdeterminator** refer [CONTRIBUTING.md](CONTRIBUTING.md). If there are any issues or ideas they can be addressed through the [issues](https://github.com/aerospaceresearch/orbitdeterminator/issues) or in [chat room](https://aerospaceresearch.zulipchat.com/#narrow/stream/147024-OrbitDeterminator).
