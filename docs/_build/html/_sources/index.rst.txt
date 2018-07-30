.. Orbit Determinator documentation master file, created by
   sphinx-quickstart on Fri Jun 23 04:28:14 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
++++++++++++++++++++++++++++++++++++++++++++++
Welcome to Orbit Determinator's documentation!
++++++++++++++++++++++++++++++++++++++++++++++

========================
About Orbit Determinator
========================

The orbitdeterminator package provides tools to compute the orbit of a satellite from positional
measurements. It supports both cartesian and spherical coordinates for the initial positional data, two filters
for smoothing and removing errors from the initial data set and finally two methods for preliminary orbit determination.
The package is labeled as an open source scientific package and can be helpful for projects concerning space orbit
tracking.

Lots of university students build their own cubesat's and set them into space orbit, lots of researchers
start building their own ground station to track active satellite missions. For those particular space enthusiasts
we suggest using and trying our package. Any feedback is more than welcome and we wish our work to inspire other's to
join us and add more helpful features.

Our future goals for the package is to add a 3d visual graph of the final computed satellite orbit, add more
filters, methods and with the help of a tracking ground station to build a server system that computes orbital elements
for many active satellite missions.

=====================
Copyright and License
=====================

The project's idea belongs to AerospaceResearch.net and Andreas Hornig and it has been developed under
Google summer of code 2017 by Nilesh Chaturvedi and Alexandros Kazantzidis.

It is distributed under an open-source MIT license. Please find `LICENSE` in top level directory for details.

============
Installation
============

Open up your control panel, pip install git if you do not already have it and then clone the github repository of the
program https://github.com/aerospaceresearch/orbitdeterminator. Create a new virtual environment for python version 3.4.
Then, all you need to do is go to the directory where the package has been cloned with cd orbitdeterminator and
run **python setup.py install**. That should install the package into your Lib/site-packages and you will be able to
import and use it. Other than import you can just use it immediately from the clone directory (preferred).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules.rst
   examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
