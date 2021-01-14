from setuptools import *

setup(
	name="orbitdeterminator",
	packages=find_packages(),
	version="1.0",
	license="MIT",
	author="Alexandros Kazantzidis, Nilesh Chaturvedi",
	author_email="alexandroskaza23@gmail.com, n.chaturvedi3@gmail.com",
	url="http://orbit-determinator.readthedocs.io/en/latest/",
	description="__orbitdeterminator__ is a package written in python3 for determining orbit of satellites based " \
				  "on positional data. Various filtering and determination algorithms are available for satellite " \
				  "operators to choose from.  ",
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib",
		"pykep",
		"pytest",
		"PyInquirer",
	]
)
