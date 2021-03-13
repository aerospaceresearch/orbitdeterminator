from setuptools import *

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
	name="orbitdeterminator",
	packages=find_packages(),
	version="1.0",
	license="MIT",
	author="aerospaceresearch.net community",
	author_email="orbitdeterminator@aerospaceresearch.net",
	url="http://orbit-determinator.readthedocs.io/en/latest/",
	description="__orbitdeterminator__ is a package written in python3 for determining orbit of satellites based " \
				  "on positional data. Various filtering and determination algorithms are available for satellite " \
				  "operators to choose from.  ",
    long_description=long_description,
    long_description_content_type="text/markdown",
	install_requires=[req for req in requirements if req[:2] != "# "and req[:4] != "http"],
    dependency_links=[req for req in requirements if req[:4] != "http"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
