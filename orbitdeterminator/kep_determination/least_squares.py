"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

import numpy as np
import matplotlib.pyplot as plt

# convention:
# a: semi-major axis
# e: eccentricity
# eps: mean longitude at epoch
# Euler angles:
# I: inclination
# Omega: longitude of ascending node
# omega: argument of pericenter