import time
import numpy as np
import argparse
import os
import astropy

from orbitdeterminator.doppler.utils.utils import *
from orbitdeterminator.doppler.utils.utils_aux import *
from orbitdeterminator.doppler.utils.utils_vis import *

from scipy.optimize import fsolve

np.random.seed(100)
np.set_printoptions(precision=4)

if __name__ == '__main__':

    # Parser
    print(astropy.__version__)
    print(type(astropy.__version__))
    print(float(astropy.__version__[0:3]))
    
