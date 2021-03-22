import os, argparse, time
import numpy as np

import astropy

from orbitdeterminator.doppler.utils.utils import *
from orbitdeterminator.doppler.utils.utils_aux import *
from orbitdeterminator.doppler.utils.utils_vis import *

from scipy.optimize import fsolve

if __name__ == '__main__':

    # Parser
    print(astropy.__version__)
    print(type(astropy.__version__))
    print(float(astropy.__version__[0:3]))
    
