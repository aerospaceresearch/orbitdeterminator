import time
import numpy as np
import argparse
import os

from astropy.time import Time   # Astropy 4.1rc1 is used

from orbdet.utils.utils import *
from orbdet.utils.utils_aux import *
from orbdet.utils.utils_vis import *

np.random.seed(100)
np.set_printoptions(precision=2)

def save_images(x_sat_orbdyn_stm, x_obs_multiple, t_sec=None, prefix="", path=""):
    """ Auxiliary function to save the images.
    """
    fig_1 = plot_example_3d(x_sat_orbdyn_stm, x_obs_multiple)
    fig_1.savefig(os.path.join(path, f"{prefix}_scenario"))

    fig_2 = plot_range_range_rate(x_sat_orbdyn_stm, x_obs_multiple, t_sec)
    fig_2.savefig(os.path.join(path, f"{prefix}_range_range_rate"))

def save_images_batch_results(x_sat_orbdyn_stm, x_0r, x_br, x_berr, prefix="", path=""):
    """ Auxiliary function to save the batch results.
    """
    fig_3 = plot_batch_results(x_sat_orbdyn_stm, x_0r, x_br, x_berr)
    fig_3.savefig(os.path.join(path, f"{prefix}_range_range_rate"))

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_id", type=int, default=0, help="Scenario id (0, 1).")
    parser.add_argument("--frame", type=str, default="teme", help="Tracking frame (teme, itrf)")    # Use TEME
    parser.add_argument("--n_samples", type=int, default=100, help="Number of sample points")

    args = parser.parse_args()
    args.scenario_id = 3

    tle = [  '1 30776U 07006E   20146.24591950  .00002116  00000-0  57170-4 0  9998',
            '2 30776  35.4350  68.4822 0003223 313.1473  46.8985 15.37715972733265']

    sat = get_6_oe_from_tle(tle)

    print(sat)