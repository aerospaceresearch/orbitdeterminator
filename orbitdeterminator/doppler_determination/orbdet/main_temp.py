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

    x_0, t_sec, x_sat_orbdyn_stm, x_obs_multiple, _ = get_example_scenario(id=args.scenario_id, frame=args.frame)
    r, rr = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple)
    tof = r / C

    print(r.shape)
    print(tof.shape)

    for i in range(r.shape[0]-1):
        tof[i,:] -= tof[0,:]

    # TDoA

