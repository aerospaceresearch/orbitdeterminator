import time
import numpy as np
import argparse
import os

from astropy.time import Time   # Astropy 4.1rc1 is used

from orbdet.utils.utils import *
from orbdet.utils.utils_aux import *
from orbdet.utils.utils_vis import *

from scipy.optimize import fsolve

np.random.seed(100)
np.set_printoptions(precision=4)

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

def equations_tdoa(vars, *data):

    x, y, z, tau = vars
    x_sat = np.array([[x], [y], [z]], dtype=np.float64)

    x_obs, tdoa = data

    r = C*(tdoa + tau) - np.linalg.norm(x_obs - x_sat, axis=0)

    return (r.item(0), r.item(1), r.item(2), r.item(3))

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_id", type=int, default=0, help="Scenario id (0, 1).")
    parser.add_argument("--frame", type=str, default="teme", help="Tracking frame (teme, itrf)")    # Use TEME
    parser.add_argument("--n_samples", type=int, default=100, help="Number of sample points")

    args = parser.parse_args()
    args.scenario_id = 3

    n_obs = 4

    x_0, t_sec, x_sat_orbdyn_stm, x_obs_multiple, _ = get_example_scenario(id=3, frame='teme')
    tdoa, tof = get_tdoa_simulated_approx(x_sat_orbdyn_stm, x_obs_multiple)
    r, rr = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple)

    idx = 0
    x_sat = np.expand_dims(x_sat_orbdyn_stm[0:3, idx], axis=1)
    x_obs = x_obs_multiple[0:3, idx, :]

    vars_0  = [x_sat.item(0)+10000, x_sat.item(1)+10000, x_sat.item(2)+10000, 1]
    vars_gt = [x_sat.item(0), x_sat.item(1), x_sat.item(2), r[0,idx]/C]
    data    = (x_obs, tdoa[:, idx])

    a = equations_tdoa(vars_0, *data)

    result = fsolve(equations_tdoa, vars_0, args=data)

    print(np.array(vars_0))
    print(np.array(vars_gt))
    print(result)
    print("")
    print(result - vars_gt)
