import time
import numpy as np
import argparse
import os

from orbdet.utils.utils import *
from orbdet.utils.utils_aux import *
from orbdet.utils.utils_vis import *

from scipy.optimize import fsolve

np.random.seed(100)
np.set_printoptions(precision=4)

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_id", type=int, default=0, help="Scenario id (0, 1).")
    parser.add_argument("--frame", type=str, default="teme", help="Tracking frame (teme, itrf)")    # Use TEME
    parser.add_argument("--n_samples", type=int, default=100, help="Number of sample points")

    args = parser.parse_args()
    args.scenario_id = 3

    x_0, t_sec, x_sat, x_obs, _ = get_example_scenario(id=args.scenario_id, frame='teme')
    tdoa, tof = get_tdoa_simulated(x_sat, x_obs)
    r, rr = range_range_rate(x_sat, x_obs)

    # Solve Time Differential of Arrival multilateration
    p_sat, tau = solve_tdoa(tdoa, x_obs)

