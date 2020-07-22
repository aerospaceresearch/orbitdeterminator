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

    n_obs = 4

    x_0, t_sec, x_sat_orbdyn_stm, x_obs_multiple, _ = get_example_scenario(id=3, frame='teme')
    tdoa, tof = get_tdoa_simulated(x_sat_orbdyn_stm, x_obs_multiple)
    r, rr = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple)

    idx = 0
    x_sat = np.expand_dims(x_sat_orbdyn_stm[0:3, idx], axis=1)
    x_obs = x_obs_multiple[0:3, idx, :]

    vars_0  = [x_sat.item(0)+10000, x_sat.item(1)+10000, x_sat.item(2)+10000, 1]
    vars_gt = [x_sat.item(0), x_sat.item(1), x_sat.item(2), r[0,idx]/C]
    data    = (x_obs, tdoa[:, idx])

    a = tdoa_objective_function(vars_0, *data)

    result = fsolve(tdoa_objective_function, vars_0, args=data)

    print(np.array(vars_0))
    print(np.array(vars_gt))
    print(result)
    print("")
    print(result - vars_gt)
