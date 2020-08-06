import time
import numpy as np
import argparse
import os

from orbitdeterminator.doppler.utils.utils import *
from orbitdeterminator.doppler.utils.utils_aux import *
from orbitdeterminator.doppler.utils.utils_vis import *

from scipy.optimize import fsolve

np.random.seed(100)
np.set_printoptions(precision=4)

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_id", type=int, default=3, help="Scenario id (3). Only 3 has 4 stations")
    #parser.add_argument("--frame", type=str, default="teme", help="Tracking frame (teme, itrf)")    # Use TEME
    #parser.add_argument("--n_samples", type=int, default=100, help="Number of sample points")

    args = parser.parse_args()

    x_0, t_sec, x_sat, x_obs, _ = get_example_scenario(id=args.scenario_id, frame='teme')
    tdoa, tof = get_tdoa_simulated(x_sat, x_obs)
    r, rr = range_range_rate(x_sat, x_obs)

    # Solve Time Differential of Arrival multilateration
    p_sat, tau = solve_tdoa(tdoa, x_obs)

    path = "images/"
    prefix = ""

    fig_1 = plot_tdoa(tdoa, tof, t_sec)
    fig_1.savefig(os.path.join(path, f"{prefix}_tdoa_measurements"))

    fig_2 = plot_tdoa_results(p_sat, x_obs, x_sat)
    fig_2.savefig(os.path.join(path, f"{prefix}_tdoa_results"))
    
    fig_3 = plot_tdoa_errors(p_sat, x_sat)
    fig_3.savefig(os.path.join(path, f"{prefix}_tdoa_errors"))

    # For example, take a 10-second sliding window for the simulation and apply Herrick-Gibbs method:

    w = 10  # window size

    x_sat_hg = np.zeros((x_sat.shape[0], x_sat.shape[1]-w))
    t_sat_hg = t_sec[0:-w]
    error_hg = [None] * int(x_sat.shape[1]-w)

    # Perform Herrick_Gibbs
    for i in range(p_sat.shape[1]-2*w):
        idx = np.array([i, i+w, i+2*w])
        x_sat_hg[:,i], error_hg[i] = herrick_gibbs(p_sat[:, idx], t_sec[idx], angle_checks=True)
    
    print(x_sat_hg.shape)
    
    