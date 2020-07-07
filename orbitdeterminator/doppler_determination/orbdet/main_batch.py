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

    x_0, t_sec, x_sat_orbdyn_stm, x_obs_multiple, _ = get_example_scenario(id=args.scenario_id, frame=args.frame)
    x_obs_1 = x_obs_multiple[:,:,0]

    # Define measurements
    _, rr_1 = range_range_rate(x_sat_orbdyn_stm, x_obs_1)
    z_rr_1 = np.expand_dims(rr_1, axis=0)       # Range rate measurements
    _, z_rr_multiple = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple)
    z_x_sat = x_sat_orbdyn_stm                  # Full state measurements

    # Uncvertainties
    R_rr_1 = np.eye(1)*1e-5                         # Measurement uncertainty range rate (1x1)
    R_rr = np.eye(x_obs_multiple.shape[2])*1e-6     # Measurement uncertainty range rate (array)
    R_x_sat = np.eye(6)*1e-12                       # Measurement uncertainty full state

    P_small = np.eye(6)*1e-6                        # Initial state uncertainty - small

    # Random sampling
    P = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])     # Initial uncertainty - random guess
    x_0r = np.random.multivariate_normal(x_0.squeeze(), P, args.n_samples).T
    x_0err = x_0r - x_0

    verbose = True
    max_iterations=250

    save_images(x_sat_orbdyn_stm, x_obs_multiple, t_sec=t_sec, prefix="00", path="images/")

    #####   Batch Test 0 - True satellite position as measurement, 
    #####   should converge with a very small error
    run_batch_0 = True
    if run_batch_0 == True:

        x_br = np.zeros(x_0r.shape)
        
        print(f"\nBatch 0 - true position of the satellite as measurement")
        print(f"\nBatch 0 Output:")

        for i in range(args.n_samples):

            x_b, output = batch(
                np.copy(np.expand_dims(x_0r[:,i], axis=1)),
                P, 
                R_x_sat, 
                z=z_x_sat, 
                t=t_sec, 
                x_obs=x_obs_1, 
                f_obs=f_obs_x_sat, 
                tolerance=1e-8,
                max_iterations=max_iterations)

            x_br[:,i] = x_b.squeeze()

            print(f"Sample {i} Number of iterations", output['num_it'])
            print(f"err_start: \t{x_0err[:,i]}, Norm {np.linalg.norm(x_0err[:,i])}")
            print(f"err_end: \t{ (x_0 - x_b).T}, Norm {np.linalg.norm( (x_0 - x_b).T)}")

    #####   Batch Test 1 - Range rate, groundtruth #####
    run_batch_1 = True
    if run_batch_1 == True:

        x_b, output = batch(
            np.copy(x_0), 
            P_small, 
            R=R_rr_1, 
            z=z_rr_1, 
            t=t_sec, 
            x_obs=x_obs_1, 
            f_obs=f_obs_range_rate, 
            tolerance=1e-8, 
            max_iterations=max_iterations
        )

        if verbose:
            print(f"\nBatch 1 - Single station, range rate measurements, true initial position")
            print(f"\nBatch 1 Output: \t{x_b.T}")
            print(f"Error: \t{(x_0 - x_b).T}\n")

    #####   Batch Test 2 - Range rate, random samples     #####
    start_time = time.time()
    run_batch_2 = True
    if run_batch_2 == True:     
        x_br = np.zeros(x_0r.shape)
        x_berr = np.zeros(x_0r.shape)

        if verbose:
            print(f"\nBatch 2 - one stations, range rate measurements")
            print(f"Batch 2 Output:")

        for i in range(args.n_samples):

            x_b, output = batch(
                np.copy(np.expand_dims(x_0r[:,i], axis=1)), 
                P, 
                R=R_rr_1, 
                z= z_rr_1, 
                t=t_sec, 
                x_obs=x_obs_1, 
                f_obs=f_obs_range_rate, 
                tolerance=1e-8, 
                max_iterations=max_iterations
            )
            x_br[:,i] = x_b.squeeze()
            x_berr[:,i] = (x_0 - x_b).squeeze()
            x_berr1 = (x_0 - x_b).T
            
            if verbose:
                print(f"Sample {i} Number of iterations", output['num_it'])
                print(f"err_start: \t{x_0err[:,i]}, Norm pos {np.linalg.norm(x_0err[0:3,i])}, \
                    Norm vel Norm pos {np.linalg.norm(x_0err[3:6,i])}")
                print(f"err_end: \t{x_berr1}, Norm pos {np.linalg.norm(x_berr1[0:3])}, \
                    Norm vel {np.linalg.norm(x_berr1[3:6])}")
    
    print(f"Elapsed batch 2 {(time.time()-start_time):.2f} s")

    save_images_batch_results(x_sat_orbdyn_stm, x_0r, x_br, x_berr, 
        prefix="batch_2", path="images/")

    # Random sampling
    P = np.diag([1, 1, 1, 1, 1, 1])*1e6             # Initial uncertainty - random guess
    x_0r = np.random.multivariate_normal(x_0.squeeze(), P, args.n_samples).T
    x_0err = x_0r - x_0

    start_time = time.time()
    #####   Batch Test 2_1 - Range rate, random samples, multiple measurements     #####
    run_batch_2_1 = True
    if run_batch_2_1 == True:     

        x_br = np.zeros(x_0r.shape)
        x_berr = np.zeros(x_0r.shape)

        if verbose:
            print(f"Batch 2_1 Output:")
            print(f"\nBatch 2_1 - three stations, range rate measurements")

        for i in range(args.n_samples):

            x_b, output = batch(
                    np.copy(np.expand_dims(x_0r[:,i], axis=1)), 
                    P, 
                    R_rr, 
                    z=z_rr_multiple, 
                    t=t_sec, 
                    x_obs=x_obs_multiple, 
                    f_obs=f_obs_range_rate, 
                    tolerance=1e-8,
                    max_iterations=max_iterations
                )
            x_br[:,i] = x_b.squeeze()
            x_berr[:,i] = (x_0 - x_b).squeeze()
            x_berr1 = (x_0 - x_b).T

            if verbose:
                print(f"Sample {i} Number of iterations", output['num_it'])
                print(f"err_start: \t{x_0err[:,i]}, Norm pos {np.linalg.norm(x_0err[0:3,i])}, \
                    Norm vel {np.linalg.norm(x_0err[3:6,i])}")
                print(f"err_end: \t{x_berr1}, Norm pos {np.linalg.norm(x_berr1[0:3])}, \
                    Norm vel {np.linalg.norm(x_berr1[3:6])}")

    print(f"Elapsed batch 2_1 {(time.time()-start_time):.2f} s")

    save_images_batch_results(x_sat_orbdyn_stm, x_0r, x_br, x_berr, 
        prefix="batch_2_1", path="images/")
