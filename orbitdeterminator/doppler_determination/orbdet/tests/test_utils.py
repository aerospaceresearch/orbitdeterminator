import unittest
import numpy as np

from orbdet.utils.utils import *
from orbdet.utils.utils_aux import *

np.set_printoptions(precision=20)

class TestTransformations(unittest.TestCase):
    """ Unit test for utils.py. Mosto of it tested compared to Vallado's MATLAB implementation.
    """

    @classmethod
    def setUpClass(cls) -> None:

        # Observer position
        # Checking in ITRF, so all observer positions are fixed
        cls.x_obs_1 = np.array([[518.1913969159496, -5282.096722846272, 3525.827476444865, 0, 0, 0]]).T*1e3
        cls.x_obs_2 = np.array([[518.1913969159496, -5282.096722846272, 3525.827476444865, 0, 0, 0]]).T*1e3

        cls.x_obs_arr = np.concatenate([cls.x_obs_1, cls.x_obs_2], axis=1)

        # Satellite positions
        cls.x_sat_1 =  np.array([[-2.5892693453515938e+05, -5.5791703748704530e+06, 3.9236384040617784e+06,
            7.1805587930556385e+03, -7.3068029074420338e+02, -5.5979417930939235e+02]]).T
        cls.x_sat_2 = np.array([[4.5208219685642188e+05, -5.6171149075666666e+06, 3.8442491231144061e+06,
            7.1685422344352655e+03, -3.4886906729108524e+01, -1.0425277899525731e+03]]).T

        cls.x_sat_3 = np.array([[-5558.91355593, -400.86892821, 3960.02332486, 
            0.78846265, -7.58966734, 0.34103136]]).T*1e3

        cls.x_sat_arr = np.concatenate([cls.x_sat_1, cls.x_sat_2], axis=1)

        # Falconsat reference frequency
        cls.f_ref = 435.103 

        ###### Tests - multiple observers #####
        x_0, t_sec, cls.x_sat_orbdyn_stm, cls.x_obs_arr_t, _ = get_example_scenario(id=0, frame='itrs')

        cls.nt = len(t_sec)

        # Set observer position
        cls.x_obs_0_t = cls.x_obs_arr_t[:,:,0]
        cls.x_obs_1_t = cls.x_obs_arr_t[:,:,1]
        cls.x_obs_2_t = cls.x_obs_arr_t[:,:,2]

    def test_range_range_rate(self):
        """ Unit test for range and range rate function.
        """

        # Test 1
        r_1, rr_1 = range_range_rate(self.x_sat_1, self.x_obs_1)
        np.testing.assert_almost_equal(r_1, 922181.755369173)
        np.testing.assert_almost_equal(rr_1, -6057.12508942666)

        # Test 2
        r_2, rr_2 = range_range_rate(self.x_sat_2, self.x_obs_2)
        np.testing.assert_almost_equal(r_2, 466904.653536031)
        np.testing.assert_almost_equal(rr_2, -1700.9516913344)
        
        # Test 3 - Array
        r_arr, rr_arr = range_range_rate(self.x_sat_arr, self.x_obs_arr)
        np.testing.assert_almost_equal(r_arr[0], 922181.755369173)
        np.testing.assert_almost_equal(r_arr[1], 466904.653536031)
        np.testing.assert_almost_equal(rr_arr[0], -6057.12508942666)

        np.testing.assert_almost_equal(rr_arr[1], -1700.9516913344)

        # Test 4 - One satellite, multiple observers
        r, rr = range_range_rate(self.x_sat_orbdyn_stm, self.x_obs_arr_t)
        rt_0, rrt_0 =  range_range_rate(self.x_sat_orbdyn_stm, self.x_obs_0_t)
        rt_1, rrt_1 =  range_range_rate(self.x_sat_orbdyn_stm, self.x_obs_1_t)
        rt_2, rrt_2 =  range_range_rate(self.x_sat_orbdyn_stm, self.x_obs_2_t)

        # Shapes
        np.testing.assert_equal(r.shape, (3, self.nt))
        np.testing.assert_equal(rr.shape, (3, self.nt))

        # Check station 1
        np.testing.assert_equal(r[0,:], rt_0)
        np.testing.assert_equal(rr[0,:], rrt_0)

        # Check station 2
        np.testing.assert_equal(r[1,:], rt_1)
        np.testing.assert_equal(rr[1,:], rrt_1)

        # Check station 3
        np.testing.assert_equal(r[2,:], rt_2)
        np.testing.assert_equal(rr[2,:], rrt_2)

    def test_doppler_shift(self):
        """ Unit test for Doppler shift.
        """
        # Test 1
        df = doppler_shift(self.x_sat_1, self.x_obs_1, self.f_ref, c=C)
        np.testing.assert_equal(df[0], -0.008790992659944792)

        # Test 2 - Array
        x_sat_arr = np.concatenate([self.x_sat_1, self.x_sat_2], axis=1)
        x_obs_arr = np.concatenate([self.x_obs_1, self.x_obs_2], axis=1)

        df_arr = doppler_shift(x_sat_arr, x_obs_arr, self.f_ref, c=C)

        np.testing.assert_almost_equal(df_arr, np.array([-0.008790992659944792, -0.002468671789450662]))

    def test_orbdyn_2body(self):
        """ Unit test for 2body integration function (state vector derivative only).
        """

        # Test 1 
        x_orbdyn_1 = orbdyn_2body(self.x_sat_1, t=1, mu=MU)
        x_orbdyn_1_ref = np.array([[7180.55879305564, -730.680290744203, -559.794179309392,
            0.324554918222522, 6.9932747167318, -4.91812929254025]]).T
        np.testing.assert_almost_equal(x_orbdyn_1, x_orbdyn_1_ref)

        # Test 2
        x_orbdyn_3 = orbdyn_2body(self.x_sat_3, t=1, mu=MU)
        x_orbdyn_3_ref = np.array([[788.46265, -7589.66734, 341.03136, 
            6.93328222128297, 0.499978527289434, -4.93908729428414]]).T
        np.testing.assert_almost_equal(x_orbdyn_3, x_orbdyn_3_ref)

        # Test 3 - Array
        x_arr = np.concatenate([self.x_sat_1, self.x_sat_3], axis=1)
        x_orbdyn_arr = orbdyn_2body(x_arr, 1, mu=MU)

        np.testing.assert_equal(x_arr.shape, x_orbdyn_arr.shape)
        np.testing.assert_almost_equal(x_orbdyn_1_ref, np.expand_dims(x_orbdyn_arr[:,0], axis=1))
        np.testing.assert_almost_equal(x_orbdyn_3_ref, np.expand_dims(x_orbdyn_arr[:,1], axis=1))


    def test_get_matrix_A(self):
        """ Unit test for A matrix.
        """
        # Reference variables
        A_1_ref = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [-1.248050094833666e-06, 1.165989213742894e-07, -8.200000627995956e-08, 0, 0, 0],
            [1.165989213742894e-07, 1.258927852507746e-06, -1.766876847314698e-06, 0, 0, 0],
            [-8.200000627995956e-08, -1.766876847314698e-06, -1.087775767408096e-08, 0, 0, 0]
        ])

        A_2_ref = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [-1.239120140248052e-06, -2.055651825938295e-07, 1.406849932631190e-07, 0, 0, 0],
            [-2.055651825938295e-07, 1.298479848325686e-06, -1.748009053274349e-06, 0, 0, 0],
            [1.406849932631190e-07, -1.748009053274349e-06, -5.935970807763372e-08, 0, 0, 0]
        ])

        # Test 1
        A_1 = get_matrix_A(self.x_sat_1, mu=MU).squeeze()

        np.testing.assert_equal(A_1.shape, (6,6))
        np.testing.assert_almost_equal(A_1, A_1_ref)

        # Test 2 - Array
        A_arr = get_matrix_A(self.x_sat_arr, mu=MU)

        np.testing.assert_equal(A_arr.shape, (6,6,2))
        np.testing.assert_almost_equal(A_arr[:,:,0], A_1_ref)
        np.testing.assert_almost_equal(A_arr[:,:,1], A_2_ref)

    def test_orbdyn_2body_stm(self):
        """ Unit test for 2 body integration function
            (state vector and state transition matrix derivatives).
        """
        
        Phi_0 = np.eye(6)
        x_1 = np.concatenate([self.x_sat_1.squeeze(), Phi_0.flatten()])
        x_2 = np.concatenate([self.x_sat_2.squeeze(), Phi_0.flatten()])
        x_arr = np.concatenate([[x_1], [x_2]], axis=0).T

        # Reference variables
        x_orbdyn_1_ref = np.array([7.180558793055639e+03, -7.306802907442034e+02, -5.597941793093923e+02, 
            0.324554918222522, 6.993274716731798, -4.918129292540249, 0, 0, 0,
            -1.248050094833666e-06, 1.165989213742894e-07, -8.200000627995956e-08, 0, 0, 0, 
            1.165989213742894e-07, 1.258927852507746e-06, -1.766876847314698e-06,  0, 0, 0,
            -8.200000627995956e-08, -1.766876847314698e-06, -1.087775767408096e-08, 
            1, 0, 0, 0, 0, 0, 
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0]).T

        x_orbdyn_2_ref = np.array([7.168542234435266e+03, -34.886906729108524, -1.042527789952573e+03, 
            -0.567663629219594, 7.053212571397640, -4.827087693401094, 0, 0, 0,
            -1.239120140248052e-06, -2.055651825938295e-07, 1.406849932631190e-07, 0, 0, 0, 
            -2.055651825938295e-07, 1.298479848325686e-06, -1.748009053274349e-06, 0, 0, 0, 
            1.406849932631190e-07, -1.748009053274349e-06, -5.935970807763372e-08, 
            1, 0, 0, 0, 0, 0, 
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0]).T

        # Test 1
        x_orbdyn_1 = orbdyn_2body_stm(x_1, 1, mu=MU)

        # Assert shape
        np.testing.assert_equal(x_orbdyn_1.shape, (42,))

        Phi_1 = x_orbdyn_1[6:].reshape((6,6))
        Phi_1_ref = x_orbdyn_1_ref[6:].reshape((6,6)).T
        Phi_2_ref = x_orbdyn_2_ref[6:].reshape((6,6)).T

        np.testing.assert_almost_equal(x_orbdyn_1[0:6], x_orbdyn_1_ref[0:6])
        np.testing.assert_almost_equal(Phi_1, Phi_1_ref)
        
        # Test 2 - Array
        x_orbdyn_arr = orbdyn_2body_stm(x_arr, 1, mu=MU)

        # Assert returned shape
        np.testing.assert_equal(x_orbdyn_arr.shape, (42, 2))

        # Check state vectors
        np.testing.assert_almost_equal(x_orbdyn_arr[0:6,0], x_orbdyn_1_ref[0:6])
        np.testing.assert_almost_equal(x_orbdyn_arr[0:6,1], x_orbdyn_2_ref[0:6])

        # Check STMs
        Phi_1_arr = x_orbdyn_arr[6:,0].reshape((6,6))
        Phi_2_arr = x_orbdyn_arr[6:,1].reshape((6,6))

        np.testing.assert_almost_equal(Phi_1_arr, Phi_1_ref)
        np.testing.assert_almost_equal(Phi_2_arr, Phi_2_ref)

    def test_get_matrix_range_rate_H(self):
        """ Unit test for observation Jacobian function.
        """

        # Test 1
        H_1 = get_matrix_range_rate_H(self.x_sat_1, self.x_obs_1)
        H_1_ref = np.array([0.002251451274953, -0.002908250817344, 0.002226382699264, 
            -0.842695408932710, -0.322142191921000, 0.431380175654917])

        np.testing.assert_equal(H_1.shape, (1, 6, 1))
        np.testing.assert_almost_equal(H_1.squeeze(), H_1_ref)

        # Test 2
        H_2 = get_matrix_range_rate_H(self.x_sat_2, self.x_obs_2)
        H_2_ref = np.array([0.014837512995586, -0.002688709755468, 2.516457481210894e-04, 
            -0.141590364454198, -0.717530189907480, 0.681984307198533])

        np.testing.assert_equal(H_2.shape, (1,6,1))
        np.testing.assert_almost_equal(H_2.squeeze(), H_2_ref)

        # Test 3 - Array
        H_arr = get_matrix_range_rate_H(self.x_sat_arr, self.x_obs_arr)

        # Shape of the observation Jacobian matrix
        # (dim_z, dim_x, n)
        # (1, 6, 2)
        np.testing.assert_equal(H_arr.shape, (1, 6, 2))     # Asser shape
    
        np.testing.assert_almost_equal(H_arr[:,:, 0].squeeze(), H_1_ref)
        np.testing.assert_almost_equal(H_arr[:,:, 1].squeeze(), H_2_ref)

        # Test 4 - MUltiple observers
        H_0_t = get_matrix_range_rate_H(self.x_sat_orbdyn_stm, self.x_obs_0_t)
        H_1_t = get_matrix_range_rate_H(self.x_sat_orbdyn_stm, self.x_obs_1_t)
        H_2_t = get_matrix_range_rate_H(self.x_sat_orbdyn_stm, self.x_obs_2_t)
        H_arr_t = get_matrix_range_rate_H(self.x_sat_orbdyn_stm, self.x_obs_arr_t)

        np.testing.assert_equal(H_arr_t.shape, (3, 6, self.nt))

        np.testing.assert_equal(H_arr_t[0,:,:], H_0_t.squeeze())
        np.testing.assert_equal(H_arr_t[1,:,:], H_1_t.squeeze())
        np.testing.assert_equal(H_arr_t[2,:,:], H_2_t.squeeze())

    def test_f_obs_range_rate(self):
        """ Unit test for observation function
        """

        # Test 1
        z_1, H_1 = f_obs_range_rate(self.x_sat_1, self.x_obs_1)
        H_1_ref = np.array([0.002251451274953, -0.002908250817344, 0.002226382699264, 
            -0.842695408932710, -0.322142191921000, 0.431380175654917])

        # Shape of the observation Jacobian matrix
        # (dim_z, dim_x, n)
        np.testing.assert_equal(H_1.shape, (1, 6, 1))
        np.testing.assert_almost_equal(z_1, -6.057125089426660e+03)
        np.testing.assert_almost_equal(H_1.squeeze(), H_1_ref)

        # Test 2
        z_2, H_2 = f_obs_range_rate(self.x_sat_2, self.x_obs_2)
        H_2_ref = np.array([0.014837512995586, -0.002688709755468, 2.516457481210894e-04, 
            -0.141590364454198, -0.717530189907480, 0.681984307198533])

        np.testing.assert_equal(H_1.shape, (1, 6, 1))
        np.testing.assert_almost_equal(z_2, -1.700951691334402e+03)
        np.testing.assert_almost_equal(H_2.squeeze(), H_2_ref)

        # Test 3 - Array
        z_arr, H_arr = f_obs_range_rate(self.x_sat_arr, self.x_obs_arr)

        np.testing.assert_equal(z_arr.shape, (1, 2))
        np.testing.assert_equal(H_arr.shape, (1, 6, 2))

        np.testing.assert_almost_equal(z_arr.squeeze(), np.array([-6.057125089426660e+03, -1.700951691334402e+03]))
        np.testing.assert_almost_equal(H_1_ref, H_arr[:,:,0].squeeze())
        np.testing.assert_almost_equal(H_2_ref, H_arr[:,:,1].squeeze())

        # Test 4 - Multiple observers
        z_t, H_t = f_obs_range_rate(self.x_sat_orbdyn_stm, self.x_obs_arr_t)
        z_0_t, H_0_t = f_obs_range_rate(self.x_sat_orbdyn_stm, self.x_obs_0_t)
        z_1_t, H_1_t = f_obs_range_rate(self.x_sat_orbdyn_stm, self.x_obs_1_t)
        z_2_t, H_2_t = f_obs_range_rate(self.x_sat_orbdyn_stm, self.x_obs_2_t)

        np.testing.assert_equal(z_t.shape, (3, 240))
        np.testing.assert_equal(H_t.shape, (3, 6, 240))

        # Compare positions 
        np.testing.assert_equal(z_t[0,:], z_0_t.squeeze())
        np.testing.assert_equal(z_t[1,:], z_1_t.squeeze())
        np.testing.assert_equal(z_t[2,:], z_2_t.squeeze())

        # Compare Jacbians
        np.testing.assert_equal(H_t[0, :,:], H_0_t.squeeze())
        np.testing.assert_equal(H_t[1, :,:], H_1_t.squeeze())
        np.testing.assert_equal(H_t[2, :,:], H_2_t.squeeze())

    def test_f_obs_x_sat(self):
        """ Unit test for satellite full state vector observation function.
        """

        H_ref = np.expand_dims(np.eye(6), axis=2)

        # Test 1
        test_x_sat_1, H_1 = f_obs_x_sat(self.x_sat_1)
        
        np.testing.assert_equal(test_x_sat_1.shape, self.x_sat_1.shape)
        np.testing.assert_equal(test_x_sat_1, self.x_sat_1)
        np.testing.assert_equal(H_1, H_ref)

        # Test 2 - Array
        test_x_sat_arr, H_arr = f_obs_x_sat(self.x_sat_arr)

        np.testing.assert_equal(test_x_sat_arr.shape, self.x_sat_arr.shape)
        np.testing.assert_equal(test_x_sat_arr, self.x_sat_arr)
        np.testing.assert_equal(H_arr, np.repeat(H_ref, 2, axis=2))

if __name__ == "__main__":
    unittest.main()