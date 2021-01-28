import unittest
import numpy as np

from orbitdeterminator.doppler.archive.utils_astro import *

class TestTransformations(unittest.TestCase):
    """ Unit test for coordinate system transofrmations.
    """

    @classmethod
    def setUpClass(cls) -> None:

        # Reference JD, 01/01/2020 00:00:00
        cls.date_0 = np.array([2000, 1, 1, 0, 0, 0])
        cls.date_1 = np.array([2018, 3, 22, 6, 58, 0.5])
        cls.date_2 = np.array([2020, 4, 15, 6, 30, 30])

        # Chilbolton
        cls.geodetic_0 = np.array([np.deg2rad(51.1483578), np.deg2rad(-1.4384458), 81.0])
        # Atlanta
        cls.geodetic_1 = np.array([np.deg2rad(33.7743331), np.deg2rad(-84.3970209), 288.0])

        cls.eci_1 = np.zeros((6,1))
        cls.eci_2 = np.zeros((6,1))

        cls.ecef_1 = np.zeros((6,1))
        cls.ecef_2 = np.zeros((6,1))

    def test_jd(self):
        """ Test Julian Day Number
        """

        # Reference JD case
        jd_0, jdfrac_0 = get_jd(self.date_0)
        self.assertEqual(jd_0, 2451544.5)
        self.assertEqual(jdfrac_0, 0)

        # Test #1, slight difference w/ MATLAB results, 14 sig. digits
        jd_1, jdfrac_1 = get_jd(self.date_1)
        self.assertEqual(jd_1, 2458199.5)
        self.assertAlmostEqual(jdfrac_1, 0.290283564814815)
        #print(f"\n{jdfrac_1}")
        #print(0.290283564814815)

        jd_2, jdfrac_2 = get_jd(self.date_2)
        self.assertEqual(jd_2, 2458954.5)
        self.assertAlmostEqual(jdfrac_2, 0.271180555555556)
        #print(f"\n{jdfrac_2}")
        #print(0.271180555555556)

    def test_gmst(self):
        """ Test Greenwitch Mean Sidereal Time
        """

        # Test #0
        # MATLAB 14 significant digits
        jd_0, jdfrac_0 = get_jd(self.date_0)
        test_gmst_0 = get_gmst(jd_0 + jdfrac_0)
        self.assertAlmostEqual(test_gmst_0, 1.744767163330613)
        #print(f"\n{test_gmst_0}")
        #print(1.744767163330613)

        # Test #1
        # MATLAB 11 significant digits
        jd_1, jdfrac_1 = get_jd(self.date_1)
        test_gmst_1 = get_gmst(jd_1 + jdfrac_1)
        self.assertAlmostEqual(test_gmst_1, 4.960910441081389)
        #print(f"\n{test_gmst_1}")
        #print(4.96091044108139)

        # Test #2
        # MATLAB 11 significant digits
        jd_2, jdfrac_2 = get_jd(self.date_2)
        test_gmst_2 = get_gmst(jd_2 + jdfrac_2)
        self.assertAlmostEqual(test_gmst_2, 5.26229132147851)
        #print(f"\n{test_gmst_2}")
        #print(5.26229132147851)

    def test_geodetic_to_ecef(self):
        """ Test latitude, longitude, altitude to Earth-Centered, Earth-Fixed (ECEF)
        transformation
        """

        # Test #1, Chilbolton
        test_ecef_0 = geodetic_to_ecef(self.geodetic_0)
        np.testing.assert_allclose(test_ecef_0,
            np.array([4007978.00308211, -100643.89986322, 4943977.86517641, 0, 0 ,0]))
        # print(f"\n{test_ecef_0}")

        # Test #1, Atlanta
        test_ecef_1 = geodetic_to_ecef(self.geodetic_1)
        np.testing.assert_allclose(test_ecef_1,
            np.array([518191.396915951, -5282096.72284627, 3525827.47644486, 0, 0 ,0]))
        # print(f"\n{test_ecef_1}")

        pass

    def test_ecef_to_pef(self):
        """ Test Earth-Centered, Earth-Fixed frame to Pseudo Earth-Centered Fixed (PEF) frame """

        # Test zero rotation
        test_ecef_0 = geodetic_to_ecef(self.geodetic_0)
        test_pef_0 = ecef_to_pef(test_ecef_0, 0, 0, 0)
        np.testing.assert_allclose(test_ecef_0, test_pef_0)

        # Test some precession
        test_pef_1 = ecef_to_pef(test_ecef_0, 0, 0.1, 0.1)
        pef_1_matlab = np.array([3494380.60530255, 430913.646165595, 5302881.44445618, 0, 0, 0])
        np.testing.assert_allclose(test_ecef_0, test_pef_0)

        # TODO: More tests

    def test_polar_motion(self):
        # Test 0
        # Zero polar motion coefficients
        xp_0, yp_0 = 0, 0
        pm_0 = polar_motion(xp_0, yp_0)
        np.testing.assert_allclose(pm_0, np.eye(3,3))

        # Test 1, IAU-76/FK5
        xp_1, yp_1 = 0.1, 0.12
        pm_1 = polar_motion(xp_1, yp_1, ttt=0, type='iau-76')
        pm_1_matlab = np.array([
            [0.995004165278026, 0, -0.0998334166468282],
            [0.0119512786679861, 0.992808635853866, 0.119114144887101],
            [0.0991154781937681, -0.119712207288919, 0.987848727998592]
        ])
        np.testing.assert_allclose(pm_1, pm_1_matlab)

        # Test 2, IAU-76/FK5
        xp_2, yp_2 = 0.02, 0.01
        pm_2 = polar_motion(xp_2, yp_2, ttt=0, type='iau-76')
        pm_2_matlab = np.array([
            [0.999800006666578, 0, -0.0199986666933331],
            [0.000199983333838881, 0.999950000416665, 0.0099978334341645],
            [0.0199976667683312, -0.00999983333416666, 0.999750017082826]
        ])
        np.testing.assert_allclose(pm_2, pm_2_matlab)

        # TODO: Test IAU-2000 polar motion

    def test_ecef_to_teme(self):
        """ Test Earth-Centered, Earth-Fixed (ECEF) to True Equator, Mean Equinox (TEME)
        coordinate system transformation.
        """

        # Test - ground site
        jd_1, jdfrac_1 = get_jd(self.date_1)
        ttt_1 = get_ttt(jd_1 + jdfrac_1)
        ecef_1 = geodetic_to_ecef(self.geodetic_0)
        test_teme_1 = ecef_to_teme(ecef_1, jd_1+jdfrac_1, ttt_1, 0, 0, 0, 0)

        teme_1_matlab = np.array([
            888294.982007313, -3909597.02433772, 4943977.86517641,
            0285.092316786936, 0064.7754929303933, 0
        ])

        print(test_teme_1)
        print(teme_1_matlab)

        np.testing.assert_allclose(test_teme_1, teme_1_matlab)

        #test_ecef_1 = geodetic_to_ecef(self.geodetic_0)


    # def test_ecef_to_eci(self):
    #     """ Test Earth-Centered, Earth-Fixed (ECEF) to Earth-Centered Inertial (ECI)
    #     coordinate system transformation.
    #     """

    #     # Test #1
    #     jd_1, jdfrac_1 = get_jd(self.date_1)
    #     test_ecef_1 = geodetic_to_ecef(self.geodetic_0)
    #     test_eci_1 = ecef_to_eci(test_ecef_1, jd_1 + jdfrac_1)
    #     #print(test_eci_1)

    #     #test_eci_1 = ecef_to_eci(self.ecef_1, get_jd(self.date_1))

    #     # Test #2
    #     #test_eci_2 = ecef_to_eci(self.ecef_2, get_jd(self.date_2))

    #     pass

    def test_eci_to_ecef(self):
        """ Test Earth-Centered Inertial (ECI) to Earth-Centered, Earth-Fixed (ECEF)
        coordinate system transformation.
        """

        # Test #1
        #test_ecef_1 = ecef_to_eci(self.eci_1, get_jd(self.date_1))

        # Test #2
        #test_ecef_2 = ecef_to_eci(self.eci_2, get_jd(self.date_2))

        pass

if __name__ == "__main__":
    unittest.main()