
# Allows user to choose perturbations, add custom perturbations, allows tweaking with predefined parameters
from __future__ import division
import re
import datetime


from astropy.utils.data import conf
from astropy import constants as cts
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_sun
from astropy.constants import G, M_earth, R_earth

# from poliastro.twobody.propagation import propagate, cowell
# from poliastro.ephem import build_ephem_interpolant
# # from poliastro.core.perturbations import atmospheric_drag, third_body, J2_perturbation,J3_perturbation,shadow_function
# import poliastro.core.perturbations
# from poliastro.bodies import Earth, Moon, Sun
# from poliastro.twobody import Orbit
# from poliastro.plotting import OrbitPlotter2D, OrbitPlotter3D
# from poliastro import neos

#Commented all codes above regarding poliastro and written one code below to import poliastro
import poliastro

from sgp4.model import Satellite
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4init

import inquirer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from kep_determination.gauss_method import *
from util.state_kep import state_kep
from util.kep_state import kep_state

mu_Earth = G.value*M_earth.value
# print(mu)
Re = R_earth.value
mu_Sun = cts.GM_sun.to(u.Unit('au3 / day2')).value
#mu_Earth = cts.GM_earth.to(u.Unit('km3 / s2')).value
# print(mu_Earth)
# we need to increase the timeout time to allow the download of data occur properly
conf.remote_timeout = 10000


# get initial orbital elements
def read_args_earth():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, help="path to IOD-formatted data file",
                        default='../example_data/SATOBS-ML-19200716.txt')
    parser.add_argument(
        '-o', '--obs_array', help="list of lines in file to be read", type=str, default=None)
    parser.add_argument('-b', '--body_name', type=str,
                        help="observed object/body name", default=None)
    parser.add_argument('-r', '--root_index', nargs='*',
                        help="user selection for multiple roots of Gauss polynomial (see docs for more information)", default=None)
    parser.add_argument('-i', '--iterations', type=int,
                        help="number of iterations of Gauss method refinement", default=0)
    parser.add_argument('-p', '--plot', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def read_args_sun():                                                                            # change here
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, help="path to MPC-formatted data file",
                        default='../example_data/mpc_eros_data.txt')
    parser.add_argument(
        '-o', '--obs_array', help="list of lines in file to be read", type=str, default=None)
    parser.add_argument('-b', '--body_name', type=str,
                        help="observed object/body name", default="Eros")
    parser.add_argument('-r', '--root_index', nargs='*',
                        help="user selection for multiple roots of Gauss polynomial (see docs for more information)", default=None)
    parser.add_argument('-i', '--iterations', type=int,
                        help="number of iterations of Gauss method refinement", default=0)
    parser.add_argument('-p', '--plot', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()



# plotting

cos = np.cos
sin = np.sin
pi = np.pi
dot = np.dot

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d', aspect=1)



max_radius = 0
def plotOrbit(semi_major_axis, eccentricity=0, inclination=0,
              right_ascension=0, argument_perigee=0, true_anomaly=0, label=None):

    global max_radius

    #Earth as globe
    Earth_radius = 6371.1 # units in km

    max_radius = max(max_radius, Earth_radius)

    # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    coefs = (1, 1, 1)

    rx, ry, rz = [Earth_radius/np.sqrt(coef) for coef in coefs]

    # Spherical angles
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates for spherical angless
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g')



    "Draws orbit around an earth in units of kilometers."
    # Rotation matrix for inclination
    inc = inclination * pi / 180
    R = np.matrix([[1, 0, 0],
                   [0, cos(inc), -sin(inc)],
                   [0, sin(inc), cos(inc)]    ])

    # Rotation matrix
    rot = (right_ascension + argument_perigee) * pi/180
    R2 = np.matrix([[cos(rot), -sin(rot), 0],
                    [sin(rot), cos(rot), 0],
                    [0, 0, 1]    ])

    # Plot orbit
    theta = np.linspace(0,2*pi, 360)
    r = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(theta))

    xr = r*cos(theta)
    yr = r*sin(theta)
    zr = 0 * theta
    pts = np.matrix(zip(xr,yr,zr))
    pts_t =  (R * R2)
    pts_t = pts_t*(pts_t.T)             #check
    pts = pts_t.T
    # pts =  (R * R2 * pts.T).T

    # Turn back into 1d vectors
    xr,yr,zr = pts[:,0].A.flatten(), pts[:,1].A.flatten(), pts[:,2].A.flatten()

    # Plot the orbit
    ax.plot(xr, yr, zr, '-')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    # plt.zlabel('Z (km)')

    # Plot the satellite
    sat_angle = true_anomaly * pi/180
    satr = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(sat_angle))
    satx = satr * cos(sat_angle)
    saty = satr * sin(sat_angle)
    satz = 0

    sat = (R * R2 * np.matrix([satx, saty, satz]).T ).flatten()
    satx = sat[0,0]
    saty = sat[0,1]
    satz = sat[0,2]

    c = np.sqrt(satx*satx + saty*saty)
    lat = np.arctan2(satz, c) * 180/pi
    lon = np.arctan2(saty, satx) * 180/pi
    print ("%s : Lat: %g° Long: %g" % (label, lat, lon))

    # Radius vector from earth and satellite as red sphere
    ax.plot([0, satx], [0, saty], [0, satz], 'r-')

    ax.plot([satx],[saty],[satz], 'ro')

    # global max_radius
    max_radius = max(max(r), max_radius)

    if label:
        ax.text(satx, saty, satz, label, fontsize=12)


    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()




#converts keplerian elements to cartesian
def kep_2_cart(a,e,i,omega_AP,omega_LAN,T, EA):
    a=a*1000
    i=np.deg2rad(i)
    omega_AP=np.deg2rad(omega_AP)
    omega_LAN=np.deg2rad(omega_LAN)
    mu_Earth = G.value*M_earth.value
    Re = R_earth.value
    t=0
    n = np.sqrt(mu_Earth/(a**3))
    M = n*(t - T)

    MA = EA - e*np.sin(EA)


    nu = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(EA/2))

    r = a*(1 - e*np.cos(EA))

    h = np.sqrt(mu_Earth*a * (1 - e**2))

    Om = omega_LAN
    w =  omega_AP

    X = r*(np.cos(Om)*np.cos(w+nu) - np.sin(Om)*np.sin(w+nu)*np.cos(i))
    Y = r*(np.sin(Om)*np.cos(w+nu) + np.cos(Om)*np.sin(w+nu)*np.cos(i))
    Z = r*(np.sin(i)*np.sin(w+nu))


    p = a*(1-e**2)

    V_X = (X*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(Om)*np.sin(w+nu) + \
    np.sin(Om)*np.cos(w+nu)*np.cos(i))
    V_Y = (Y*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(Om)*np.sin(w+nu) - \
    np.cos(Om)*np.cos(w+nu)*np.cos(i))
    V_Z = (Z*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(w+nu)*np.sin(i))

    return [X/1000,Y/1000,Z/1000,V_X/1000,V_Y/1000,V_Z/1000]


def propagate_state(r,v,t0,tf,bstar=0.21109E-4):
    """Propagates a state vector

       Args:
           r(1x3 numpy array): the position vector at epoch
           v(1x3 numpy array): the velocity vector at epoch
           t0(float): initial time (epoch)
           tf(float): final time

       Returns:
           pos(1x3 numpy array): the position at tf
           vel(1x3 numpy array): the velocity at tf
    """

    kep = state_kep(r,v)
    return propagate_kep(kep,t0,tf,bstar)



def __true_to_mean(T,e):
    """Converts true anomaly to mean anomaly.

       Args:
           T(float): true anomaly in degrees
           e(float): eccentricity

       Returns:
           float: the mean anomaly in degrees
    """

    T = np.radians(T)
    E = np.arctan2((1-e**2)*np.sin(T),e+np.cos(T))
    M = E - e*np.sin(E)
    M = np.degrees(M)
    M = M%360
    return M



# Parts of this method have been copied from:
# https://github.com/brandon-rhodes/python-sgp4/blob/master/sgp4/io.py
def kep_to_sat(kep,epoch,bstar=0.21109E-4,whichconst=wgs72,afspc_mode=False):
    """kep_to_sat(kep,epoch,bstar=0.21109E-4,whichconst=wgs72,afspc_mode=False)

       Converts a set of keplerian elements into a Satellite object.

       Args:
           kep(1x6 numpy array): the osculating keplerian elements at epoch
           epoch(float): the epoch
           bstar(float): bstar drag coefficient
           whichconst(float): gravity model. refer pypi sgp4 documentation
           afspc_mode(boolean): refer pypi sgp4 documentation

      Returns:
           Satellite object: an sgp4 satellite object encapsulating the arguments
    """

    deg2rad  =  np.pi / 180.0;         #    0.0174532925199433
    xpdotp   =  1440.0 / (2.0 * np.pi);  #  229.1831180523293

    tumin = whichconst.tumin

    satrec = Satellite()
    satrec.error = 0;
    satrec.whichconst = whichconst # Python extension: remembers its consts

    satrec.satnum = 0
    dt_obj = datetime.utcfromtimestamp(epoch)
    t_obj = dt_obj.timetuple()
    satrec.epochdays = (t_obj.tm_yday +
                        t_obj.tm_hour/24 +
                        t_obj.tm_min/1440 +
                        t_obj.tm_sec/86400)
    satrec.ndot = 0
    satrec.nddot = 0
    satrec.bstar = bstar

    satrec.inclo = kep[2]
    satrec.nodeo = kep[4]
    satrec.ecco = kep[1]
    satrec.argpo = kep[3]
    satrec.mo = __true_to_mean(kep[5],kep[1])
    satrec.no = 86400/(2*np.pi*(kep[0]**3/398600.4405)**0.5)

    satrec.no   = satrec.no / xpdotp; #   rad/min
    satrec.a    = pow( satrec.no*tumin , (-2.0/3.0) );

    #  ---- find standard orbital elements ----
    satrec.inclo = satrec.inclo  * deg2rad;
    satrec.nodeo = satrec.nodeo  * deg2rad;
    satrec.argpo = satrec.argpo  * deg2rad;
    satrec.mo    = satrec.mo     * deg2rad;

    satrec.alta = satrec.a*(1.0 + satrec.ecco) - 1.0;
    satrec.altp = satrec.a*(1.0 - satrec.ecco) - 1.0;

    satrec.epochyr = dt_obj.year
    satrec.jdsatepoch = epoch/86400.0 + 2440587.5
    satrec.epoch = dt_obj

    #  ---------------- initialize the orbit at sgp4epoch -------------------
    sgp4init(whichconst, afspc_mode, satrec.satnum, satrec.jdsatepoch-2433281.5, satrec.bstar,
             satrec.ecco, satrec.argpo, satrec.inclo, satrec.mo, satrec.no,
             satrec.nodeo, satrec)


    return satrec



def propagate_kep(kep,t0,tf,bstar=0.21109E-4):
    """Propagates a set of keplerian elements.

       Args:
           kep(1x6 numpy array): osculating keplerian elements at epoch
           t0(float): initial time (epoch)
           tf(float): final time

       Returns:
           pos(1x3 numpy array): the position at tf
           vel(1x3 numpy array): the velocity at tf
    """

    sat = kep_to_sat(kep,t0,bstar=bstar)
    tf = datetime.utcfromtimestamp(tf).timetuple()
    pos, vel = sat.propagate(
        tf.tm_year, tf.tm_mon, tf.tm_mday, tf.tm_hour, tf.tm_min, tf.tm_sec)

    return np.array(list(pos)),np.array(list(vel))




# Methods for perturbed propogation:


def third_body_moon(initial,time,f_time):                               # propogate under perturbation of a third body(moon)

    # database keeping positions of bodies in Solar system over time
    x_ephem="de432s"
    questions = [
      inquirer.List('Ephemerides',
                    message="Select ephemerides[de432s(default,small in size,faster)','de430(more precise)]:",
                    choices=['de432s','de430'],
                ),
    ]
    answers = inquirer.prompt(questions)


    x_ephem=answers["Ephemerides"]

    solar_system_ephemeris.set(x_ephem)


    #epoch = Time(time, format='iso', scale='utc')  # setting the exact event date is important
    epoch=Time(time, format='iso').jd
    f_time_1=Time(f_time, format='iso').jd

    # create interpolant of 3rd body coordinates (calling in on every iteration will be just too slow)     #check
    body_r = build_ephem_interpolant(
        Moon, 28 * u.day, (epoch*u.day,f_time_1*u.day), rtol=1e-2                                          #check
    )



    k_third= Moon.k.to(u.km ** 3 / u.s ** 2).value
    x=400
    print("Use default constants or you want to customize?\n1.Default.\n2.Custom.")
    check=input()
    if(check=='1'):
        pass
    else:
        k_third=input("Enter moon's gravity:")
        x=input("Multiply Moon's gravity by X times so that effect is visible,input X:")

    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))



    # multiply Moon gravity by x so that effect is visible :)
    rr = propagate(
        initial,
        tofs,
        method=cowell,
        rtol=1e-6,
        ad=third_body,
        k_third=x *k_third,
        third_body=body_r,
    )

    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))

    # if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
    #     print("In shadow.")
    # else:
    #     print("Not in shadow.")







def no_pert_earth(initial,time,f_time):                                                   # No perturbation


    # parameters of a body
    C_D = 2.2  # dimentionless (any value would do)
    A = ((np.pi / 4.0) * (u.m ** 2)).to(u.km ** 2).value  # km^2
    m = 100  # kg
    B = C_D * A / m



    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))


    rr =propagate(
        initial,
        tofs,
        method=cowell,
        ad=None
        )


    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    print("")
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))


    #if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
        #print("In shadow.")
    #else:
        #print("Not in shadow.")




def j2_earth(initial,time,f_time):                                                  # perturbation for oblateness
    R = Earth.R.to(u.km).value
    k = Earth.k.to(u.km ** 3 / u.s ** 2).value
    #s0 = Orbit.from_classical(Earth, x[0] * u.km, x[1] * u.one, x[4] * u.deg, * u.deg, x[3] * u.deg, 0 * u.deg, epoch=Time(0, format='jd', scale='tdb'))

    #orbit = Orbit.circular(
    #    Earth, 250 * u.km, epoch=Time(0.0, format="jd", scale="tdb"))

    # parameters of a body
    C_D = 2.2  # dimentionless (any value would do)
    A = ((np.pi / 4.0) * (u.m ** 2)).to(u.km ** 2).value  # km^2
    m = 100  # kg
    B = C_D * A / m

    J2=Earth.J2.value
    print("Use default constants or you want to customize?\n1.Default.\n2.Custom.")
    check=input()
    if(check=='1'):
        pass
    else:
        J2=input("Enter J2 constant")
        R=input("Enter radius of earth in km")



    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))
    rr =propagate(
        initial,
        tofs,
        method=cowell,
        ad=J2_perturbation,
        J2=J2,
        R=R
        )


    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    print("")
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))

    #if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
        #print("In shadow.")
    #else:
        #print("Not in shadow.")




def j3_earth(initial,time,f_time):                                                   # perturbation for oblateness
    R = Earth.R.to(u.km).value
    k = Earth.k.to(u.km ** 3 / u.s ** 2).value
    #s0 = Orbit.from_classical(Earth, x[0] * u.km, x[1] * u.one, x[4] * u.deg, * u.deg, x[3] * u.deg, 0 * u.deg, epoch=Time(0, format='jd', scale='tdb'))

    #orbit = Orbit.circular(
    #    Earth, 250 * u.km, epoch=Time(0.0, format="jd", scale="tdb"))

    # parameters of a body
    C_D = 2.2  # dimentionless (any value would do)
    A = ((np.pi / 4.0) * (u.m ** 2)).to(u.km ** 2).value  # km^2
    m = 100  # kg
    B = C_D * A / m

    J3=Earth.J3.value
    print("Use default constants or you want to customize?\n1.Default.\n2.Custom.")
    check=input()
    if(check=='1'):
        pass
    else:
        J3=input("Enter J3 constant:")
        R=input("Enter radius of earth in km:")


    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))


    rr =propagate(
        initial,
        tofs,
        method=cowell,
        ad=J3_perturbation,
        J3=J3,
        R=R
        )


    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    print("")
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))

    #if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
        #print("In shadow.")
    #else:
        #print("Not in shadow.")





def atmd_earth(initial,time,f_time):                                               # perturbation for atmospheric drag
    R = Earth.R.to(u.km).value
    k = Earth.k.to(u.km ** 3 / u.s ** 2).value
    #s0 = Orbit.from_classical(Earth, x[0] * u.km, x[1] * u.one, x[4] * u.deg, * u.deg, x[3] * u.deg, 0 * u.deg, epoch=Time(0, format='jd', scale='tdb'))

    #orbit = Orbit.circular(
    #    Earth, 250 * u.km, epoch=Time(0.0, format="jd", scale="tdb"))

    # parameters of a body
    C_D = 2.2  # dimentionless (any value would do)
    A = ((np.pi / 4.0) * (u.m ** 2)).to(u.km ** 2).value  # km^2
    m = 100  # kg
    B = C_D * A / m

    # parameters of the atmosphere
    rho0 = Earth.rho0.to(u.kg / u.km ** 3).value  # kg/km^3
    H0 = Earth.H0.to(u.km).value

    print("Use default constants or you want to customize?\n1.Default.\n2.Custom.")
    check=input()
    if(check=='1'):
        pass
    else:
        m=input("Enter mass of the body in kg:")
        R=input("Enter radius of earth in km:")
        C_D=input("Enter C_D(dimension):")
        H0=input("Enter atmospheric parameter H0:")
        #rho0=input("Enter atmospheric parameter rho0:")
        R=R*u.km
        H0=H0*u.km
        #rho0=rho0*(u.kg / u.km ** 3)

    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))



    #print("tofs:")
    #print(tofs)
    #print("ie:")
    #print(initial.epoch.iso)
    rr =propagate(
        initial,
        tofs,
        method=cowell,
        ad=atmospheric_drag,
        R=R,
        C_D=C_D,
        A=A,
        m=m,
        H0=H0,
        rho0=rho0,
    )

    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    print("")
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))

    #if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
        #print("In shadow.")
    #else:
        #print("Not in shadow.")


def a_d_1(t0, state,k, J2, R,C_D, A, m, H0, rho0):                                                    # J2+atmospheric_drag
    return (J2_perturbation(t0, state, k, J2, R)+ atmospheric_drag(t0, state, k, R, C_D, A, m, H0, rho0))


def a_d_2(t0, state,k, J3, R,C_D, A, m, H0, rho0):                                                    # J2+atmospheric_drag
    return (J3_perturbation(t0, state, k, J3, R)+ atmospheric_drag(t0, state, k, R, C_D, A, m, H0, rho0))


#@njit
def accel(t0, state, k):
     """Constant acceleration aligned with the velocity. """                            # requires modification and validation
     v_vec = state[3:]
     norm_v = (v_vec * v_vec).sum() ** .5
     return 1e-5 * v_vec / norm_v



def custom(initial,choice,state,time,f_time):                                          # requires modification and validation
    R = Earth.R.to(u.km).value
    k = Earth.k.to(u.km ** 3 / u.s ** 2).value
    #s0 = Orbit.from_classical(Earth, x[0] * u.km, x[1] * u.one, x[4] * u.deg, * u.deg, x[3] * u.deg, 0 * u.deg, epoch=Time(0, format='jd', scale='tdb'))

    #orbit = Orbit.circular(
    #    Earth, 250 * u.km, epoch=Time(0.0, format="jd", scale="tdb"))

    # parameters of a body
    C_D = 2.2  # dimentionless (any value would do)
    A = ((np.pi / 4.0) * (u.m ** 2)).to(u.km ** 2).value  # km^2
    m = 100  # kg
    B = C_D * A / m

    # parameters of the atmosphere
    rho0 = Earth.rho0.to(u.kg / u.km ** 3).value  # kg/km^3
    H0 = Earth.H0.to(u.km).value

    #J2,J3 parameters
    J3=Earth.J3.value
    J2=Earth.J2.value

    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    diff =datetime.strptime(str(f_time), datetimeFormat)\
        - datetime.strptime(str(time), datetimeFormat)
    print("Time difference:")
    print(diff)
    tofs=TimeDelta(f_time-time)
    tofs = TimeDelta(np.linspace(0, 31* u.day, num=1))


    if(choice=='7'):


        rr =propagate(
            initial,
            tofs,
            method=cowell,
            ad=accel,
            #k=k,

        )

    elif(choice=='5'):

        rr =propagate(
            initial,
            tofs,
            method=cowell,
            ad=a_d_1,
            #k=k,
            J2=J2,
            R=R,
            C_D=C_D,
            A=A,
            m=m,
            H0=H0,
            rho0=rho0
        )

    elif(choice=='6'):

        rr =propagate(
            initial,
            tofs,
            method=cowell,
            ad=a_d_2,
            #k=k,
            J3=J3,
            R=R,
            C_D=C_D,
            A=A,
            m=m,
            H0=H0,
            rho0=rho0
        )

    else:
        pass



    print("")
    print("Positions and velocity vectors are:")
    #print(str(rr.x))
    #print([float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))])
    r=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.z))][0]]* u.km
    v=[[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_x))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_y))][0],[float(s) for s in re.findall(r'-?\d+\.?\d*',str(rr.v_z))][0]]* u.km / u.s
    f_orbit= Orbit.from_vectors(Earth, r, v)
    print(r)
    print(v)
    #f_orbit.plot()
    print("")
    print("Orbital elements:")
    print(f_orbit.classical())
    print("")
    #print("")
    #print(f_orbit.ecc)
    plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))

    #if(shadow_function(np.asarray(r),get_sun(Time(datetime.now(),format='datetime')),Earth.R.to(u.km).value)):
        #print("In shadow.")
    #else:
        #print("Not in shadow.")

# converts iso format time to tle format, returns float

def con_iso_tle(time):                                              # Time object
    str_1=str(time)[2:4]
    str_2=str(time)[0:4]+"-01-01 00:00:00"
    str_2=Time(str_2,format='iso',scale='utc')
    gap=TimeDelta(time-str_2)
    str_f=str_1+str(gap)
    return float(str_f)


if __name__ == "__main__":


    print("Attractor body is?\n1.Earth\n2.Sun")
    a_b = input()
    if(a_b == '1'):
        args = read_args_earth()
        if args.obs_array is None:
            print("NONE")
            print(args.file_path)
            x0 = gauss_method_sat(args.file_path, bodyname=args.body_name,
                                  r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=False)

            time=Time(x0[7], format='jd').iso

            x0=np.asarray(x0)

            #x0[3:6] = np.deg2rad(x0[3:6])
            time_p=Time(x0[2], format='jd').iso
            time_obj=Time(time,format="iso",scale="utc")
            time_p_obj=Time(time_p,format="iso",scale="utc")
            #print("chk:")
            #print(time_obj.tdb.jd)
            nu=time2truean(x0[0],x0[1], mu_Earth,time_obj.tdb.jd,x0[2])
            eccentric_anomaly=truean2eccan(x0[1],nu)

            nu=np.rad2deg(nu)
            kep_i=(x0[0],x0[1],x0[4],x0[3],x0[5],nu)
            kep_i=np.asarray(kep_i)
            initial_state = kep_state(kep_i.reshape(-1, 1))
            initial = Orbit.from_classical(Earth, x0[0] * u.km, x0[1] * u.one, x0[4] * u.deg, x0[5]* u.deg, x0[3] * u.deg, nu* u.deg,epoch=Time(time, format='iso', scale='utc'))


        else:
            print("OBS_ARRAY")
            obs_arr = [int(item) for item in args.obs_array.split(',')]
            x0 = gauss_method_sat(args.file_path, obs_arr=obs_arr, bodyname=args.body_name,
                                  r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=False)
            time=Time(x0[7], format='jd').iso

            x0=np.asarray(x0)

            #x0[3:6] = np.deg2rad(x0[3:6])
            time_p=Time(x0[2], format='jd').iso
            time_obj=Time(time,format="iso",scale="utc")
            time_p_obj=Time(time_p,format="iso",scale="utc")
            #print("chk:")
            #print(time_obj.tdb.jd)
            nu=time2truean(x0[0],x0[1], mu_Earth,time_obj.tdb.jd,x0[2])
            eccentric_anomaly=truean2eccan(x0[1],nu)

            nu=np.rad2deg(nu)
            kep_i=(x0[0],x0[1],x0[4],x0[3],x0[5],nu)
            kep_i=np.asarray(kep_i)
            initial_state = kep_state(kep_i.reshape(-1, 1))
            initial = Orbit.from_classical(Earth, x0[0] * u.km, x0[1] * u.one, x0[4] * u.deg, x0[5]* u.deg, x0[3] * u.deg, nu* u.deg,epoch=Time(time, format='iso', scale='utc'))



        initial_state_r=[float(initial_state[0, 0]),float(initial_state[1, 0]),float(initial_state[2, 0])]
        initial_state_v=[float(initial_state[3, 0]),float(initial_state[4, 0]),float(initial_state[5, 0])]
        initial_state=[float(initial_state[0, 0]),float(initial_state[1, 0]),float(initial_state[2, 0]),float(initial_state[3, 0]),float(initial_state[4, 0]),float(initial_state[5, 0])]
        initial_state=np.asarray(initial_state)

        print("\nDo you want to add perturbations?[y/n]")           # tofs in the form of time delta
        a_p = input()

        print("Enter time to propagate to(format='iso', scale='utc'):\n2016-08-20 01:33:42.250")
        f_time ="2016-08-20 01:33:42.250"

        time=Time(time, format='iso', scale='utc')
        f_time=Time(f_time, format='iso', scale='utc')

        if(a_p == 'N' or a_p == 'n'):
            no_pert_earth(initial,time,f_time)         # normal propagation without perturbations

        elif(a_p=='Y' or a_p=='y'):

            print("\nWhat perturbation do you want to add?\n1.J2(Oblateness).\n2.J3(Oblateness).\n3.Atmospheric Drag.\n4.Third body perturbations.(Moon)")
            print("5.J2+atmospheric drag.\n6.J3+atmospheric drag.")
            print("7.Add custom perturbation acceleration aligned with velocity.\n8.Propogate using SGP4/SDP model.")
            # print("")
            ch = input()

            if(ch == '1'):
                j2_earth(initial,time,f_time)

            elif(ch == '2'):
                j3_earth(initial,time,f_time)

            elif(ch == '3'):
                atmd_earth(initial,time,f_time)

            elif(ch=='4'):
                third_body_moon(initial,time,f_time)

            elif(ch=='5'):
                custom(initial,ch,initial_state,time,f_time)

            elif(ch=='6'):
                custom(initial,ch,initial_state,time,f_time)

            elif(ch=='7'):
                custom(initial,ch,initial_state,time,f_time)

            elif(ch=='8'):
                #kep=[x0[0],x0[4],x0[5],x0[1],x0[3],mean_anomaly]
                #kep=np.asarray(kep)

                time=time.tt.datetime
                f_time=f_time.tt.datetime
                time=time.timestamp()
                f_time=f_time.timestamp()

                kep = state_kep(np.asarray(initial_state_r),np.asarray(initial_state_v))
                f_r,f_v = propagate_state(np.asarray(initial_state_r),np.asarray(initial_state_v),time,f_time,bstar=0.21109E-4)

                #r= [float(f_v[0]),float(initial_state[1, 0]),float(initial_state[2, 0])]
                f_r= np.asarray(f_r)
                f_v= np.asarray(f_v)
                #f_orbit= Orbit.from_vectors(Earth, r, v)

                #f_orbit.plot()
                print("Orbital elements:")
                print(state_kep(f_r,f_v))
                print("")
                print("Position and velocity vectors are(r,v):")
                print(f_r)
                print(f_v)
                print("")

            else:
                print("Invalid Input.Exiting...")
                sys.exit()
        else:
            print("Invalid Input.Exiting...")
            sys.exit()




    elif(a_b=='2'):
        print("\nDetermine orbit for-")
        print("1.An mpc datafile.")
        print("2.A neo using NASA NEO Webservice(you’ll need an internet connection).")
        opt_1=input()
        if(opt_1=='1'):
             args = read_args_sun()
             args.obs_array = [1, 14, 15, 24, 32, 37, 68, 81, 122, 162, 184, 206, 223]
             if args.obs_array is None:


                 x0 = gauss_method_mpc(args.file_path,bodyname=args.body_name,
                                  r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=False)
                 time=Time(x0[7], format='jd').iso

                 x0=np.asarray(x0)
                 #x0[3:6] = np.deg2rad(x0[3:6])

                 mean_motion=meanmotion(mu_Sun,x0[0])
                 mean_anomaly=meananomaly(mean_motion,time.tdb.jd,x0[2])
                 eccentric_anomaly=eccentricanomaly(x0[1],mean_anomaly)
                 nu=trueanomaly(x0[1],eccentric_anomaly)
                 nu=np.rad2deg(nu)


                 #initial = kep_2_cart(x0[0],x0[1],x0[4],x0[3],x0[5],x0[6],eccentric_anomaly)
                 initial = Orbit.from_classical(Sun, x0[0] * u.AU, x0[1] * u.one, x0[4] * u.deg, x0[5]* u.deg, x0[3] * u.deg, nu * u.deg,epoch=Time(time, format='iso', scale='utc'))


                 print("")
                 print(initial)
                 print(initial.rv())

                 print("\nPropagate to(Enter time in format='iso', scale='utc':")
                 p_t_s=input()
                 p_t_o= Time(p_t_s, format='iso', scale='utc')
                 print("")
                 final=initial.propagate(p_t_o)
                 #final.plot()
                 print("Orbital elements:")
                 print(final.classical())
                 print("")
                 print("Final co-ordinates:")
                 final.rv()
                 print("")
                 f_orbit=final
                 #print(f_orbit.ecc)
                 plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))



             else:

                 obs_arr=[1, 14, 15, 24, 32, 37, 68, 81, 122, 162, 184, 206, 223]
                 #obs_arr = [int(item) for item in args.obs_array.split(',')]

                 x0 = gauss_method_mpc(args.file_path, obs_arr=obs_arr, bodyname=args.body_name,
                                       r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=False)

                 time=Time(x0[7], format='jd').iso
                 x0=np.asarray(x0)


                 #x0[3:6] = np.deg2rad(x0[3:6])
                 mean_motion=meanmotion(mu_Sun,x0[0])
                 mean_anomaly=meananomaly(mean_motion,time.tdb.jd,x0[2])
                 eccentric_anomaly=eccentricanomaly(x0[1],mean_anomaly)
                 nu=trueanomaly(x0[1],eccentric_anomaly)
                 nu=np.rad2deg(nu)


                 #initial = kep_2_cart(x0[0],x0[1],x0[4],x0[3],x0[5],x0[6],eccentric_anomaly)
                 initial = Orbit.from_classical(Sun, x0[0] * u.AU, x0[1] * u.one, x0[4] * u.deg, x0[5]* u.deg, x0[3] * u.deg, nu * u.deg,epoch=Time(time, format='iso', scale='utc'))
                 print("")
                 print(initial)
                 print(initial.rv())
                 print("\nPropagate to(Enter time in format='iso', scale='utc':")
                 p_t_s=input()
                 p_t_o= Time(p_t_s, format='iso', scale='utc')
                 print("")
                 final=initial.propagate(p_t_o)
                 #final.plot()
                 print("Orbital elements:")
                 print(final.classical())
                 print("")
                 print("Final co-ordinates:")
                 print(final.rv())
                 print("")
                 f_orbit=final
                 #print(f_orbit.ecc)
                 plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))



        elif(opt_1=='2'):
            print("\n1.Find orbit using-")
            print("1.Name.")
            print("2.SPK id.")
            opt_2=input()
            if(opt_2=='1'):
                print("Enter name:")
                str_1=input()
                orbit= neows.orbit_from_name(str_1)
                print(orbit)
                print(orbit.rv())
                frame = OrbitPlotter2D()
                frame.plot(orbit, label=str_1)
                print("\nPropagate to(Enter time in format='iso', scale='utc':")
                p_t_s=input()
                p_t_o= Time(p_t_s, format='iso', scale='utc')
                print("")
                final=orbit.propagate(p_t_o)
                #final.plot()
                print("Orbital elements:")
                print(final.classical())
                print("")
                print("Final co-ordinates:")
                print(final.rv())
                print("")
                f_orbit=final
                #print(f_orbit.ecc)
                plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))




            elif(opt_2=='2'):
                print("Enter SPK id:")
                str_2=input()
                orbit=neows.orbit_from_spk_id(str_2)
                print(orbit)
                print(orbit.rv())
                frame = OrbitPlotter2D()
                frame.plot(orbit, label=str_2)

                print("\nPropagate to(Enter time in format='iso', scale='utc':")
                p_t_s=input()
                p_t_o= Time(p_t_s, format='iso', scale='utc')
                print("")
                final=orbit.propagate(p_t_o)
                #final.plot()
                print("Orbital elements:")
                print(final.classical())
                print("")
                print("Final co-ordinates:")
                print(final.rv())
                print("")
                f_orbit=final
                #print(f_orbit.ecc)
                plotOrbit((f_orbit.a.value),(f_orbit.ecc.value),(f_orbit.inc.value),(f_orbit.raan.value),(f_orbit.argp.value),(f_orbit.nu.value))


            else:
                print("Invalid Input.Exiting...")
                sys.exit()



        else:
            print("Invalid Input.Exiting...")
            sys.exit()
