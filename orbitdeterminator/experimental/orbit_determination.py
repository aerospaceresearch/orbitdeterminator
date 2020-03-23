import numpy as np
import json
import emcee
import time
from datetime import datetime

starttime = time.time()

# constants
mu = 398600.0
c_speedoflight = 300000.0
r_earth = 6378.135


elevation_minimum = 170.0


def get_state_vector(radius_apoapsis, radius_periapsis, inclincation, raan, AoP, time_after_periapsis):


    eccentricity = (radius_apoapsis - radius_periapsis) / (radius_apoapsis + radius_periapsis)
    h_angularmomentuum = np.sqrt(radius_periapsis * (1 + eccentricity * np.cos(0)) * mu)
    T_orbitperiod = 2.0 * np.pi / mu**2 * (h_angularmomentuum / np.sqrt(1 - eccentricity**2))**3


    # newton method to find the true anomaly for a given time after perapsis passage
    Me = 2.0 * np.pi * time_after_periapsis / T_orbitperiod

    if Me > np.pi:
        E0 = Me - eccentricity / 2
    else:
        E0 = Me + eccentricity / 2


    ratioi = 1.0
    while np.abs(ratioi) > 10E-8:
        f_Ei = E0 - eccentricity * np.sin(E0) - Me
        f__Ei = 1 - eccentricity * np.cos(E0)

        ratioi = f_Ei / f__Ei

        E0 = E0 - ratioi


    true_anomaly = np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(E0 / 2)
    true_anomaly = np.arctan(true_anomaly) * 2.0 * 180.0 / np.pi

    if true_anomaly < 0.0:
        true_anomaly = true_anomaly + 360.0


    # degrees to rad again
    inclincation = inclincation * np.pi / 180.0
    raan = raan * np.pi / 180.0
    AoP = AoP * np.pi / 180.0
    true_anomaly = true_anomaly * np.pi / 180.0


    # determination of radius vector on plane of the elipsis
    r = h_angularmomentuum**2 / mu * 1.0 / (1.0 + eccentricity * np.cos(true_anomaly))
    r = np.multiply(r, [np.cos(true_anomaly), np.sin(true_anomaly), 0.0])

    # determination of velocity vector on plane of the elipsis
    v = mu / h_angularmomentuum
    v = np.multiply(v, [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0])


    # stepwise creation of the coordination transform matrix
    Q_Xx = np.array([[np.cos(AoP), np.sin(AoP), 0.0],
                     [-np.sin(AoP), np.cos(AoP), 0.0],
                     [0.0, 0.0, 1.0]])

    Q_Xx = Q_Xx.dot(np.array([[1.0, 0.0, 0.0],
                              [0.0, np.cos(inclincation), np.sin(inclincation)],
                              [0.0, -np.sin(inclincation), np.cos(inclincation)]]))

    Q_Xx = Q_Xx.dot(np.array([[np.cos(raan), np.sin(raan), 0.0],
                              [-np.sin(raan), np.cos(raan), 0.0],
                              [0.0, 0.0, 1.0]]))


    # transforming the radius vectors and velocities into the 3d reference frame.
    R = Q_Xx.T.dot(r)
    V = Q_Xx.T.dot(v)

    return R, V


def geocentricequatorial_into_topocentrichorizon(rel_position, latitude, local_sidereal_time):
    theta = local_sidereal_time * np.pi / 180.0
    phi = latitude * np.pi / 180.0

    Q_Xx = np.array([[-np.sin(theta), +np.cos(theta), 0.0],
            [-np.sin(phi) * +np.cos(theta), -np.sin(phi) * +np.sin(theta), +np.cos(phi)],
            [+np.cos(phi) * +np.cos(theta), +np.cos(phi) * +np.sin(theta), +np.sin(phi)]])

    return Q_Xx.dot(rel_position)


def get_cartesian(lat=None,lon=None, alt=0.0):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = alt # radius of the earth + altitude
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z


def zeroTo360(x):
    if x >= 360.0:
        x = x - int(x/360) * 360.0
    elif x < 0.0:
        x = x - (int(x/360) - 1) * 360.0

    return x


def sidereal(utc, lon):

    ts = utc

    t = datetime.utcfromtimestamp(ts)

    # 1901 <= y <= 2099
    y = t.year
    # 1 <= m <= 12
    m = t.month
    # 1 <= d <= 31
    d = t.day

    # (5.48)
    J0 = 367.0 * y - int(7.0*(y + int((m+9.0)/12.0))/ 4.0) + int((275.0 * m)/9.0) + d + 1721013.5

    hh = t.hour
    mm = t.minute
    ss = t.second + t.microsecond/1000000.0

    UT = hh + mm/60.0 + ss/3600.0

    JD = J0 + UT / 24.0 # (5.47)

    J2000 = 2451545.0
    julian_century_days = 36525.0

    T0 = (J0 - J2000) / julian_century_days

    greenwich_siderual_time_G0 =  100.4606184 + 36000.77004 * T0 + 0.000387933 * T0**2 - 2.583 * 10**-8 * T0**3
    greenwich_siderual_time_G0 = greenwich_siderual_time_G0 - int(greenwich_siderual_time_G0/360) * 360.0

    greenwich_siderual_time_G = greenwich_siderual_time_G0 + 360.98564724 * UT / 24.0


    east_longitude_lambda = lon

    local_sidereal_time = greenwich_siderual_time_G + east_longitude_lambda
    local_sidereal_time = zeroTo360(local_sidereal_time)

    return local_sidereal_time, greenwich_siderual_time_G0


def get_state_sum(r_a, r_p, inc, raan, AoP, tp, td, station, timestamp_min, timestamps, mode, measurements):
    state_sum = 0.0

    distances = measurements["range"]
    r_geos = measurements["position"]
    f_dopplers = measurements["doppler"]
    azimuths = measurements["azimuth"]
    elevations = measurements["elevation"]



    for s in range(len(distances)):

        for t0 in range(len(timestamps[s])):

            time_step = timestamps[s][t0] + td[s]

            rotation, g0 = sidereal(timestamp_min + time_step, station[s][1])

            R, V = get_state_vector(r_a, r_p, inc, raan, AoP, tp + time_step)




            sat_radius = np.linalg.norm(R)
            sat_lat = np.arcsin(R[2] / sat_radius)
            sat_long = np.arctan2(R[1], R[0])
            sat_lat = sat_lat * 180 / np.pi
            sat_long = sat_long * 180 / np.pi

            R_rot = get_cartesian(sat_lat, sat_long - rotation, alt=sat_radius)

            R_station = get_cartesian(station[s][0], rotation, r_earth + station[s][2] / 1000.0)

            sat_r = np.array([R[0] - R_station[0], R[1] - R_station[1], R[2] - R_station[2]])

            elevation = np.arccos(
                (sat_r[0] * R_station[0] + sat_r[1] * R_station[1] + sat_r[2] * R_station[2]) /
                (np.linalg.norm(sat_r) * np.linalg.norm(R_station))
            ) * 180.0 / np.pi



            #############
            #### mode 1: coordinates

            if mode == 1:
                if elevation <= elevation_minimum:
                    state_sum += (r_geos[s][t0][0] - R_rot[0]) ** 2 + \
                            (r_geos[s][t0][1] - R_rot[1]) ** 2 + \
                            (r_geos[s][t0][2] - R_rot[2]) ** 2
                else:
                    state_sum += np.inf



            #############
            #### mode 2: frequencies

            if mode == 2:
                sat_vel = np.linalg.norm(V)
                pointing = R_station - R
                angle = np.arccos(np.dot(pointing, V) / (np.linalg.norm(pointing) * sat_vel)) * 180.0 / np.pi


                f_0 = 137500000.0
                f_d = sat_vel * np.cos(angle * np.pi / 180.0) * f_0 / c_speedoflight


                if elevation <= elevation_minimum:
                    state_sum += (f_d - f_dopplers[s][t0])**2
                else:
                    state_sum += np.inf



            #############
            #### mode 3: azimuth and elevation

            if mode == 3:
                sat_r_rel = geocentricequatorial_into_topocentrichorizon(sat_r, station[s][0], rotation)
                sat_r_rel = np.divide(sat_r_rel, np.linalg.norm(sat_r_rel))
                elevation1 = np.arcsin(sat_r_rel[2])

                sinA = sat_r_rel[0] / np.cos(elevation1)
                cosA = sat_r_rel[1] / np.cos(elevation1)

                if cosA > 1.0:
                    cosA = 1.0
                elif cosA < -1.0:
                    cosA = -1.0

                azimuth = np.arccos(cosA) * 180.0 / np.pi

                if sinA <= 0:
                    azimuth = 360.0 - azimuth
                elevation1 = elevation1 * 180.0 / np.pi


                if elevation <= elevation_minimum:
                    state_sum += (elevation1 - elevations[s][t0])**2 + (azimuth - azimuths[s][t0])**2
                else:
                    state_sum += np.inf


             #############
             #### mode 4: distances
             #### tb


    return -0.5 * state_sum


def log_likelihood1(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements):
    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)

    sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, td, station, timestamp_min, timestamps, mode, measurements)
    return sum


def log_prior1(theta, parameters, finding):
    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)

    if r_earth < r_p and r_p <= r_a and inc >= -90.0 and raan >= -90.0 and AoP > -90.0:
        return 0.0

    return -np.inf


def log_probability1(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements):
    lp = log_prior1(theta, parameters, finding)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood1(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements)



def get_kepler_parameters(theta, parameters, finding):
    key = "r_p"
    r_p = parameters[key]
    if key in finding:
        r_p = theta[finding[key]]

    key = "r_a"
    r_a = parameters[key]
    if key in finding:
        r_a = theta[finding[key]]

    key = "AoP"
    AoP = parameters[key]
    if key in finding:
        AoP = theta[finding[key]]

    key = "inc"
    inc = parameters[key]
    if key in finding:
        inc = theta[finding[key]]

    key = "raan"
    raan = parameters[key]
    if key in finding:
        raan = theta[finding[key]]

    key = "tp"
    tp = parameters[key]
    if key in finding:
        tp = theta[finding[key]]

    key = "td"
    td = parameters[key]
    if "td" in finding:
        for keykey in finding[key].keys():
            td[int(keykey)] = theta[finding[key][keykey]]


    return r_p, r_a, AoP, inc, raan, tp, td


def save_progress(counter, r_p, r_a, AoP, inc, raan, tp, td, sum):
    #### tracking the progress
    progress = {}
    if counter > 0:
        f = open(str(int(starttime)) + '.json')
        progress = json.load(f)
        #data = progress["data"]
    else:
        progress["data"] = {}

    with open(str(int(starttime)) + '.json', 'w') as f:
        #data = {}
        #print(progress)

        progress["data"][str(counter)] = {}
        progress["data"][str(counter)]["r_p"] = r_p
        progress["data"][str(counter)]["r_a"] = r_a
        progress["data"][str(counter)]["AoP"] = AoP
        progress["data"][str(counter)]["inc"] = inc
        progress["data"][str(counter)]["raan"] = raan
        progress["data"][str(counter)]["tp"] = tp
        progress["data"][str(counter)]["td"] = td.tolist()
        progress["data"][str(counter)]["sum"] = sum

        json.dump(progress, f, indent=2)


def find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements):

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability1,
                                    args=(parameters, finding, station, timestamp_min, timestamps, mode, measurements))

    result_b4 = np.zeros(len(pos))

    for i in range(loops):
        pos, prob, state = sampler.run_mcmc(pos, walks)

        samples = sampler.chain[:, 0:, :].reshape((-1, ndim))

        result = np.percentile(samples, [16, 50, 84], axis=0)


        for r in range(len(result[1])):


            for key in finding.keys():
                if finding[key] == r:
                    parameters_name = key

                #if finding[key].keys() > 1:
                #    for keykey in finding[key].keys():
                #        if finding[key][keykey] == r:
                #            parameters_name = r


            ###### orbit
            dif = 0
            for key in finding.keys():
                if finding[key] == r:
                    #print(key, finding[key], parameters[key])
                    dif = orbit_data[key]

                if key is "td":
                    for keykey in finding[key].keys():
                        if finding[key][keykey] == r:
                            #print(key, finding[key][keykey], parameters[key][int(keykey)])
                            dif = orbit_data[key][int(keykey)]
            dif = dif - result[1][r]



            print(parameters_name,":\t", result[1][r], result[0][r] - result[1][r], result[2][r] - result[1][r], (result[1][r] - result_b4[r])/result[1][r], dif)
            result_b4[r] = result[1][r]


        theta = []
        for r in range(len(result[1])):
            theta.append(result[1][r])

        r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)
        print(r_p, r_a, AoP, inc, raan, tp, td)


        sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, td, station, timestamp_min, timestamps, mode, measurements)

        save_progress(counter, r_p, r_a, AoP, inc, raan, tp, td, sum)

        print(counter+1, i+1, "/", loops, sum, time.time() - starttime, mode, elevation_minimum)
        print("")

        counter += 1
        sampler.reset()


    return result, counter


def main():

    # settings
    loops = 12
    walks = 70
    mode = 1

    counter = 0


    f = open('inputdata.json')
    observations = json.load(f)


    n_observations = len(observations["observations"])

    station = []

    timestamps = []
    distances = []
    f_dopplers = []
    r_geos = []
    azimuths = []
    elevations = []

    station_gnss = np.zeros(n_observations)
    station_gnss_mix = []

    print("pre-work: loading in the measurement data")
    for n in range(n_observations):
        lon = observations["observations"][n]["ground_station"]["lon"]
        lat = observations["observations"][n]["ground_station"]["lat"]
        alt = observations["observations"][n]["ground_station"]["alt"]
        station.append([lat, lon, alt])

        distance = []
        timestamp = []
        r_geo = []
        f_doppler = []
        azimuth = []
        elevation = []

        for n1 in range(len(observations["observations"][n]["data_stream"])):

            distance.append(observations["observations"][n]["data_stream"][n1]["gs_distance"])
            r_geo.append(observations["observations"][n]["data_stream"][n1]["r_geo"])
            f_doppler.append(observations["observations"][n]["data_stream"][n1]["f_doppler"])
            azimuth.append(observations["observations"][n]["data_stream"][n1]["azimuth"])
            elevation.append(observations["observations"][n]["data_stream"][n1]["elevation"])
            if observations["observations"][n]["data_stream"][n1]["time_gnss"] is None:
                timestamp.append(observations["observations"][n]["data_stream"][n1]["time_unix"])
                station_gnss[n] = 0
            else:
                timestamp.append(observations["observations"][n]["data_stream"][n1]["time_gnss"])
                station_gnss[n] = 1

        timestamps.append(timestamp)
        distances.append(distance)
        r_geos.append(r_geo)
        f_dopplers.append(f_doppler)
        azimuths.append(azimuth)
        elevations.append(elevation)


    measurements = {}
    measurements["range"] = distances
    measurements["position"] = r_geos
    measurements["doppler"] = f_dopplers
    measurements["azimuth"] = azimuths
    measurements["elevation"] = elevations


    orbit_data = []
    if "orbit" in observations:
        orbit_data = observations["orbit"]

    # creating look up table which station is not synched by a gnss and the system delay time needs to be determined
    for i in range(len(station_gnss)):
        if station_gnss[i] == 0:
            station_gnss_mix.append(i)

    print("ssss", station_gnss_mix)



    print("pre-work: finding the earliest time...")
    timestamp_min = np.max(np.max(timestamps))
    for s in range(len(timestamps)):
        for stamps in range(len(timestamps[s])):
            if timestamp_min > timestamps[s][stamps]:
                timestamp_min = timestamps[s][stamps]

    print("timestamp_min", timestamp_min)

    print("pre-work: ...and correcting it. Time of perige passing is until the timestamp_min")
    for s in range(len(timestamps)):
        for stamps in range(len(timestamps[s])):
            timestamps[s][stamps] = timestamps[s][stamps] - timestamp_min


    print("")
    print("## Determination1: Finding the orbit parameters")

    # distributing the initial positions within the scope of AoP, inclination and raan.
    pos = []
    for index_aop in range(10):
        for index_inclination in range(7):
            for index_raan in range(10):

                # the following ranges for the random initial parameter are gut feelings.
                # for later automation, this needs to be configurable by the user
                r_p0 = np.random.randint(6371 + 300, 6371 + 1000) / 1.0
                r_a0 = np.random.randint(r_p0, 6371 + 1000) / 1.0
                AoP0 = float(360.0 / 10.0 * index_aop)
                inc0 = float(180.0 / 7.0 * index_inclination)
                raan0 = float(360.0 / 10.0 * index_raan)


                eccentricity = (r_a0 - r_p0) / (r_a0 + r_p0)
                h_angularmomentuum = np.sqrt(r_p0 * (1 + eccentricity * np.cos(0)) * mu)
                T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

                tp0 = np.random.randint(0, int(T_orbitperiod))
                pos.append([r_p0, r_a0, AoP0, inc0, raan0, tp0])

    td = np.zeros(len(station_gnss))


    parameters = {
        "r_p": 0,
        "r_a": 0,
        "AoP": 0,
        "inc": 0,
        "raan": 0,
        "tp": 0,
        "td": td
    }

    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5#,
        #"td": {"0": 6,
        #       "2": 7}
    }


    #the overall number of walkers need to be multiple of 2.
    ndim = 6
    nwalkers = len(pos)


    # initiating the optimization.
    # every loop will provide an output.
    # walks is the number of steps/hops a walker will be doing.


    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")
    # finding the orbits now...
    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements)



    ## finding the right argument of perigee
    print("## Global Search: shaking up the system, maybe we're stuck in local optimum")

    theta = []
    for r in range(len(result[1])):
        theta.append(result[1][r])

    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)


    print("we have so far:",
          "perigee=", r_p,
          "apogee=", r_a,
          "AoP=", AoP,
          "inclination=", inc,
          "raan=", raan,
          "time after perigee=", tp,
          "time delay at stations=", td)

    global_grid_dif = []

    iter_AoP = 180
    iter_raan = 90

    for i in range(0, iter_AoP, 1):
        raan_dif = []
        for j in range(0, iter_raan, 1):
            AoP_test = float(i * 360 / iter_AoP)
            # inc_test = 0.0
            raan_test = float(j * 360 / iter_raan)
            dif = get_state_sum(r_a, r_p, inc, raan, AoP, tp, td, station, timestamp_min, timestamps, mode, measurements)
            raan_dif.append(dif)

        global_grid_dif.append(raan_dif)

    optimum_index = np.unravel_index(np.argmax(global_grid_dif), np.shape(global_grid_dif))

    print("global search says: AoP=", optimum_index[0] * 360 / iter_AoP, "raan=", optimum_index[1] * 360 / iter_raan)
    print("best dif value=", global_grid_dif[optimum_index[0]][optimum_index[1]])
    print("")

    AoP_set = optimum_index[0] * 360 / iter_AoP
    raan_set = optimum_index[1] * 360 / iter_raan



    #### again
    print("## Determination1: Finding the orbit parameters, again")
    print("")

    nwalkers = 15 * 32
    ndim = 6

    pos = []
    for walk in range(nwalkers):
        r_p0 = r_p + np.random.randint(-100000, 100000) / 1000.0
        r_a0 = r_a + np.random.randint(-100000, 100000) / 1000.0
        if r_a0 < r_p0:
            r_a0 = r_p0
        AoP0 = AoP_set + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
        inc0 = inc + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
        raan0 = raan_set + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
        tp0 = tp + np.random.randint(-10000, 10000) / 100.0
        pos.append([r_p0, r_a0, AoP0, inc0, raan0, tp0])

    td = td
    print("tdddd", td)
    parameters = {
        "r_p": 0,
        "r_a": 0,
        "AoP": 0,
        "inc": 0,
        "raan": 0,
        "tp": 0,
        "td": td
    }

    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5  # ,
        # "td": {"0": 6,
        #       "2": 7}
    }


    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")

    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements)

    theta = []
    for r in range(len(result[1])):
        theta.append(result[1][r])

    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)



    print("## Determination1.1: Finding the orbit parameters, again again")
    print("")

    nwalkers, ndim = 15 * 32, 6

    pos = []
    for index_aop in range(10):
        for index_tp in range(10):
            for index_raan in range(10):
                # the following ranges for the random initial parameter are gut feelings.
                # for later automation, this needs to be configurable by the user
                r_p0 = r_p + np.random.randint(-10000, 10000) / 1000.0
                r_a0 = r_a + np.random.randint(-10000, 10000) / 1000.0
                if r_a0 < r_p0:
                    r_a0 = r_p0
                AoP0 = float(360.0 / 10.0 * index_aop)
                inc0 = inc + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
                raan0 = float(360.0 / 10.0 * index_raan)

                eccentricity = (r_a - r_p) / (r_a + r_p)
                h_angularmomentuum = np.sqrt(r_p * (1 + eccentricity * np.cos(0)) * mu)
                T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

                tp0 = float(T_orbitperiod / 10 * index_tp)
                pos.append([r_p0, r_a0, AoP0, inc0, raan0, tp0])

    td = td
    print("tdddd", td)
    parameters = {
        "r_p": 0,
        "r_a": 0,
        "AoP": 0,
        "inc": 0,
        "raan": 0,
        "tp": 0,
        "td": td
    }

    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5  # ,
        # "td": {"0": 6,
        #       "2": 7}
    }


    # the overall number of walkers need to be multiple of 2.
    nwalkers = len(pos)

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")

    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements)

    theta = []
    for r in range(len(result[1])):
        theta.append(result[1][r])

    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)


    print("## Determination2: Finding the system time delay for each station")
    #################
    if len(station_gnss_mix) > 0 and len(station_gnss) > len(station_gnss_mix):

        nwalkers = 30 * 32
        ndim = 6 + len(station_gnss_mix)

        pos = []
        for walk in range(nwalkers):
            tmp = []
            r_p0 = r_p + np.random.randint(-10000, 10000) / 1000.0
            r_a0 = r_a + np.random.randint(-10000, 10000) / 1000.0
            if r_a0 < r_p0:
                r_a0 = r_p0
            AoP0 = AoP + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
            inc0 = inc + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
            raan0 = raan + np.random.randint(int(-20) * 1000, int(+20) * 1000) / 1000.0
            tp0 = tp + np.random.randint(-10000, 10000) / 100.0
            tmp.append(r_p0)
            tmp.append(r_a0)
            tmp.append(AoP0)
            tmp.append(inc0)
            tmp.append(raan0)
            tmp.append(tp0)

            for i in range(len(station_gnss_mix)):
                t_delay = np.random.randint(-8000, 8000) / 1000.0
                tmp.append(t_delay)

            pos.append(tmp)

        td = np.zeros(len(station_gnss))
        print("tdddd", td)
        parameters = {
            "r_p": 0,
            "r_a": 0,
            "AoP": 0,
            "inc": 0,
            "raan": 0,
            "tp": 0,
            "td": td
        }

        finding = {
            "r_p": 0,
            "r_a": 1,
            "AoP": 2,
            "inc": 3,
            "raan": 4,
            "tp": 5  # ,
            # "td": {"0": 6,
            #       "2": 7}
        }

        finding_td = {}
        counter_td = len(finding)
        for i in range(len(station_gnss_mix)):
            finding_td[str(int(station_gnss_mix[i]))] = counter_td
            counter_td += 1

        finding["td"] = finding_td


        print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
        print("please wait now...")

        result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements)

    theta = []
    for r in range(len(result[1])):
        theta.append(result[1][r])

    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)



    print("## Determination3: Finding the tp and AoP parameters")
    nwalkers, ndim = 20 * 32, 4

    # distributing the initial positions within the scope of AoP, inclination and raan.
    pos = []
    for i in range(nwalkers):
        eccentricity = (r_a - r_p) / (r_a + r_p)
        h_angularmomentuum = np.sqrt(r_p * (1 + eccentricity * np.cos(0)) * mu)
        T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

        AoP0 = np.random.randint(int(AoP * 0.8) * 100, int(AoP * 1.2) * 100) / 100.0
        tp0 = np.random.randint(0, int(T_orbitperiod * 10)) / 10.0
        r_p0 = int(r_p) + np.random.randint(-200, 200) / 100.0
        r_a0 = int(r_a) + np.random.randint(-200, 200) / 100.0
        if r_a0 < r_p:
            r_a0 = r_p0
            print("equal r_a r_p now")
        pos.append([tp0, AoP0, r_p0, r_a0])

    parameters = {
        "r_p": r_p,
        "r_a": r_a,
        "AoP": AoP,
        "inc": inc,
        "raan": raan,
        "tp": tp,
        "td": td
    }

    finding = {
        "tp": 0,
        "AoP": 1,
        "r_p": 2,
        "r_a": 3
    }

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")
    # finding the orbits now...
    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, orbit_data, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements)

    theta = []
    for r in range(len(result[1])):
        theta.append(result[1][r])

    r_p, r_a, AoP, inc, raan, tp, td = get_kepler_parameters(theta, parameters, finding)

    print(r_p, r_a, AoP, inc, raan, tp, td)


if __name__== "__main__":

    main()