import numpy as np
import json
import emcee
import time

starttime = time.time()

# constants
mu = 398600.0


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


def get_cartesian(lat=None,lon=None, alt=0.0):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371.0 + alt # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z


def create_stations_position(station_geo):

    gs_x, gs_y, gs_z = get_cartesian(station_geo[0], station_geo[1], station_geo[2])

    station = np.array([gs_x, gs_y, gs_z])

    return station


def state_sum1(timestamps, distances, f_dopplers, r_a, r_p, inc, raan, AoP, tp, station, td):
    sum1 = 0.0

    for s in range(len(distances)):

        for t0 in range(len(timestamps[s])):

            R, V = get_state_vector(r_a, r_p, inc, raan, AoP, timestamps[s][t0] + tp + td[s])

            sum1 += (distances[s][t0] - ((R[0] - station[s][0])** 2 + (R[1] - station[s][1])** 2 + (R[2] - station[s][2])** 2) ** 0.5)**2


    return -0.5 * sum1


def log_likelihood1(theta, station, station_gnss_mix, timestamps, distances, f_dopplers):
    td = np.zeros(len(station))
    r_p = theta[0]
    r_a = theta[1]
    AoP = theta[2]
    inc = theta[3]
    raan = theta[4]
    tp = theta[5]
    for i in range(6, len(theta), 1):
        td[station_gnss_mix[i-6]] = theta[i]

    sum = state_sum1(timestamps, distances, f_dopplers, r_a, r_p, inc, raan, AoP, tp, station, td)
    return sum


def log_prior1(theta):
    #td = np.zeros(len(station))
    r_p = theta[0]
    r_a = theta[1]
    AoP = theta[2]
    inc = theta[3]
    raan = theta[4]
    tp = theta[5]

    if 6371.0 < r_p and r_p < r_a and 0.0 <= inc <= 180.0 and 0.0 <= raan <= 360.0:
        return 0.0

    return -np.inf


def log_probability1(theta, station, station_gnss_mix, timestamps, distances, f_dopplers):
    lp = log_prior1(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood1(theta, station, station_gnss_mix, timestamps, distances, f_dopplers)



def find_orbit(nwalkers, ndim, pos, station, station_gnss_mix, timestamps, distances, f_dopplers, loops, walks,
               counter):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability1,
                                    args=(station, station_gnss_mix, timestamps, distances, f_dopplers))

    for i in range(loops):
        pos, prob, state = sampler.run_mcmc(pos, walks)

        samples = sampler.chain[:, 0:, :].reshape((-1, ndim))

        result = np.percentile(samples, [16, 50, 84], axis=0)

        print(counter, i, time.time() - starttime)
        for r in range(len(result[1])):
            print(result[1][r], result[0][r] - result[1][r], result[2][r] - result[1][r])
        print("")
        counter += 1

        sampler.reset()

    pos, prob, state = sampler.run_mcmc(pos, walks)

    samples = sampler.chain[:, 0:, :].reshape((-1, ndim))

    result = np.percentile(samples, [16, 50, 84], axis=0)

    print(counter, loops, time.time() - starttime)
    for r in range(len(result[1])):
        print(result[1][r], result[0][r] - result[1][r], result[2][r] - result[1][r])
    print("")

    counter += 1

    return result, counter


def main():

    f = open('inputdata.json')
    observations = json.load(f)


    n_observations = len(observations["observations"])

    station = []

    timestamps = []
    distances = []
    f_dopplers = []

    station_gnss = np.zeros(n_observations)
    station_gnss_mix = []

    print("pre-work: loading in the measurement data")
    for n in range(n_observations):
        lon = observations["observations"][n]["ground_station"]["lon"]
        lat = observations["observations"][n]["ground_station"]["lat"]
        alt = observations["observations"][n]["ground_station"]["alt"]
        station.append(create_stations_position([lat, lon, alt]))

        distance = []
        timestamp = []

        for n1 in range(len(observations["observations"][n]["data_stream"])):

            distance.append(observations["observations"][n]["data_stream"][n1]["gs_distance"])
            if observations["observations"][n]["data_stream"][n1]["time_gnss"] is None:
                timestamp.append(observations["observations"][n]["data_stream"][n1]["time_unix"])
                station_gnss[n] = 0
            else:
                timestamp.append(observations["observations"][n]["data_stream"][n1]["time_gnss"])
                station_gnss[n] = 1

        timestamps.append(timestamp)
        distances.append(distance)

    # creating look up table which station is not synched by a gnss and the system delay time needs to be determined
    for i in range(len(station_gnss)):
        if station_gnss[i] == 0:
            station_gnss_mix.append(i)


    print("pre-work: finding the earliest time...")
    timestamp_min = np.max(np.max(timestamps))
    for s in range(len(timestamps)):
        for stamps in range(len(timestamps[s])):
            if timestamp_min > timestamps[s][stamps]:
                timestamp_min = timestamps[s][stamps]

    print("pre-work: ...and correcting it. Time of perige passing is until the timestamp_min")
    for s in range(len(timestamps)):
        for stamps in range(len(timestamps[s])):
            timestamps[s][stamps] = timestamps[s][stamps] - timestamp_min


    print("")
    print("## Determination1: Finding the orbit parameters")
    nwalkers, ndim = 12 * 32, 6

    # distributing the initial positions within the scope of AoP, inclination and raan.
    pos = []
    for index_aop in range(10):
        for index_inclination in range(7):
            for index_raan in range(10):

                # the following ranges for the random initial parameter are gut feelings.
                # for later automation, this needs to be configurable by the user
                r_p = np.random.randint(6371 + 400, 6371 + 1100) / 1.0
                r_a = np.random.randint(r_p, 6371 + 1300) / 1.0
                AoP = float(360.0 / 10.0 * index_aop)
                inc = float(180.0 / 7.0 * index_inclination)
                raan = float(360.0 / 10.0 * index_raan)
                ts = np.random.randint(0, 8000)
                pos.append([r_p, r_a, AoP, inc, raan, ts])

    #the overall number of walkers need to be multiple of 2.
    nwalkers = len(pos)

    counter = 0
    # initiating the optimization.
    # every loop will provide an output.
    # walks is the number of steps/hops a walker will be doing.
    loops = 9
    walks = 130

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")
    # finding the orbits now...
    result, counter = find_orbit(nwalkers, ndim, pos, station, station_gnss_mix, timestamps, distances, f_dopplers,
                                 loops, walks, counter)



    ## finding the right argument of perigee
    print("## Global Search: shaking up the system, maybe we're stuck in local optimum")
    r_p0 = result[1][0]
    r_a0 = result[1][1]
    AoP0 = result[1][2]
    inc0 = result[1][3]
    raan0 = result[1][4]
    tp = result[1][5]
    td = np.zeros(len(station))

    print("we have so far:",
          "perigee=", r_p0,
          "apogee=", r_a0,
          "AoP=", AoP0,
          "inclination=", inc0,
          "raan=", raan0,
          "time after perigee=", tp)

    global_grid_dif = []

    iter_AoP = 180
    iter_raan = 90

    for i in range(0, iter_AoP, 1):
        raan_dif = []
        for j in range(0, iter_raan, 1):
            AoP_test = float(i * 360 / iter_AoP)
            # inc_test = 0.0
            raan_test = float(j * 360 / iter_raan)
            dif = state_sum1(timestamps, distances, f_dopplers, r_a0, r_p0, inc0, raan_test, AoP_test, tp, station, td)
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

    nwalkers, ndim = 15 * 32, 6

    pos = []
    for walk in range(nwalkers):
        r_p = np.random.randint(int(result[0][0]), int(result[2][0])) / 1.0
        r_a = np.random.randint(int(result[0][1]), int(result[2][1])) / 1.0
        AoP = np.random.randint((int(AoP_set * 0.8) - 1) * 10, (int(AoP_set * 1.2) + 1) * 10) / 10.0
        inc = np.random.randint(int(result[1][3]) - 10, int(result[1][3]) + 10) / 1.0
        raan = np.random.randint((int(raan_set * 0.8) - 1) * 10, (int(raan_set * 1.2) + 1) * 10) / 10.0
        tp = np.random.randint(int(result[0][5]), int(result[2][5]))
        pos.append([r_p, r_a, AoP, inc, raan, tp])


    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")

    result, counter = find_orbit(nwalkers, ndim, pos, station, station_gnss_mix, timestamps, distances, f_dopplers,
                                 loops, walks, counter)


    print("## Determination2: Finding the system time delay for each station")
    #################
    nwalkers, ndim = 28 * 32, 6 + len(station_gnss_mix)

    pos = []
    for walk in range(nwalkers):
        tmp = []
        r_p = int(result[1][0]) + np.random.randint(-3, 3)
        r_a = int(result[1][1]) + np.random.randint(-3, 3)
        AoP = int(result[1][2]) + np.random.randint(-5, 5)
        inc = int(result[1][3]) + np.random.randint(-2, 2)
        raan = int(result[1][4]) + np.random.randint(-5, 5)
        tp = int(result[1][5]) + np.random.randint(-15, 15)
        tmp.append(r_p)
        tmp.append(r_a)
        tmp.append(AoP)
        tmp.append(inc)
        tmp.append(raan)
        tmp.append(tp)

        for i in range(len(station_gnss_mix)):
            t_delay = np.random.randint(-80, 80) / 10.0
            tmp.append(t_delay)

        pos.append(tmp)


    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")

    result, counter = find_orbit(nwalkers, ndim, pos, station, station_gnss_mix, timestamps, distances, f_dopplers,
                                 loops, walks, counter)


if __name__== "__main__":

    main()