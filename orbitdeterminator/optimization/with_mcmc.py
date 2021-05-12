import numpy as np
import emcee

from astropy.time import Time
from datetime import datetime, timedelta
from astropy.coordinates import Longitude, SkyCoord
from astropy import units as uts
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import astropy.units as u

from skyfield.api import EarthSatellite
from skyfield.api import load, wgs84

import matplotlib.pylab as plt
import time

from sgp4.api import Satrec, WGS72
from sgp4 import exporter

mu = 398600.0

plotnames = "ggg_"
starttime = time.time()


def filler(input, fplaces, ffiller, bplaces):
    split = input.split(".")

    tmp = ""
    if len(split[0]) < fplaces:

        for i in range(fplaces - len(split[0])):
            tmp += ffiller
    tmp += split[0]

    tmp += "."
    tmp += split[1]

    if len(split[1]) < bplaces:

        for i in range(bplaces - len(split[1])):
            tmp += "0"

    return tmp

def tle_bstar(bstar):
    exp_max = 8
    x = float(bstar) / 10**exp_max

    if x == 0.0:
        #out = "+33299-6"
        out = "        "

    else:
        exp = np.floor(np.log10(np.abs(x)))
        c = 10**(exp+1)

        if np.sign(x) == 1.0:
            s = "+"
        else:
            s = "-"

        number = x/c
        number = str(number)
        number = number.split(".")[1]
        for i in range(5-len(number)):
            number += "0"
        number = number[:5]

        sign = int(exp)
        if sign < 0.0:
            out = s + number + str(sign)
        if sign >= 0.0:
            out = s + number + "+" + str(sign)

    return out


def tle_bstar1(bstar):
    exp_max = 8
    x = float(bstar) / 10**exp_max

    if x == 0.0:
        #out = "+33299-6"
        out = "        "

    else:
        exp = np.floor(np.log10(np.abs(x)))
        c = 10**(exp+1)

        if np.sign(x) == 1.0:
            s = "+"
        else:
            s = "-"

        number = x/c
        number = str(number)
        number = number.split(".")[1]
        for i in range(5-len(number)):
            number += "0"
        number = number[:5]

        sign = int(exp)
        if sign < 0.0:
            out = s + number + str(sign)
        if sign >= 0.0:
            out = s + number + "+" + str(sign)

    return out

def tle_mod10(line):
    sum = 0

    line_out = ""
    for i in range(len(line)-1):

        if line[i].isnumeric():
            sum += int(line[i])

        if line[i] == "-":
            sum += 1

        line_out += line[i]

    line_out += str(np.mod(sum, 10))

    return line_out


def tlestuf(epoch, inc, ecc, aop, raan, me, meanmove, bstar):
    observing_time = Time(str(epoch), format="unix", scale="utc")
    startofday = Time(str(observing_time.to_datetime().timetuple().tm_year)+"-"+str(observing_time.to_datetime().timetuple().tm_mon)+"-"+str(observing_time.to_datetime().timetuple().tm_mday), scale="utc").unix
    fractionofday = filler(str((epoch-startofday)/(24.0*3600.0)),0 ," " , 8)
    year = str(observing_time.to_datetime().timetuple().tm_year)[2:5]
    dayofyear = str(observing_time.to_datetime().timetuple().tm_yday)
    dayofyear = filler(str(dayofyear)+".",3,"0", 0)

    sgp4_epoch = year+dayofyear+fractionofday[2:10]
    sgp4_inc = filler(str(inc), 3, " ", 4)[:8]
    sgp4_ecc = filler(ecc, 1, " ", 7)[2:9]
    sgp4_aop = filler(str(aop), 3, " ", 4)[:8]
    sgp4_raan = filler(str(raan), 3, " ", 4)[:8]
    sgp4_me = filler(str(me), 3, " ", 4)[:8]
    sgp4_meanmove = filler(str(meanmove), 2, " ", 8)[:11]
    sgp4_revolutions = "00000" # any
    sgp4_checksum = "0" # removed, any
    sgp4_bstar = tle_bstar(bstar)

    s = '1 12345U 00000A   '+sgp4_epoch+' -.00000180  00000-0 '+sgp4_bstar+' 0  0000'
    t = '2 12345 ' + sgp4_inc + ' ' + sgp4_raan + ' ' + sgp4_ecc + ' ' + sgp4_aop + ' ' + sgp4_me + ' ' + sgp4_meanmove + sgp4_revolutions + sgp4_checksum

    s = tle_mod10(s)
    t = tle_mod10(t)

    return s, t


def get_satrec(satnum, epoch, ecco, argpo, inclo, mo, no_kozai, nodeo, bstar, ndot, nddot=0.0):

    time_ref = -631238400.0  # Time("1949-12-31T00:00:00", format="isot", scale="utc").unix

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,  # gravity model
        'i',  # 'a' = old AFSPC mode, 'i' = improved mode
        satnum,  # satnum: Satellite number
        (epoch - time_ref) / (24.0 * 3600.0),  # epoch: days since 1949 December 31 00:00 UT
        bstar,  # bstar: drag coefficient (/earth radii)
        ndot,  # ndot: ballistic coefficient (revs/day)
        nddot,  # nddot: second derivative of mean motion (revs/day^3)
        ecco,  # ecco: eccentricity
        argpo * np.pi / 180.0,  # argpo: argument of perigee (radians)
        inclo * np.pi / 180.0,  # inclo: inclination (radians)
        mo * np.pi / 180.0,  # mo: mean anomaly (radians)
        no_kozai * 2.0 * np.pi / (24.0 * 60.0),  # no_kozai: mean motion (radians/minute)
        nodeo * np.pi / 180.0,  # nodeo: right ascension of ascending node (radians)
    )

    return satrec


def zeroTo360(x):
    if x >= 360.0:
        x = x - int(x/360) * 360.0
    elif x < 0.0:
        x = x - (int(x/360) - 1) * 360.0

    return x

def zeroTo180(x):
    if x >= 180.0:
        x = x - int(x/180) * 180.0
    elif x < 0.0:
        x = x - (int(x/180) - 1) * 180.0

    return x


def get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements):

    # putting in the measurements
    ras = measurements["rightascension"]
    decs = measurements["declination"]
    el = measurements["el"]
    az = measurements["az"]
    ranging = measurements["range"]
    doppler = measurements["doppler"]
    satellite_pos = measurements["satellite_pos"]

    # preparing the orbit track that is being simulated and used for comparing to the measurements
    track = np.zeros_like(satellite_pos)


    # preparing the orbit parameter needed for the simulated orbit
    eccentricity = (r_a - r_p) / (r_a + r_p)
    h_angularmomentuum = np.sqrt(r_p * (1.0 + eccentricity * np.cos(0)) * mu)
    T_orbitperiod = 2.0 * np.pi / mu ** 2.0 * (h_angularmomentuum / np.sqrt(1.0 - eccentricity ** 2)) ** 3

    me = tp * (2.0 * np.pi) / T_orbitperiod * 180.0/np.pi
    me = zeroTo360(me)
    AoP = zeroTo360(AoP)
    raan = zeroTo360(raan)
    n = 24.0 * 3600.0 / T_orbitperiod


    # preparing the orbit by putting in the orbit parameters
    ts = load.timescale()

    satnum = 0
    epoch = timestamp_min
    ecco = eccentricity
    argpo = AoP
    inclo = inc
    mo = me
    no_kozai = n
    nodeo = raan
    ndot = 0.0
    satrec = get_satrec(satnum, epoch, ecco, argpo, inclo, mo, no_kozai, nodeo, bstar, ndot, nddot=0.0)
    satellite = EarthSatellite.from_satrec(satrec, ts)


    # now for each station s the measurements are being iterated through by its measurements.
    # for each measurement, the orbit state is calculated based on the timestamp t0
    for s in range(len(timestamps)):

        for t0 in range(len(timestamps[s])):

            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            R1 = satellite.at(t).position.km
            V1 = satellite.at(t).velocity.km_per_s

            #R = np.array(R1)
            #V = np.array(V1)

            track[s][t0][0] = R1[0]
            track[s][t0][1] = R1[1]
            track[s][t0][2] = R1[2]

            #############

            #if mode == 0:
            #    #state_sum += (R[0] - satellite_pos[s][t0][0]) ** 2 + \
            #    #             (R[1] - satellite_pos[s][t0][1]) ** 2 + \
            #    #             (R[2] - satellite_pos[s][t0][2]) ** 2

    # now we just do a simple Root-mean-square of the measurement positions
    # and the orbit positions based on the simulation
    rms = np.subtract(track, satellite_pos)
    rms_sum = np.sum(np.square(rms))

    return -0.5 * rms_sum



def get_kepler_parameters(theta, parameters, finding, orbit):

    key = "r_a"
    r_a = parameters[key]
    if key in finding:
        r_a = theta[finding[key]]

    key = "r_p"
    r_p = parameters[key]
    if key in finding:
        r_p = theta[finding[key]]
    if orbit == 0:
        r_p = r_a

    key = "AoP"
    AoP = parameters[key]
    if key in finding:
        AoP = theta[finding[key]]
    if orbit == 0:
        AoP = 0.0
    AoP = AoP - 360.0 * np.floor(AoP / 360.0)

    key = "inc"
    inc = parameters[key]
    if key in finding:
        inc = theta[finding[key]]

    key = "raan"
    raan = parameters[key]
    if key in finding:
        raan = theta[finding[key]]
    raan = raan - 360.0 * int(raan / 360.0)

    key = "tp"
    tp = parameters[key]
    if key in finding:
        tp = theta[finding[key]]

    key = "bstar"
    bstar = parameters[key]
    if key in finding:
        bstar = theta[finding[key]]

    key = "td"
    td = parameters[key]
    if "td" in finding:
        for keykey in finding[key].keys():
            td[int(keykey)] = theta[finding[key][keykey]]


    return r_p, r_a, AoP, inc, raan, tp, bstar, td


def log_likelihood(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit):
    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)

    sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements)
    return sum


def log_prior(theta, parameters, finding, orbit):
    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)

    r_earth = 6378.0

    if r_earth < r_p and r_p <= r_a and\
            inc >= 0.0 and inc <= 180.0 and\
            raan >= -90.0 and \
            AoP > -90.0 and \
            np.abs(bstar) <= 1.0:

        return 0.0

    return -np.inf


def log_probability(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit):
    lp = log_prior(theta, parameters, finding, orbit)

    if not np.isfinite(lp):
        return -np.inf

    sum = log_likelihood(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit)

    if np.isnan(sum) == True:
        return -np.inf

    return lp + sum


def find_orbit(nwalkers, ndim, pos, parameters, finding, loops, walks, counter, station, timestamp_min, timestamps, mode, measurements, orbit):

    # preparing the optimizer
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit))


    # preparing the storage of the best result of all loops
    results_min = []
    for i in range(ndim):
        results_min.append([])
    residual_min = []

    result_b4 = np.zeros(len(pos))
    b4 = 0
    b4_tle_line1 = ""
    b4_tle_line2 = ""
    b4_result = []

    # now we can run the loops at each iteration we have the EMCEE results
    for i in range(loops):
        #start the optimization with emcee
        pos, prob, state = sampler.run_mcmc(pos, walks, progress=True)

        # extracting the orbit parameters back from the resulting POS.
        # in this case, we just use the best result, which is argmax of POS
        r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(pos[np.argmax(prob)], parameters, finding, orbit)

        eccentricity = (r_a - r_p) / (r_a + r_p)
        h_angularmomentuum = np.sqrt(r_p * (1.0 + eccentricity * np.cos(0)) * mu)
        T_orbitperiod = 2.0 * np.pi / mu ** 2.0 * (h_angularmomentuum / np.sqrt(1.0 - eccentricity ** 2)) ** 3

        me = tp * (2.0 * np.pi) / T_orbitperiod * 180.0 / np.pi
        me = zeroTo360(me)
        AoP = zeroTo360(AoP)
        raan = zeroTo360(raan)
        n = 24.0 * 3600.0 / T_orbitperiod

        satnum = 0
        epoch = timestamp_min
        ecco = eccentricity
        argpo = AoP
        inclo = inc
        mo = me
        no_kozai = n
        nodeo = raan
        ndot = 0.0
        satrec = get_satrec(satnum, epoch, ecco, argpo, inclo, mo, no_kozai, nodeo, bstar, ndot, nddot=0.0)
        tle_line1, tle_line2 = exporter.export_tle(satrec)

        if i == 0:
            b4 = np.max(prob)
            b4_tle_line1 = tle_line1
            b4_tle_line2 = tle_line2
            b4_result = pos[np.argmax(prob)]
        else:
            if np.max(prob) > b4:
                b4 = np.max(prob)
                b4_tle_line1 = tle_line1
                b4_tle_line2 = tle_line2
                b4_result = pos[np.argmax(prob)]

        #print("prob", np.max(prob), np.argmax(prob), b4)
        #print(pos[np.argmax(prob)])

        #save_progress(plotnames, counter, r_p, r_a, AoP, inc, raan, tp, bstar, td, np.max(prob), s1, t1, mode, filename_1)
        #save_progress_pos(plotnames, pos, prob, state)

        '''
        for ii in range(ndim):
            results_min[ii].append(pos[np.argmax(prob)][ii])
        residual_min.append(np.max(prob))
        plt.plot(np.abs(residual_min))
        plt.yscale("log")
        plt.grid()
        plt.savefig(plotnames+"_residual")
        plt.clf()
        plt.plot(results_min[0])
        plt.plot(results_min[1])
        plt.grid()
        plt.savefig(plotnames+"_radius")
        plt.clf()
        print(np.mean(results_min[0]), np.mean(results_min[1]))
        '''

        #get_state_plot(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements)


        samples = sampler.chain[:, 0:, :].reshape((-1, ndim))

        result_percentile = np.percentile(samples, [16, 50, 84], axis=0)

        #import corner
        #import matplotlib.pyplot as plt
        #flat_samples = samples
        #fig = corner.corner(flat_samples)
        #plt.show()

        #print(samples.shape)
        #print(len(samples))
        #print(samples[:, 0])
        #print(samples[:, 1])
        #print(samples[:, 2])
        #print(samples[:, 3])
        #print(samples[:, 4])
        #print(samples[:, 5])

        #plt.plot(samples[:, 0])
        #plt.plot(samples[:, 1])
        #plt.show()
        #print(np.mean(samples[:, 0]))
        #print(np.mean(samples[:, 1]))



        for r in range(len(result_percentile[1])):


            for key in finding.keys():
                if finding[key] == r:
                    parameters_name = key

                #if finding[key].keys() > 1:
                #    for keykey in finding[key].keys():
                #        if finding[key][keykey] == r:
                #            parameters_name = r


            result_b4[r] = result_percentile[1][r]





        print("tle_line1:", b4_tle_line1)
        print("tle_line2:", b4_tle_line2)
        print("overall", counter+1, i+1, "/", loops,"", b4,"runtime=", time.time() - starttime)
        print("")



        # filtering the POS to change the values within their limits.
        # this shall help also with the percentiles and averages
        keys = list(finding.keys())
        for k in range(len(pos)):
            for l in range(len(pos[k])):

                if keys[l] == "AoP":
                    pos[k][l] = zeroTo360(pos[k][l])

                if keys[l] == "raan":
                    pos[k][l] = zeroTo360(pos[k][l])

                if keys[l] == "inc":
                    pos[k][l] = zeroTo180(pos[k][l])

        counter += 1
        sampler.reset()


    return b4_result, counter



def start(station, timestamp_min, timestamps, mode, measurements, loops=30, walks=100):
    print("")
    print("## Determination1: Finding the orbit parameters")

    # distributing the initial positions within the scope of AoP, inclination and raan.
    pos = []
    for index_aop in range(7):
        for index_inclination in range(7):
            for index_raan in range(7):
                # the following ranges for the random initial parameter are gut feelings.
                # for later automation, this needs to be configurable by the user
                r_p0 = np.random.randint(63780 + 0, 63780 + 10000) / 10.0
                r_a0 = np.random.randint(r_p0 * 10.0, 63780 + 10000) / 10.0
                AoP0 = float(360.0 / 7 * index_aop)


                inc0 = np.random.randint(0, 18000) / 100.0
                raan0 = np.random.randint(0, 36000) / 100.0

                eccentricity = (r_a0 - r_p0) / (r_a0 + r_p0)
                h_angularmomentuum = np.sqrt(r_p0 * (1 + eccentricity * np.cos(0)) * mu)
                T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

                tp0 = np.random.randint(0, int(T_orbitperiod))
                bstar = np.random.randint(-11111 * 333, 11111 * 333)

                inputs = [r_p0, r_a0, AoP0, inc0, raan0, tp0]
                # inputs = [r_p0, r_a0, AoP0, inc0, raan0, tp0, bstar]

                pos.append(inputs)

    td = np.zeros(len(station))

    orbit = 1 # 0 = circle

    parameters = {
        "r_p": 0,
        "r_a": 0,
        "AoP": 0,
        "inc": 0,
        "raan": 0,
        "tp": 0,
        "bstar": 0.0,
        "td": td
    }

    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5,
        #"bstar": 6# ,
        # "td": {"0": 7,
        #       "2": 8}
    }

    # the overall number of walkers need to be multiple of 2.
    ndim = len(finding)
    nwalkers = len(pos)

    # initiating the optimization.
    # every loop will provide an output.
    # walks is the number of steps/hops a walker will be doing.

    counter = 0

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")
    # finding the orbits now...


    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, loops, walks, counter, station,
                                 timestamp_min, timestamps, mode, measurements, orbit)


    theta = []
    for r in range(len(result)):
        theta.append(result[r])

    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)
    sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements)

    print("rp=", r_p,
          "ra=", r_a,
          "AoP=", AoP,
          "inc=", inc,
          "raan=", raan,
          "tp=", tp,
          "bstar=", bstar,
          "td=", td)






    ############# next optimization, just the bstar now
    print("")
    print("## Determination2: Finding the bstar")
    pos = []
    for index_bstar in range(220):
        bstar = np.random.randint(-100000, 100000) / 100000.0

        inputs = [bstar]
        pos.append(inputs)

    td = np.zeros(len(station))

    orbit = 1  # 0 = circle

    parameters = {
        "r_p": r_p,
        "r_a": r_a,
        "AoP": AoP,
        "inc": inc,
        "raan": raan,
        "tp": tp,
        "bstar": 0.0,
        "td": td
    }

    finding = {
        "bstar": 0
    }

    # the overall number of walkers need to be multiple of 2.
    ndim = len(finding)
    nwalkers = len(pos)

    # initiating the optimization.
    # every loop will provide an output.
    # walks is the number of steps/hops a walker will be doing.

    #counter = 0

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait again now...")
    # finding the orbits now...

    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, loops, walks, counter, station,
                                 timestamp_min, timestamps, mode, measurements, orbit)

    theta = []
    for r in range(len(result)):
        theta.append(result[r])

    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)
    sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements)

    print("rp=", r_p,
          "ra=", r_a,
          "AoP=", AoP,
          "inc=", inc,
          "raan=", raan,
          "tp=", tp,
          "bstar=", bstar,
          "td=", td)





    '''
    global_grid_dif = []

    iter_AoP = 180
    iter_tp = 90

    for i in range(0, iter_AoP, 1):
        raan_dif = []
        for j in range(0, iter_tp, 1):
            AoP_test = float(i * 360.0 / iter_AoP)
            # inc_test = 0.0
            tp_test = float(j * 4200.0 / iter_tp)
            r_p = 6746
            r_a = 6752
            inc = 53.085
            raan = 90.929
            bstar = 10000000
            dif = get_state_sum(r_a, r_p, inc, raan, AoP_test, tp_test, bstar, td, station, timestamp_min, timestamps, mode,
                                measurements)
            raan_dif.append(dif)

        global_grid_dif.append(raan_dif)

    optimum_index = np.unravel_index(np.argmax(global_grid_dif), np.shape(global_grid_dif))

    print("global search says: AoP=", optimum_index[0] * 360 / iter_AoP, "raan=", optimum_index[1] * 4200 / iter_tp)
    print("best dif value=", global_grid_dif[optimum_index[0]][optimum_index[1]])
    print("")

    AoP_set = optimum_index[0] * 360 / iter_AoP
    tp_set = optimum_index[1] * 4200 / iter_tp
    print(AoP_set, tp_set)
    '''

    return r_p, r_a, AoP, inc, raan, tp, bstar, td


def fromposition(timestamp, sat, mode=0):

    station =[[]]
    el = [[]]
    az = [[]]
    ranging = [[]]
    doppler = [[]]
    t = []
    satellite_pos = []

    satellite_pos.append(sat)
    t.append(timestamp)

    measurements = {}
    measurements["el"] = el
    measurements["az"] = az
    measurements["rightascension"] = el
    measurements["declination"] = az
    measurements["satellite_pos"] = satellite_pos
    measurements["range"] = ranging
    measurements["doppler"] = doppler
    timestamps = t
    timestamp_min = np.min(np.min(timestamps))

    for s in range(len(timestamps)):
        for t0 in range(len(timestamps[s])):
            timestamps[s][t0] = timestamps[s][t0] - timestamp_min

    r_p, r_a, AoP, inc, raan, tp, bstar, td = start(station, timestamp_min, timestamps, mode, measurements, loops = 15, walks=60)

    return r_p, r_a, AoP, inc, raan, tp, bstar, td


if __name__== "__main__":
    fromposition()