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
    # but first min-max-rescaling or currently mean.

    # normalization
    rms_sum = 0
    emcee_factor = 10000  # somehow emcee seems to not like rms_sum smaller 1.0, so we artificially boost that up

    satellite_radius = []
    for s in range(len(satellite_pos)):
        for pos in range(len(satellite_pos[s])):
            satellite_radius.append((satellite_pos[s][pos][0]**2 +
                                     satellite_pos[s][pos][1]**2 +
                                     satellite_pos[s][pos][2]**2)**0.5)

        mean_radius = np.mean(satellite_radius)

        # unfortunately inputs can be ragged arrays, and numpy does not like it.
        # so we iterate through it and use numpy sub-array wise.

        # normalizing with mean radius
        track1 = np.divide(track[s], mean_radius)
        satellite_pos1 = np.divide(satellite_pos[s], mean_radius)

        rms = np.subtract(track1, satellite_pos1)
        rms = np.multiply(rms, emcee_factor)
        rms_sum += np.sum(np.square(rms))

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
        print("overall", counter+1, i+1, "/", loops,"rms=", b4, "runtime=", time.time() - starttime)
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


def optimize_with_mcmc(parameters, finding, loops, walks, nwalkers, counter, station, timestamp_min, timestamps, mode,
                       measurements, orbit=0,
                       r_a_lim= [0.0, 10.0],
                       r_p_lim= [0.0, 10.0],
                       AoP_lim= [0.0, 360.0],
                       inc_lim= [0.0, 180.0],
                       raan_lim= [0.0, 360.0],
                       tp_lim= [0.0, 1.0],
                       bstar_lim= [-100000.0, 100000.0]):

    pos = []
    for _ in range(nwalkers):

        inputs = []

        # the following ranges for the random initial parameter are gut feelings.
        # for later automation, this needs to be configurable by the user

        r_p = parameters["r_p"]
        r_a = parameters["r_a"]
        # todo check if a smaller/bigger check is needed also here.

        if "r_p" in finding:
            r_p = np.random.randint(int(r_p_lim[0] * 10.0), int(r_p_lim[1] * 10.0)) / 10.0 + r_p
            inputs.append(r_p)

        if "r_a" in finding:
            r_a = np.random.randint(int(r_a_lim[0] * 10.0), int(r_a_lim[1] * 10.0)) / 10.0 + r_a
            if r_a < r_p:
                r_a = r_p
            inputs.append(r_a)
            # todo what if only r_a or r_p is set?

        if "AoP" in finding:
            AoP = np.random.randint(int(AoP_lim[0] * 100.0), int(AoP_lim[1] * 100.0)) / 100.0 + parameters["AoP"]
            inputs.append(AoP)

        if "inc" in finding:
            inc = np.random.randint(int(inc_lim[0] * 100.0), int(inc_lim[1] * 100.0)) / 100.0 + parameters["inc"]
            inputs.append(inc)

        if "raan" in finding:
            raan = np.random.randint(int(raan_lim[0] * 100.0), int(raan_lim[1] * 100.0)) / 100.0 + parameters["raan"]
            inputs.append(raan)

        if "tp" in finding:

            eccentricity = (r_a - r_p) / (r_a + r_p)
            h_angularmomentuum = np.sqrt(r_p * (1 + eccentricity * np.cos(0)) * mu)
            T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

            tp = np.random.randint(int(tp_lim[0] * 100.0), int(tp_lim[1] * 100.0)) / 100.0
            if parameters["tp"] < 0.0:
                tp = tp * T_orbitperiod
            else:
                tp = tp + parameters["tp"]
            # todo: if r_a0 and r_p0 are not set, what to do then?
            inputs.append(tp)

        if "bstar" in finding:
            bstar = np.random.randint(int(bstar_lim[0]), int(bstar_lim[1])) / 100000.0 + parameters["bstar"]
            inputs.append(bstar)

        if "td" in finding:
            for f in range(len(finding["td"])):
                td = np.random.randint(-10000, 10000) / 1000.0
                inputs.append(td)

        pos.append(inputs)

    # the overall number of walkers need to be multiple of 2.
    ndim = len(finding)

    # initiating the optimization.
    # every loop will provide an output.
    # walks is the number of steps/hops a walker will be doing.

    print("performing:", loops, "loops,", walks, "walks, for", nwalkers, "walkers")
    print("please wait now...")
    # finding the orbits now...

    result, counter = find_orbit(nwalkers, ndim, pos, parameters, finding, loops, walks, counter, station,
                                 timestamp_min, timestamps, mode, measurements, orbit)

    theta = []
    for r in range(len(result)):
        theta.append(result[r])

    r_p0, r_a0, AoP0, inc0, raan0, tp0, bstar0, td0 = get_kepler_parameters(theta, parameters, finding, orbit)
    sum = get_state_sum(r_a0, r_p0, inc0, raan0, AoP0, tp0, bstar0, td0, station, timestamp_min, timestamps, mode,
                        measurements)

    print("rp=", r_p0,
          "ra=", r_a0,
          "AoP=", AoP0,
          "inc=", inc0,
          "raan=", raan0,
          "tp=", tp0,
          "bstar=", bstar0,
          "td=", td0)

    parameters = {
        "r_p": r_p0,
        "r_a": r_a0,
        "AoP": AoP0,
        "inc": inc0,
        "raan": raan0,
        "tp": tp0,
        "bstar": bstar0,
        "td": td0
    }

    return parameters



def start(station, timestamp_min, timestamps, mode, measurements, loops=30, walks=100):
    print("")
    print("## Determination1: Finding the orbit parameters")

    # distributing the initial positions within the scope of AoP, inclination and raan.
    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5,
        # "bstar": 6# ,
        # "td": {"0": 7,
        #       "2": 8}
    }

    r_a0 = 6378.0
    r_p0 = 6378.0
    AoP0 = 0.0
    inc0 = 0.0
    raan0 = 0.0
    tp0 = -1.0
    bstar0 = 0.0
    td0 = np.zeros(len(station))

    r_a_min = 0.0
    r_a_max = 1000.0
    r_p_min = 0.0
    r_p_max = 1000.0
    AoP_min = 0.0
    AoP_max = 360.0
    inc_min = 0.0
    inc_max = 180.0
    raan_min = 0.0
    raan_max = 360.0
    tp_min = 0.0
    tp_max = 1.0
    bstar_min = -100000.0
    bstar_max = 100000.0

    orbit = 1  # 0 = circle

    counter = 0

    parameters = {
        "r_p": r_p0,
        "r_a": r_a0,
        "AoP": AoP0,
        "inc": inc0,
        "raan": raan0,
        "tp": tp0,
        "bstar": bstar0,
        "td": td0
    }

    nwalkers = 400

    parameters = optimize_with_mcmc(parameters, finding, loops, walks, nwalkers,
                                    counter, station, timestamp_min, timestamps,
                                    mode, measurements, orbit= orbit,
                                    r_a_lim= [r_a_min, r_a_max],
                                    r_p_lim= [r_p_min, r_p_max],
                                    AoP_lim= [AoP_min, AoP_max],
                                    inc_lim= [inc_min, inc_max],
                                    raan_lim= [raan_min, raan_max],
                                    tp_lim= [tp_min, tp_max],
                                    bstar_lim= [bstar_min, bstar_max])

    counter += loops


    ############# next optimization, just the bstar now
    print("")
    print("## Determination2: Finding the bstar")

    finding = {
        "r_p": 0,
        "r_a": 1,
        "AoP": 2,
        "inc": 3,
        "raan": 4,
        "tp": 5,
        "bstar": 6
    }

    r_a_min = -1.0
    r_a_max = 1.0
    r_p_min = -1.0
    r_p_max = 1.0
    AoP_min = -1.0
    AoP_max = 1.0
    inc_min = -1.0
    inc_max = 1.0
    raan_min = -1.0
    raan_max = 1.0
    tp_min = -10.0
    tp_max = 10.0
    bstar_min = -100000.0
    bstar_max = 100000.0

    orbit = 1  # 0 = circle

    nwalkers = 200

    parameters = optimize_with_mcmc(parameters, finding, loops, walks, nwalkers,
                                    counter, station, timestamp_min, timestamps,
                                    mode, measurements, orbit=orbit,
                                    r_a_lim=[r_a_min, r_a_max],
                                    r_p_lim=[r_p_min, r_p_max],
                                    AoP_lim=[AoP_min, AoP_max],
                                    inc_lim=[inc_min, inc_max],
                                    raan_lim=[raan_min, raan_max],
                                    tp_lim=[tp_min, tp_max],
                                    bstar_lim=[bstar_min, bstar_max])

    counter += loops

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

    return parameters


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

    parameters = start(station, timestamp_min, timestamps, mode, measurements, loops= 15, walks=50)

    return parameters


if __name__== "__main__":
    fromposition()