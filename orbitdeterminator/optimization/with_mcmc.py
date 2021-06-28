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

import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import kep_determination.positional_observation_reporting as por
import util.read_data as rd

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
    x = float(bstar) / 10 ** exp_max

    if x == 0.0:
        # out = "+33299-6"
        out = "        "

    else:
        exp = np.floor(np.log10(np.abs(x)))
        c = 10 ** (exp + 1)

        if np.sign(x) == 1.0:
            s = "+"
        else:
            s = "-"

        number = x / c
        number = str(number)
        number = number.split(".")[1]
        for i in range(5 - len(number)):
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
    x = float(bstar) / 10 ** exp_max

    if x == 0.0:
        # out = "+33299-6"
        out = "        "

    else:
        exp = np.floor(np.log10(np.abs(x)))
        c = 10 ** (exp + 1)

        if np.sign(x) == 1.0:
            s = "+"
        else:
            s = "-"

        number = x / c
        number = str(number)
        number = number.split(".")[1]
        for i in range(5 - len(number)):
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
    for i in range(len(line) - 1):

        if line[i].isnumeric():
            sum += int(line[i])

        if line[i] == "-":
            sum += 1

        line_out += line[i]

    line_out += str(np.mod(sum, 10))

    return line_out


def tlestuf(epoch, inc, ecc, aop, raan, me, meanmove, bstar):
    observing_time = Time(str(epoch), format="unix", scale="utc")
    startofday = Time(str(observing_time.to_datetime().timetuple().tm_year) + "-" + str(
        observing_time.to_datetime().timetuple().tm_mon) + "-" + str(observing_time.to_datetime().timetuple().tm_mday),
                      scale="utc").unix
    fractionofday = filler(str((epoch - startofday) / (24.0 * 3600.0)), 0, " ", 8)
    year = str(observing_time.to_datetime().timetuple().tm_year)[2:5]
    dayofyear = str(observing_time.to_datetime().timetuple().tm_yday)
    dayofyear = filler(str(dayofyear) + ".", 3, "0", 0)

    sgp4_epoch = year + dayofyear + fractionofday[2:10]
    sgp4_inc = filler(str(inc), 3, " ", 4)[:8]
    sgp4_ecc = filler(ecc, 1, " ", 7)[2:9]
    sgp4_aop = filler(str(aop), 3, " ", 4)[:8]
    sgp4_raan = filler(str(raan), 3, " ", 4)[:8]
    sgp4_me = filler(str(me), 3, " ", 4)[:8]
    sgp4_meanmove = filler(str(meanmove), 2, " ", 8)[:11]
    sgp4_revolutions = "00000"  # any
    sgp4_checksum = "0"  # removed, any
    sgp4_bstar = tle_bstar(bstar)

    s = '1 12345U 00000A   ' + sgp4_epoch + ' -.00000180  00000-0 ' + sgp4_bstar + ' 0  0000'
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

    satrec.classification = 'U'
    satrec.intldesg = "OrbDet"

    return satrec


def zeroTo360(x):
    if x >= 360.0:
        x = x - int(x / 360) * 360.0
    elif x < 0.0:
        x = x - (int(x / 360) - 1) * 360.0

    return x


def zeroTo180(x):
    if x >= 180.0:
        x = x - int(x / 180) * 180.0
    elif x < 0.0:
        x = x - (int(x / 180) - 1) * 180.0

    return x


def get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements,
                  meta):
    elevation_min = -30

    # putting in the measurements
    ra = measurements["ra"]
    dec = measurements["dec"]
    el = measurements["el"]
    az = measurements["az"]
    ranging = measurements["range"]
    doppler = measurements["doppler"]
    satellite_pos = measurements["satellite_pos"]

    # preparing the orbit track that is being simulated and used for comparing to the measurements
    # track = np.zeros_like(satellite_pos) # did not work with strange arrays.
    track = []
    for tr in range(len(satellite_pos)):
        track.append(np.zeros_like(satellite_pos[tr]))

    track_az = []
    track_el = []
    for tr in range(len(az)):
        track_az.append(np.zeros_like(az[tr]))
        track_el.append(np.zeros_like(el[tr]))

    track_range = []
    for tr in range(len(ranging)):
        track_range.append(np.zeros_like(ranging[tr]))

    track_doppler = []
    for tr in range(len(doppler)):
        track_doppler.append(np.zeros_like(doppler[tr]))

    track_ra = []
    track_dec = []
    for tr in range(len(ra)):
        track_ra.append(np.zeros_like(ra[tr]))
        track_dec.append(np.zeros_like(dec[tr]))

    # preparing the orbit parameter needed for the simulated orbit
    eccentricity = (r_a - r_p) / (r_a + r_p)
    h_angularmomentuum = np.sqrt(r_p * (1.0 + eccentricity * np.cos(0)) * mu)
    T_orbitperiod = 2.0 * np.pi / mu ** 2.0 * (h_angularmomentuum / np.sqrt(1.0 - eccentricity ** 2)) ** 3

    me = tp * (2.0 * np.pi) / T_orbitperiod * 180.0 / np.pi
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

        for t0 in range(len(satellite_pos[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            R1 = satellite.at(t).position.km
            # V1 = satellite.at(t).velocity.km_per_s

            # R = np.array(R1)
            # V = np.array(V1)

            track[s][t0][0] = R1[0]
            track[s][t0][1] = R1[1]
            track[s][t0][2] = R1[2]

            #############

            # if mode == 0:
            #    #state_sum += (R[0] - satellite_pos[s][t0][0]) ** 2 + \
            #    #             (R[1] - satellite_pos[s][t0][1]) ** 2 + \
            #    #             (R[2] - satellite_pos[s][t0][2]) ** 2

        ### station
        if "long" in station[s]:
            gs_long = station[s]["long"]
            gs_lat = station[s]["lat"]
            gs_alt = station[s]["alt"]

            observer = wgs84.latlon(gs_lat, gs_long, gs_alt)
            difference = satellite - observer

        for t0 in range(len(az[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]
            if np.abs(td[s]) > 2.0:
                # the system time is expected to be jittery only by a few seconds.
                # so this should be enough and stops the time seach to run amok
                return -np.inf

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            topocentric = difference.at(t)
            alt1, az1, distance1 = topocentric.altaz()

            if alt1.degrees >= elevation_min:
                track_az[s][t0] = az1.degrees
                track_el[s][t0] = alt1.degrees
            else:
                # track_az[s][t0] = np.inf
                # track_el[s][t0] = np.inf

                # if one value is not good, then it is already infinity and we can also quit now
                return -np.inf

        for t0 in range(len(ranging[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            topocentric = difference.at(t)
            alt1, az1, distance1 = topocentric.altaz()

            if alt1.degrees >= elevation_min:
                track_range[s][t0] = distance1.km
            else:
                # track_range[s][t0] = np.inf

                # if one value is not good, then it is already infinity and we can also quit now
                return -np.inf

        for t0 in range(len(doppler[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            topocentric = difference.at(t)
            alt1, az1, distance1 = topocentric.altaz()
            pointing = topocentric.position.km
            velo = topocentric.velocity.km_per_s
            # angle = np.dot(pointing, velo)
            angle = (pointing[0] * velo[0] + pointing[1] * velo[1] + pointing[2] * velo[2]) / (
                    np.linalg.norm(pointing) * np.linalg.norm(velo))
            angle = np.arccos(angle)
            range_rate = np.cos(angle) * np.linalg.norm(velo)
            f_0 = meta[s][0]["rf"]["fc"]
            c_speedoflight = 299792.458
            doppler_c = -range_rate * f_0 / c_speedoflight

            if alt1.degrees >= elevation_min:
                track_doppler[s][t0] = doppler_c
            else:
                # track_doppler[s][t0] = np.inf

                # if one value is not good, then it is already infinity and we can also quit now
                return -np.inf

        for t0 in range(len(ra[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            topocentric = difference.at(t)
            ra1, dec1, distance1 = topocentric.radec()
            alt1, az1, distance2 = topocentric.altaz()

            if alt1.degrees >= elevation_min:
                track_ra[s][t0] = ra1.radians * 180.0 / np.pi
                track_dec[s][t0] = dec1.degrees
            else:
                # track_ra[s][t0] = np.inf
                # track_dec[s][t0] = np.inf

                # if one value is not good, then it is already infinity and we can also quit now
                return -np.inf

    # now we just do a simple Root-mean-square of the measurement positions
    # and the orbit positions based on the simulation
    # but first min-max-rescaling or currently mean.

    # normalization
    rms_sum = 0
    emcee_factor = 10000  # somehow emcee seems to not like rms_sum smaller 1.0, so we artificially boost that up

    satellite_radius = []
    for s in range(len(satellite_pos)):
        if len(satellite_pos[s]) > 0:
            for pos in range(len(satellite_pos[s])):
                satellite_radius.append((satellite_pos[s][pos][0] ** 2 +
                                         satellite_pos[s][pos][1] ** 2 +
                                         satellite_pos[s][pos][2] ** 2) ** 0.5)

            mean_radius = np.mean(satellite_radius)

            # unfortunately inputs can be ragged arrays, and numpy does not like it.
            # so we iterate through it and use numpy sub-array wise.

            # normalizing with mean radius
            track1 = np.divide(track[s], mean_radius)
            satellite_pos1 = np.divide(satellite_pos[s], mean_radius)

            rms = np.subtract(track1, satellite_pos1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

    for s in range(len(az)):
        if len(az[s]) > 0:
            mean_az = np.mean(np.abs(az[s]))
            mean_el = np.mean(np.abs(el[s]))

            # normalizing with mean az
            track_az1 = np.divide(track_az[s], mean_az)
            az1 = np.divide(az[s], mean_az)

            rms = np.subtract(track_az1, az1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

            # normalizing with mean el
            track_el1 = np.divide(track_el[s], mean_el)
            el1 = np.divide(el[s], mean_el)

            rms = np.subtract(track_el1, el1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

    for s in range(len(ranging)):
        if len(ranging[s]) > 0:
            mean_range = np.mean(ranging[s])

            # normalizing with mean az
            track_range1 = np.divide(track_range[s], mean_range)
            range1 = np.divide(ranging[s], mean_range)

            rms = np.subtract(track_range1, range1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

    for s in range(len(doppler)):
        if len(doppler[s]) > 0:
            mean_doppler = np.mean(np.abs(doppler[s]))

            # normalizing with mean az
            track_doppler1 = np.divide(track_doppler[s], mean_doppler)
            doppler1 = np.divide(doppler[s], mean_doppler)

            rms = np.subtract(track_doppler1, doppler1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

    for s in range(len(ra)):
        if len(ra[s]) > 0:
            mean_ra = np.mean(np.abs(ra[s]))
            mean_dec = np.mean(np.abs(dec[s]))

            # normalizing with mean az
            track_ra1 = np.divide(track_ra[s], mean_ra)
            ra1 = np.divide(ra[s], mean_ra)

            rms = np.subtract(track_ra1, ra1)
            rms = np.multiply(rms, emcee_factor)
            rms_sum += np.sum(np.square(rms))

            # normalizing with mean el
            track_dec1 = np.divide(track_dec[s], mean_dec)
            dec1 = np.divide(dec[s], mean_dec)

            rms = np.subtract(track_dec1, dec1)
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


def log_likelihood(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit, meta):
    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)

    sum = get_state_sum(r_a, r_p, inc, raan, AoP, tp, bstar, td,
                        station, timestamp_min, timestamps, mode, measurements, meta)
    return sum


def log_prior(theta, parameters, finding, orbit):
    r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(theta, parameters, finding, orbit)

    r_earth = 6378.0

    if r_earth < r_p and r_p <= r_a and \
            inc >= 0.0 and inc <= 180.0 and \
            raan >= -90.0 and \
            AoP > -90.0 and \
            np.abs(bstar) <= 1.0:
        return 0.0

    return -np.inf


def log_probability(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit, meta):
    lp = log_prior(theta, parameters, finding, orbit)

    if not np.isfinite(lp):
        return -np.inf

    sum = log_likelihood(theta, parameters, finding, station, timestamp_min, timestamps, mode, measurements, orbit,
                         meta)

    if np.isnan(sum) == True:
        return -np.inf

    return lp + sum


def compare(line1_1, line1_2, line2_1, line2_2, timestamp_min, timestamps, td):
    ts = load.timescale()

    satrec1 = Satrec.twoline2rv(line1_1, line1_2)
    satrec2 = Satrec.twoline2rv(line2_1, line2_2)
    satellite1 = EarthSatellite.from_satrec(satrec1, ts)
    satellite2 = EarthSatellite.from_satrec(satrec2, ts)

    residual = 0
    number_of_measurements = 0
    for s in range(len(timestamps)):
        number_of_measurements += len(timestamps[s])
        for t0 in range(len(timestamps[s])):
            time_step = timestamps[s][t0]

            timestamp1 = timestamp_min + time_step + td[s]

            observing_time = Time(timestamp1, format="unix", scale="utc")
            t = ts.from_astropy(observing_time)

            R1 = satellite1.at(t).position.km
            R2 = satellite2.at(t).position.km

            distance = ((R1[0] - R2[0]) ** 2 + (R1[1] - R2[1]) ** 2 + (R1[2] - R2[2]) ** 2) ** 0.5
            residual += distance

    return residual / number_of_measurements


def find_orbit(nwalkers, ndim, pos, parameters, finding, loops, walks, counter, station, timestamp_min, timestamps,
               mode, measurements, orbit, meta=[[]], generated={}):
    # preparing the optimizer
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(parameters, finding, station, timestamp_min, timestamps, mode,
                                          measurements, orbit, meta))

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
    print("maybe you will see this message:")
    print("RuntimeWarning: invalid value encountered in double_scalars lnpdiff = f + nlp - state.log_prob[j]")
    print("just ignore it. it will run anyways")
    for i in range(loops):
        # start the optimization with emcee
        pos, prob, state = sampler.run_mcmc(pos, walks, progress=True)

        # extracting the orbit parameters back from the resulting POS.
        # in this case, we just use the best result, which is argmax of POS
        r_p, r_a, AoP, inc, raan, tp, bstar, td = get_kepler_parameters(pos[np.argmax(prob)], parameters, finding,
                                                                        orbit)

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

        # print("prob", np.max(prob), np.argmax(prob), b4)
        # print(pos[np.argmax(prob)])

        # save_progress(plotnames, counter, r_p, r_a, AoP, inc, raan, tp, bstar, td, np.max(prob), s1, t1, mode, filename_1)
        # save_progress_pos(plotnames, pos, prob, state)

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

        # get_state_plot(r_a, r_p, inc, raan, AoP, tp, bstar, td, station, timestamp_min, timestamps, mode, measurements)

        samples = sampler.chain[:, 0:, :].reshape((-1, ndim))

        result_percentile = np.percentile(samples, [16, 50, 84], axis=0)

        # import corner
        # import matplotlib.pyplot as plt
        # flat_samples = samples
        # fig = corner.corner(flat_samples)
        # plt.show()

        # print(samples.shape)
        # print(len(samples))
        # print(samples[:, 0])
        # print(samples[:, 1])
        # print(samples[:, 2])
        # print(samples[:, 3])
        # print(samples[:, 4])
        # print(samples[:, 5])

        # plt.plot(samples[:, 0])
        # plt.plot(samples[:, 1])
        # plt.show()
        # print(np.mean(samples[:, 0]))
        # print(np.mean(samples[:, 1]))

        for r in range(len(result_percentile[1])):

            for key in finding.keys():
                if finding[key] == r:
                    parameters_name = key

                # if finding[key].keys() > 1:
                #    for keykey in finding[key].keys():
                #        if finding[key][keykey] == r:
                #            parameters_name = r

            result_b4[r] = result_percentile[1][r]

        print("tle_line1:", b4_tle_line1)
        print("tle_line2:", b4_tle_line2)
        if "tle" in generated:
            print("compare", compare(b4_tle_line1, b4_tle_line2, generated["tle"]["line1"], generated["tle"]["line2"],
                                     timestamp_min, timestamps, td))
        print("rp=", r_p,
              "ra=", r_a,
              "AoP=", AoP,
              "inc=", inc,
              "raan=", raan,
              "tp=", tp,
              "bstar=", bstar,
              "td=", td)
        print("overall", counter + 1, i + 1, "/", loops, "rms=", b4, "runtime=", time.time() - starttime)
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
                       measurements, orbit=0, meta=[[]], generated={},
                       r_a_lim=[0.0, 10.0],
                       r_p_lim=[0.0, 10.0],
                       AoP_lim=[0.0, 360.0],
                       inc_lim=[0.0, 180.0],
                       raan_lim=[0.0, 360.0],
                       tp_lim=[0.0, 1.0],
                       bstar_lim=[-100000.0, 100000.0],
                       td_lim=[-0.5, 0.5]):
    pos = []
    for _ in range(nwalkers):

        inputs = []

        # the following ranges for the random initial parameter are gut feelings.
        # for later automation, this needs to be configurable by the user

        r_p = parameters["r_p"]
        r_a = parameters["r_a"]
        # todo check if a smaller/bigger check is needed also here.

        if "r_p" in finding:
            random_steps = 10000.0
            random_factor = random_steps / np.abs(r_p_lim[1] - r_p_lim[0])
            r_p = np.random.randint(int(r_p_lim[0] * random_factor),
                                    int(r_p_lim[1] * random_factor)) / random_factor + r_p
            inputs.append(r_p)

        if "r_a" in finding:
            random_steps = 10000.0
            random_factor = random_steps / np.abs(r_a_lim[1] - r_a_lim[0])
            r_a = np.random.randint(int(r_a_lim[0] * random_factor),
                                    int(r_a_lim[1] * random_factor)) / random_factor + r_a
            if r_a < r_p:
                r_a = r_p
            inputs.append(r_a)
            # todo what if only r_a or r_p is set?

        if "AoP" in finding:
            random_steps = 10000.0
            random_factor = random_steps / np.abs(AoP_lim[1] - AoP_lim[0])
            AoP = np.random.randint(int(AoP_lim[0] * random_factor), int(AoP_lim[1] * random_factor)) / random_factor + \
                  parameters["AoP"]
            inputs.append(AoP)

        if "inc" in finding:
            random_steps = 10000.0
            random_factor = random_steps / np.abs(inc_lim[1] - inc_lim[0])
            inc = np.random.randint(int(inc_lim[0] * random_factor), int(inc_lim[1] * random_factor)) / random_factor + \
                  parameters["inc"]
            inputs.append(inc)

        if "raan" in finding:
            random_steps = 10000.0
            random_factor = random_steps / np.abs(raan_lim[1] - raan_lim[0])
            raan = np.random.randint(int(raan_lim[0] * random_factor),
                                     int(raan_lim[1] * random_factor)) / random_factor + parameters["raan"]
            inputs.append(raan)

        if "tp" in finding:

            eccentricity = (r_a - r_p) / (r_a + r_p)
            h_angularmomentuum = np.sqrt(r_p * (1 + eccentricity * np.cos(0)) * mu)
            T_orbitperiod = 2.0 * np.pi / mu ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

            random_steps = 10000.0
            random_factor = random_steps / np.abs(tp_lim[1] - tp_lim[0])

            tp = np.random.randint(int(tp_lim[0] * random_factor), int(tp_lim[1] * random_factor)) / random_factor
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

            random_steps = 10000.0
            random_factor = random_steps / np.abs(td_lim[1] - td_lim[0])

            for f in range(len(finding["td"])):
                td = np.random.randint(int(td_lim[0] * random_factor), int(td_lim[1] * random_factor)) / random_factor
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
                                 timestamp_min, timestamps, mode, measurements, orbit, meta=meta, generated=generated)

    theta = []
    for r in range(len(result)):
        theta.append(result[r])

    r_p0, r_a0, AoP0, inc0, raan0, tp0, bstar0, td0 = get_kepler_parameters(theta, parameters, finding, orbit)
    sum = get_state_sum(r_a0, r_p0, inc0, raan0, AoP0, tp0, bstar0, td0, station, timestamp_min, timestamps, mode,
                        measurements, meta)

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


def start(opt, station, timestamp_min, timestamps, mode, measurements, meta=[[]], generated={}, loops=30, walks=100):

    # initial parameters
    r_a0 = 6378.0
    r_p0 = 6378.0
    AoP0 = 0.0
    inc0 = 0.0
    raan0 = 0.0
    tp0 = -1.0
    bstar0 = 0.0
    td0 = np.zeros(len(station))


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

    counter = 0

    for loop in range(len(opt.nwalkers)):
        print("")
        print("## Determination", loop, ": Finding the orbit parameters")

        # distributing the initial positions within the scope of AoP, inclination and raan.
        finding = opt.finding[loop]

        #parameters["tp"] = -1

        nwalkers = opt.nwalkers[loop]

        parameters = optimize_with_mcmc(parameters, finding, opt.loops[loop], opt.walks[loop], nwalkers,
                                        counter, station, timestamp_min, timestamps,
                                        mode, measurements, orbit=opt.orbit[loop], meta=meta, generated=generated,
                                        r_a_lim=opt.r_a_lim[loop],
                                        r_p_lim=opt.r_p_lim[loop],
                                        AoP_lim=opt.AoP_lim[loop],
                                        inc_lim=opt.inc_lim[loop],
                                        raan_lim=opt.raan_lim[loop],
                                        tp_lim=opt.tp_lim[loop],
                                        bstar_lim=opt.bstar_lim[loop],
                                        td_lim=opt.td_lim[loop])

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
    station = [[]]
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
    measurements["ra"] = el
    measurements["dec"] = az
    measurements["satellite_pos"] = satellite_pos
    measurements["range"] = ranging
    measurements["doppler"] = doppler
    timestamps = t

    timestamp_min = 0.0
    tested = 0
    for ii in range(len(timestamps)):
        if len(timestamps[ii]):
            if tested == 0:
                timestamp_min = np.min(timestamps[ii])
                tested = 1
            else:
                if timestamp_min > np.min(timestamps[ii]):
                    timestamp_min = np.min(timestamps[ii])

    for s in range(len(timestamps)):
        for t0 in range(len(timestamps[s])):
            timestamps[s][t0] = timestamps[s][t0] - timestamp_min

    parameters = start(station, timestamp_min, timestamps, mode, measurements, loops=15, walks=50)

    return parameters


def extract_key_and_time_from_data(i, data, keys):
    keys_data = []
    times_data = []

    for key in keys:

        key_data = []
        time_data = []

        if "solve" in data["signal"][i]:
            if key in data["signal"][i]["solve"]:
                if data["signal"][i]["solve"][key] == 1:
                    print("solve activated for ", key)

                    for j in range(len(data["signal"][i]["data"])):
                        line = data["signal"][i]["data"][j]

                        if key in line:
                            key_data.append(line[key])
                            time_data.append(line["systemtime"])

        keys_data.append(key_data)
        times_data.append(time_data)

    return keys_data, times_data


def from_iod(filenames=["../example_data/SATOBS-ML-19200716.txt"]):
    print("loading IOD files")

    Rs = []
    station = []
    els = []
    azs = []
    rangings = []
    dopplers = []
    ras = []
    decs = []
    t = []

    for file in filenames:
        # load IOD data for a given satellite

        iod_object_data = por.load_iod_data(file)

        timestamp = []
        ra = []
        dec = []
        az = []
        el = []

        lat = 0.0
        lon = 0.0
        alt = 0.0

        for i in range(len(iod_object_data["object"])):
            yr = iod_object_data["yr"][i]
            month = iod_object_data["month"][i]
            day = iod_object_data["day"][i]
            hour = iod_object_data["hr"][i]
            min = iod_object_data["min"][i]
            sec = iod_object_data["sec"][i]
            msec = iod_object_data["msec"][i]

            time_iod = datetime(yr, month, day, hour, min, sec, msec * 1000)
            time_iod = observing_time = Time(time_iod, scale="utc").unix

            if iod_object_data["right_ascension"][i] != -1 or iod_object_data["declination"][i] != -1:
                timestamp.append(time_iod)
                ra.append(iod_object_data["right_ascension"][i])
                dec.append(iod_object_data["declination"][i])

                site_codes_0 = iod_object_data["station"][i]

                sat_observatories_data = por.load_sat_observatories_data(
                    '../station_observatory_data/sat_tracking_observatories.txt')
                gs = por.get_station_data(site_codes_0, sat_observatories_data)
                lat = gs['Latitude']  # deg
                lon = gs['Longitude']  # deg
                alt = gs['Elev']  # meters

        station.append({"lat": lat, "long": lon, "alt": alt})

        t.append(timestamp)
        ras.append(ra)
        decs.append(dec)
        els.append(el)  # could be in iod format. not checked for now
        azs.append(az)  # could be in iod format. not checked for now
        rangings.append([])
        dopplers.append([])
        Rs.append([])

        timestamp = []
        ra = []
        dec = []
        az = []
        el = []

        lat = 0.0
        lon = 0.0
        alt = 0.0

        for i in range(len(iod_object_data["object"])):
            yr = iod_object_data["yr"][i]
            month = iod_object_data["month"][i]
            day = iod_object_data["day"][i]
            hour = iod_object_data["hr"][i]
            min = iod_object_data["min"][i]
            sec = iod_object_data["sec"][i]
            msec = iod_object_data["msec"][i]

            time_iod = datetime(yr, month, day, hour, min, sec, msec * 1000)
            time_iod = observing_time = Time(time_iod, scale="utc").unix

            if iod_object_data["azimuth"][i] != -1 or iod_object_data["elevation"][i] != -1:
                timestamp.append(time_iod)
                az.append(iod_object_data["azimuth"][i])
                el.append(iod_object_data["elevation"][i])

                site_codes_0 = iod_object_data["station"][i]

                sat_observatories_data = por.load_sat_observatories_data(
                    '../station_observatory_data/sat_tracking_observatories.txt')
                gs = por.get_station_data(site_codes_0, sat_observatories_data)
                lat = gs['Latitude']  # deg
                lon = gs['Longitude']  # deg
                alt = gs['Elev']  # meters

        station.append({"lat": lat, "long": lon, "alt": alt})

        t.append(timestamp)
        ras.append(ra)
        decs.append(dec)
        els.append(el)  # could be in iod format. not checked for now
        azs.append(az)  # could be in iod format. not checked for now
        rangings.append([])
        dopplers.append([])
        Rs.append([])

    measurements = {}
    measurements["el"] = els
    measurements["az"] = azs
    measurements["ra"] = ras
    measurements["dec"] = decs
    measurements["satellite_pos"] = Rs
    measurements["range"] = rangings
    measurements["doppler"] = dopplers

    timestamps = t

    timestamp_min = 0.0
    tested = 0
    for ii in range(len(timestamps)):
        if len(timestamps[ii]):
            if tested == 0:
                timestamp_min = np.min(timestamps[ii])
                tested = 1
            else:
                if timestamp_min > np.min(timestamps[ii]):
                    timestamp_min = np.min(timestamps[ii])

    for s in range(len(timestamps)):
        for t0 in range(len(timestamps[s])):
            timestamps[s][t0] = timestamps[s][t0] - timestamp_min

    return station, timestamp_min, timestamps, measurements


def from_json(opt, filenames=["../example_data/stuttgart.json"]):
    print("loading JSON files")

    Rs = []
    timestamps = []

    station = []
    els = []
    azs = []
    rangings = []
    dopplers = []
    ras = []
    decs = []
    t = []
    generated = {}
    meta = []

    for file in filenames:

        print("detecting file", rd.detect_file_format(file))

        with open(file, 'r') as infile:
            data = json.load(infile)

        for i in range(len(data["signal"])):

            timestamp_t = []
            timestamp_azel = []
            timestamp_ranging = []
            timestamp_doppler = []
            timestamp_radec = []

            if "generated" in data["signal"][i]["meta"]:
                generated = data["signal"][i]["meta"]["generated"]

            R = []
            az = []
            el = []
            ranging = []
            doppler = []
            ra = []
            dec = []

            ## position

            keys = ["position"]

            R_unit = "km"
            if "meta" in data["signal"][i]:
                meta.append([data["signal"][i]["meta"]])

                if "unit" in data["signal"][i]["meta"]:
                    if keys[0] in data["signal"][i]["meta"]["unit"]:
                        R_unit = data["signal"][i]["meta"]["unit"][keys[0]]

            else:
                meta.append([])

            input, input_time = extract_key_and_time_from_data(i, data, keys)

            R = input[0]
            if R_unit == "m" or R_unit == "meters":
                R = np.divide(R, 1000.0)

            timestamp_t = input_time[0]

            station.append([])
            Rs.append(R)
            azs.append(az)
            els.append(el)
            rangings.append(ranging)
            dopplers.append(doppler)
            ras.append(ra)
            decs.append(dec)

            R = []
            az = []
            el = []
            ranging = []
            doppler = []
            ra = []
            dec = []

            ## AzEl
            keys = ["az", "el"]

            az_unit = "deg"
            el_unit = "deg"

            if "meta" in data["signal"][i]:
                meta.append([data["signal"][i]["meta"]])

                if "unit" in data["signal"][i]["meta"]:
                    if keys[0] in data["signal"][i]["meta"]["unit"]:
                        az_unit = data["signal"][i]["meta"]["unit"][keys[0]]

                    if keys[1] in data["signal"][i]["meta"]["unit"]:
                        el_unit = data["signal"][i]["meta"]["unit"][keys[1]]

            else:
                meta.append([])

            input, input_time = extract_key_and_time_from_data(i, data, keys)

            az = input[0]
            if az_unit == "deg" or az_unit == "degrees":
                az = np.divide(az, 1.0)  # todo, adds other conversion

            el = input[1]
            if el_unit == "deg" or el_unit == "degrees":
                el = np.divide(el, 1.0)  # todo, adds other conversion

            timestamp_azel = input_time[0]

            station.append(data["location"]["fixed"]["data"][0])
            Rs.append(R)
            azs.append(az)
            els.append(el)
            rangings.append(ranging)
            dopplers.append(doppler)
            ras.append(ra)
            decs.append(dec)

            R = []
            az = []
            el = []
            ranging = []
            doppler = []
            ra = []
            dec = []

            ## range

            keys = ["range"]

            ranging_unit = "km"

            if "meta" in data["signal"][i]:
                meta.append([data["signal"][i]["meta"]])

                if "unit" in data["signal"][i]["meta"]:
                    if keys[0] in data["signal"][i]["meta"]["unit"]:
                        ranging_unit = data["signal"][i]["meta"]["unit"][keys[0]]

            else:
                meta.append([])

            input, input_time = extract_key_and_time_from_data(i, data, keys)

            ranging = input[0]
            if ranging_unit == "m" or ranging_unit == "meters":
                ranging = np.divide(ranging, 1000.0)

            timestamp_ranging = input_time[0]

            station.append(data["location"]["fixed"]["data"][0])
            Rs.append(R)
            azs.append(az)
            els.append(el)
            rangings.append(ranging)
            dopplers.append(doppler)
            ras.append(ra)
            decs.append(dec)

            R = []
            az = []
            el = []
            ranging = []
            doppler = []
            ra = []
            dec = []

            ## doppler

            keys = ["doppler"]

            doppler_unit = "hz"
            if "meta" in data["signal"][i]:
                meta.append([data["signal"][i]["meta"]])

                if "unit" in data["signal"][i]["meta"]:
                    if keys[0] in data["signal"][i]["meta"]["unit"]:
                        doppler_unit = data["signal"][i]["meta"]["unit"][keys[0]]

            else:
                meta.append([])

            input, input_time = extract_key_and_time_from_data(i, data, keys)

            doppler = input[0]
            if doppler_unit == "khz" or doppler_unit == "KHz":
                doppler = np.multiply(doppler, 1000.0)

            timestamp_doppler = input_time[0]

            station.append(data["location"]["fixed"]["data"][0])
            Rs.append(R)
            azs.append(az)
            els.append(el)
            rangings.append(ranging)
            dopplers.append(doppler)
            ras.append(ra)
            decs.append(dec)

            R = []
            az = []
            el = []
            ranging = []
            doppler = []
            ra = []
            dec = []

            ## RaDec

            keys = ["ra", "dec"]

            ra_unit = "deg"
            dec_unit = "deg"

            if "meta" in data["signal"][i]:
                meta.append([data["signal"][i]["meta"]])

                if "unit" in data["signal"][i]["meta"]:
                    if keys[0] in data["signal"][i]["meta"]["unit"]:
                        ra_unit = data["signal"][i]["meta"]["unit"][keys[0]]

                    if keys[1] in data["signal"][i]["meta"]["unit"]:
                        dec_unit = data["signal"][i]["meta"]["unit"][keys[1]]

            else:
                meta.append([])

            input, input_time = extract_key_and_time_from_data(i, data, keys)

            ra = input[0]
            if ra_unit == "deg" or ra_unit == "degrees":
                ra = np.divide(az, 1.0)  # todo, adds other conversion

            dec = input[1]
            if dec_unit == "deg" or dec_unit == "degrees":
                dec = np.divide(el, 1.0)  # todo, adds other conversion
            timestamp_radec = input_time[0]

            station.append(data["location"]["fixed"]["data"][0])
            Rs.append(R)
            azs.append(az)
            els.append(el)
            rangings.append(ranging)
            dopplers.append(doppler)
            ras.append(ra)
            decs.append(dec)

            timestamps.append(timestamp_t)
            timestamps.append(timestamp_azel)
            timestamps.append(timestamp_ranging)
            timestamps.append(timestamp_doppler)
            timestamps.append(timestamp_radec)

    # putting the inputs into the measurements
    measurements = {}
    measurements["el"] = els
    measurements["az"] = azs
    measurements["satellite_pos"] = Rs
    measurements["range"] = rangings
    measurements["doppler"] = dopplers
    measurements["ra"] = ras
    measurements["dec"] = decs

    # finding the minimum time stamp. from this on, the epoch for the TLE is calculated
    timestamp_min = 0.0
    tested = 0
    for stamp in range(len(timestamps)):
        if len(timestamps[stamp]):
            if tested == 0:
                timestamp_min = np.min(timestamps[stamp])
                tested = 1
            else:
                if timestamp_min > np.min(timestamps[stamp]):
                    timestamp_min = np.min(timestamps[stamp])

    for stamp in range(len(timestamps)):
        for t0 in range(len(timestamps[stamp])):
            timestamps[stamp][t0] = timestamps[stamp][t0] - timestamp_min

    mode = 0
    parameters = start(opt, station, timestamp_min, timestamps, mode, measurements, meta=meta, generated=generated,
                       loops=40, walks=50)


class optimizer:
    def __init__(self, name='Unknown'):
        self.name = name
        self.test = []

        # for input files
        self.filepath = []

        # for optimization loops
        self.nwalkers = []
        self.walks = []
        self.loops = []
        self.finding = []

        # for optimization
        # modes
        self.orbit = []

        # limits
        self.r_a_lim = []
        self.r_p_lim = []
        self.AoP_lim = []
        self.inc_lim = []
        self.raan_lim = []
        self.tp_lim = []
        self.bstar_lim = []
        self.td_lim = []

        self.station = []
        self.timestamp_min = []
        self.timestamps = []
        self.mode = []
        self.measurements = []

    def add_optimization_runs(self, nwalkers=200, walks=50, loops=50, finding=None, orbit=1):
        self.nwalkers.append(nwalkers)
        self.walks.append(walks)
        self.loops.append(loops)
        self.finding.append(finding)
        self.orbit.append(orbit) # orbit = 0 circle, orbit = 1 ellitic

    def add_limits(self, r_a_lim, r_p_lim, AoP_lim, inc_lim, raan_lim, tp_lim, bstar_lim, td_lim):
        self.r_a_lim.append(r_a_lim)
        self.r_p_lim.append(r_p_lim)
        self.AoP_lim.append(AoP_lim)
        self.inc_lim.append(inc_lim)
        self.raan_lim.append(raan_lim)
        self.tp_lim.append(tp_lim)
        self.bstar_lim.append(bstar_lim)
        self.td_lim.append(td_lim)

    def add_file(self, path):
        filenames = rd.get_all_files(path)
        self.filepath = filenames

    def convert_file(self):
        station, timestamp_min, timestamps, measurements = from_iod(filenames=self.filepath)
        self.station = station
        self.timestamp_min = timestamp_min
        self.timestamps = timestamps
        self.measurements = measurements

    def start(self, opt):
        parameters = start(opt, self.station, self.timestamp_min, self.timestamps, self.mode, self.measurements, meta=[[]], generated={})


if __name__ == "__main__":
    # path = os.path.join("..", "example_data", "iod_23908_20200316")
    # filenames = rd.get_all_files(path)
    # from_iod(filenames=filenames)

    # path = os.path.join("..", "example_data", "json")
    # filenames = rd.get_all_files(path)
    # from_json(filenames=filenames)

    opt = optimizer()
    path = os.path.join("..", "example_data", "iod_23908_20200316")
    opt.add_file(path)
    opt.convert_file()

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

    r_a_lim = [0.0, 2000.0]
    r_p_lim = [0.0, 2000.0]
    AoP_lim = [0.0, 360.0]
    inc_lim = [0.0, 180.0]
    raan_lim = [0.0, 360.0]
    tp_lim = [0.0, 1.0]
    bstar_lim = [-100000.0, 100000.0]
    td_lim = [0.0, 0.0]

    opt.add_optimization_runs(nwalkers=300, walks=50, loops=40, finding=finding)
    opt.add_limits(r_a_lim=r_a_lim, r_p_lim=r_p_lim, AoP_lim=AoP_lim, inc_lim=inc_lim,
                   raan_lim=raan_lim, tp_lim=tp_lim, bstar_lim=bstar_lim, td_lim=td_lim)

    r_a_lim = [-1.0, 1.0]
    r_p_lim = [-4.0, 4.0]
    AoP_lim = [0.0, 360.0]
    inc_lim = [-1.0, 1.0]
    raan_lim = [-1.0, 1.0]
    tp_lim = [0.0, 1.0]
    bstar_lim = [-100000.0, 100000.0]
    td_lim = [0.0, 1.0]

    opt.add_optimization_runs(nwalkers=300, walks=50, loops=40, finding=finding)
    opt.add_limits(r_a_lim=r_a_lim, r_p_lim=r_p_lim, AoP_lim=AoP_lim, inc_lim=inc_lim,
                   raan_lim=raan_lim, tp_lim=tp_lim, bstar_lim=bstar_lim, td_lim=td_lim)

    opt.start(opt)
