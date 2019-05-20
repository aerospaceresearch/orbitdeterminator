import numpy as nu
import datetime as dt
import re
import astropy.time as at

def julian_equinox_from_date(utc_date_time):
	'''
	Helper function.

	Args:
		utc_date_time (datetime object): A datetime object

	Returns:
		A string containing the absolute equinox as per utc_date_time, acceptable by astropy functions
	'''
	return("J" + str(2000.0 + (at.Time(utc_date_time).jd - at.Time(dt.datetime(2000, 1, 1, 12, 0, 0, 0)).jd) / 365.25))

def station_status_iod(status_code):
	'''
	Helper function.

	Args:
		status_code (char): One letter code for station status (from iod_one_line[21])

	Returns:
		Description string as per status_code explaining station status
	'''
	switcher = {
		ord('E'): "Excellent",
		ord('G'): "Good",
		ord('F'): "Fair",
		ord('P'): "Poor",
		ord('B'): "Bad",
		ord('T'): "Terrible",
		ord('C'): "Clouded out",
		ord('O'): "Sky clear, observer not available"
	}
	return(switcher.get(ord(status_code), "NA"))

def epoch_star_chart(epoch_code):
	'''
	Helper function.

	Args:
		epoch_code (char): One letter code for epoch chart (from iod_one_line[45])

	Returns:
		Description string as per epoch_code stating epoch chart
	'''
	switcher = {
		ord('0'): "JDATE",
		ord('1'): "J1855",
		ord('2'): "J1875",
		ord('3'): "J1900",
		ord('4'): "J1950",
		ord('5'): "J2000",
		ord('6'): "J2050"
	}
	return(switcher.get(ord(epoch_code), "J2000"))

def optical_behaviour_iod(behaviour_code):
	'''
	Helper function.

	Args:
		behaviour_code (char): One letter code for optical behaviour (from iod_one_line[65])

	Returns:
		Description string as per behaviour_code explaining optical behaviour
	'''
	switcher = {
		ord('E'): "Unusually faint because of eclipse transition",
		ord('F'): "Flashing with constant flash period",
		ord('I'): "Irregular",
		ord('R'): "Regular variations",
		ord('S'): "Steady",
		ord('X'): "Flashing with irregular flash period",
		ord('B'): "T0 for averaging several flash cycles",
		ord('H'): "One flash in a series",
		ord('P'): "Tn for averaging several flash cycles",
		ord('A'): "Became visible, was invisible",
		ord('D'): "Object in FOV but not visible",
		ord('M'): "Brightest",
		ord('N'): "Faintest",
		ord('V'): "Best seen using averted vision"
	}
	return(switcher.get(ord(behaviour_code), "NA"))

def coord_format_to_deg_iod(clipped_iod, pos_format):
	'''
	Extracts the positional information from the input string as per the input format assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the IOD reporting format by SatObs. For more information, visit http://www.satobs.org/position/IODformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_iod (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the IOD reporting format by SatObs. (from iod_one_line[47:64])
		pos_format: The format, units and resolution assumed by clipped_iod. (from iod_one_line[44])

	Returns:
		A tuple containing the following:
			ra_or_az (double)(degrees): The right acension or azimuth value in clipped_iod. (from clipped_iod[0:7])
			dec_or_el (double)(degrees): The declination or elevation value in clipped_iod. (from clipped_iod[7:14])
			coord_uncertain (double)(degrees): The uncertainty of measurement for ra_or_az and dec_or_el. (from clipped_iod[15:17])
	'''
	ra_or_az = 0.0
	dec_or_el = 0.0
	coord_uncertain = 0.0

	# RA/DEC = HHMMSSs+DDMMSS MX   (MX in seconds of arc)
	if(pos_format is '1'):
		ra_or_az = (15.0 * float(clipped_iod[0:2].replace(' ', '') if len(clipped_iod[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_iod[2:4].replace(' ', '') if len(clipped_iod[2:4].replace(' ', '')) > 0 else "0"))\
			+ (((15.0 / 60.0) / 60.0) * float(clipped_iod[4:6].replace(' ', '') if len(clipped_iod[4:6].replace(' ', '')) > 0 else "0"))\
			+ ((((15.0 / 60.0) / 60.0) / 10.0) * float(clipped_iod[6].replace(' ', '') if len(clipped_iod[6].replace(' ', '')) > 0 else "0"))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[10:12].replace(' ', '') if len(clipped_iod[10:12].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_iod[12:14].replace(' ', '') if len(clipped_iod[12:14].replace(' ', '')) > 0 else "0"))\
			)
		coord_uncertain = (((1.0 / 60.0) / 60.0) * (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8)))))
	
	# RA/DEC = HHMMmmm+DDMMmm MX   (MX in minutes of arc)
	if(pos_format is '2'):
		ra_or_az = (15.0 * float(clipped_iod[0:2].replace(' ', '') if len(clipped_iod[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_iod[2:4].replace(' ', '') if len(clipped_iod[2:4].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * (float(clipped_iod[4:7].replace(' ', '') if len(clipped_iod[4:7].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[4:7].replace(' ', ''))))))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[10:12].replace(' ', '') if len(clipped_iod[10:12].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * (float(clipped_iod[12:14].replace(' ', '') if len(clipped_iod[12:14].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[12:14].replace(' ', ''))))))\
			)
		coord_uncertain = ((1.0 / 60.0) * (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8)))))
	
	# RA/DEC = HHMMmmm+DDdddd MX   (MX in degrees of arc)
	if(pos_format is '3'):
		ra_or_az = (15.0 * float(clipped_iod[0:2].replace(' ', '') if len(clipped_iod[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_iod[2:4].replace(' ', '') if len(clipped_iod[2:4].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * (float(clipped_iod[4:7].replace(' ', '') if len(clipped_iod[4:7].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[4:7].replace(' ', ''))))))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ float(clipped_iod[10:14].replace(' ', '') if len(clipped_iod[10:14].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[10:14].replace(' ', ''))))\
			)
		coord_uncertain = (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8))))
	
	# AZ/EL  = DDDMMSS+DDMMSS MX   (MX in seconds of arc)
	if(pos_format is '4'):
		ra_or_az = float(clipped_iod[0:3].replace(' ', '') if len(clipped_iod[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[3:5].replace(' ', '') if len(clipped_iod[3:5].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_iod[5:7].replace(' ', '') if len(clipped_iod[5:7].replace(' ', '')) > 0 else "0"))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[10:12].replace(' ', '') if len(clipped_iod[10:12].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_iod[12:14].replace(' ', '') if len(clipped_iod[12:14].replace(' ', '')) > 0 else "0"))\
			)
		coord_uncertain = (((1.0 / 60.0) / 60.0) * (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8)))))
	
	# AZ/EL  = DDDMMmm+DDMMmm MX   (MX in minutes of arc)
	if(pos_format is '5'):
		ra_or_az = float(clipped_iod[0:3].replace(' ', '') if len(clipped_iod[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[3:5].replace(' ', '') if len(clipped_iod[3:5].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * (float(clipped_iod[5:7].replace(' ', '') if len(clipped_iod[5:7].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[5:7].replace(' ', ''))))))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_iod[10:12].replace(' ', '') if len(clipped_iod[10:12].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * (float(clipped_iod[12:14].replace(' ', '') if len(clipped_iod[12:14].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[12:14].replace(' ', ''))))))\
			)
		coord_uncertain = ((1.0 / 60.0) * (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8)))))
	
	# AZ/EL  = DDDdddd+DDdddd MX   (MX in degrees of arc)
	if(pos_format is '6'):
		ra_or_az = float(clipped_iod[0:3].replace(' ', '') if len(clipped_iod[0:3].replace(' ', '')) > 0 else "0")\
			+ float(clipped_iod[3:7].replace(' ', '') if len(clipped_iod[3:7].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[3:7].replace(' ', ''))))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ float(clipped_iod[10:14].replace(' ', '') if len(clipped_iod[10:14].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[10:14].replace(' ', ''))))\
			)
		coord_uncertain = (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8))))
	
	# RA/DEC = HHMMSSs+DDdddd MX   (MX in degrees of arc)
	if(pos_format is '7'):
		ra_or_az = (15.0 * float(clipped_iod[0:2].replace(' ', '') if len(clipped_iod[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_iod[2:4].replace(' ', '') if len(clipped_iod[2:4].replace(' ', '')) > 0 else "0"))\
			+ (((15.0 / 60.0) / 60.0) * float(clipped_iod[4:6].replace(' ', '') if len(clipped_iod[4:6].replace(' ', '')) > 0 else "0"))\
			+ ((((15.0 / 60.0) / 60.0) / 10.0) * float(clipped_iod[6].replace(' ', '') if len(clipped_iod[6].replace(' ', '')) > 0 else "0"))
		dec_or_el = (-1 if clipped_iod[7] is '-' else 1)\
			* (\
			float(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0")\
			+ float(clipped_iod[10:14].replace(' ', '') if len(clipped_iod[10:14].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[10:14].replace(' ', ''))))\
			)
		coord_uncertain = (0.0 if (len(clipped_iod[15:17].replace(' ', '')) == 0) else (int(clipped_iod[15]) * (10 ** (int(clipped_iod[16]) - 8))))
	
	return(ra_or_az, dec_or_el, coord_uncertain)

def datetime_from_iod(clipped_iod):
	'''
	Extracts the temporal information from the input string as per the input format assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the IOD reporting format by SatObs. For more information, visit http://www.satobs.org/position/IODformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_iod (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the IOD reporting format by SatObs. (from iod_one_line[23:43])

	Returns:
		A tuple containing the following:
			utc_date_time (datetime object)(UTC): The time of observation in clipped_iod. (from clipped_uk[0:17])
			time_uncertain (double)(seconds): The uncertainty of measurement for utc_date_time in clipped_iod. (from clipped_uk[18:20])
			A number to check if time has been reported or not which will be helpful while printing the tuple
	'''
	# UTC+00:00 Datetime = YYYYMMDDHHMMSSsss
	utc_date_time = dt.datetime(int(clipped_iod[0:4]), int(clipped_iod[4:6]), int(clipped_iod[6:8]),\
		int(clipped_iod[8:10].replace(' ', '') if len(clipped_iod[8:10].replace(' ', '')) > 0 else "0"),\
		int(clipped_iod[10:12].replace(' ', '') if len(clipped_iod[10:12].replace(' ', '')) > 0 else "0"),\
		int(clipped_iod[12:14].replace(' ', '') if len(clipped_iod[12:14].replace(' ', '')) > 0 else "0"),\
		int(clipped_iod[14:17].replace(' ', '') if len(clipped_iod[14:17].replace(' ', '')) > 0 else "0") * (10 ** (6 - len(clipped_iod[14:17].replace(' ', '')))),\
		tzinfo=dt.timezone.utc)

	# uncertainty = MX
	time_uncertain = "NA" if (len(clipped_iod[18:20].replace(' ', '')) == 0) else str(int(clipped_iod[18]) * (10 ** (int(clipped_iod[19]) - 8)))
	return(utc_date_time, time_uncertain, len(clipped_iod[8:17].replace(' ', '')))

def pos_coord_to_str(position_deg):
	'''
	Prints positions and their uncertainties in readable formats.

	Args:
		A tuple returned by coord_format_to_deg_iod(...) and coord_format_to_deg_uk(...)

	Returns:
		A pretty string with position, uncertainty and units
	'''
	# Maximum 4 significant digits
	return(str("%.4f" % position_deg[0]) + ((" " + u"\u00B1" + str("%.4f" % position_deg[2])) if position_deg[2] != 0.0 else "") + "deg " + ("+" if position_deg[1] >= 0.0 else "") + str("%.4f" % position_deg[1]) + ((" " + u"\u00B1" + str("%.4f" % position_deg[2])) if position_deg[2] != 0.0 else "") + "deg")

def datetime_to_str(utc_date_time_uncertain):
	'''
	Prints datetime objects and their uncertainties in readable formats.

	Args:
		A tuple returned by datetime_from_iod(...) and datetime_from_uk(...)

	Returns:
		A pretty string with date, time, uncertainty and units
	'''
	# Time might not have been reported
	return((str(utc_date_time_uncertain[0])[0:10] + " UTC+00:00") if (utc_date_time_uncertain[2] == 0) else ((str(utc_date_time_uncertain[0])[0:23] if str(utc_date_time_uncertain[0])[19] != '+' else str(utc_date_time_uncertain[0])[0:19]) + ("" if utc_date_time_uncertain[1] is "NA" else (" " + u"\u00B1" + utc_date_time_uncertain[1] + "sec")) + " UTC+00:00"))

def apparent_mag_iod(clipped_iod):
	'''
	Extracts the apparent magnitude from the input string assuming that the input string is the clipped substring
	of a single positional observation (one line of 80 characters) from a file that follows the IOD reporting format by SatObs.
	For more information, visit http://www.satobs.org/position/IODformat.html and read detailed documentation (recommended).

	Args:
		clipped_iod (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the IOD reporting format by SatObs. (from iod_one_line[66:73])

	Returns:
		A pretty string containing apparent magnitude and its measurement uncertainty
	'''
	app_mag = (-1 if clipped_iod[0] is '-' else 1)\
			* (\
			float(clipped_iod[1:3].replace(' ', '') if len(clipped_iod[1:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 10.0) * float(clipped_iod[3].replace(' ', '') if len(clipped_iod[3].replace(' ', '')) > 0 else "0"))\
			)
	app_mag_uncertain = float(clipped_iod[5].replace(' ', '') if len(clipped_iod[5].replace(' ', '')) > 0 else "0")\
		+ ((1.0 / 10.0) * float(clipped_iod[6].replace(' ', '') if len(clipped_iod[6].replace(' ', '')) > 0 else "0"))
	return(("NA" if len(clipped_iod[0:4].replace(' ', '')) == 0 else (("+" if app_mag >= 0.0 else "") + str("%.1f" % app_mag))) + ("" if len(clipped_iod[5:7].replace(' ', '')) == 0 else (" " + u"\u00B1" + str("%.1f" % app_mag_uncertain))))

def flash_period_iod(clipped_iod):
	'''
	Extracts the flash period information from from the input string assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the IOD reporting format by SatObs. For more information, visit http://www.satobs.org/position/IODformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_iod (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the IOD reporting format by SatObs. (from iod_one_line[74:80])

	Returns:
		A pretty string for the input string containing flash period and units
	'''
	flash_period = float(clipped_iod[0:3].replace(' ', '') if len(clipped_iod[0:3].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_iod[3:6].replace(' ', '') if len(clipped_iod[3:6].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_iod[3:6].replace(' ', '')))))
	return("NA" if len(clipped_iod.replace(' ', '')) == 0 else (str(flash_period) + "sec"))

def sat_info_iod(clipped_iod):
	'''
	Extracts the satellite's NORAD identification information from from the input string assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the IOD reporting format by SatObs. For more information, visit http://www.satobs.org/position/IODformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_iod (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the IOD reporting format by SatObs. (from iod_one_line[0:15])

	Returns:
		A tuple containing the following:
			catalog_number (string): NORAD catalog number of the satellite (from clipped_iod[0:5])
			international_designator (string): NORAD international designator of the satellite (from clipped_iod[6:15])
	'''
	catalog_number = "NA" if len(clipped_iod[0:5].replace(' ', '')) == 0 else clipped_iod[0:5].replace(' ', '')
	international_designator = "NA" if (len(clipped_iod[6:15].replace(' ', '')) == 0) else (("20" if int(clipped_iod[6:8].replace(' ', '')) < 57 else "19") + clipped_iod[6:8].replace(' ', '') + '-' + clipped_iod[9:15].replace(' ', ''))
	return(catalog_number, international_designator)

def parse_iod(iod):
	'''
	Extracts the all the positional information from from the input string assuming that
	the input string is a single positional observation (one line of 80 characters) from a file
	that follows the IOD reporting format by SatObs. For more information, visit http://www.satobs.org/position/IODformat.html
	and read detailed documentation (recommended).

	Args:
		iod (string): A single positional observation (one line of 80 characters) from
		a file that follows the IOD reporting format by SatObs.

	Returns:
		A tuple that contains the following:
			sat_info[0] (string): NORAD catalog number
			sat_info[1] (string): NORAD international designator
			station_number (string): Station number
			station_status (string): Station status
			utc_date_time_uncertain (tuple): Returned by datetime_from_iod(...)
			epoch_chart (string): Assumed equinox
			position_deg (tuple): Returned by coord_format_to_deg_iod(...)
			optical_behaviour (string): Optical behaviour
			brightness_mag (string): Apparent magnitude
			flash_period (string): Time period of flashing
	'''
	if(len(iod) < 80):
		for i in range(80 - len(iod)):
			iod = iod + " "
	sat_info = sat_info_iod(iod[0:15])
	station_number = "NA" if len(iod[16:20].replace(' ', '')) == 0 else iod[16:20].replace(' ', '')
	station_status = station_status_iod(iod[21])
	utc_date_time_uncertain = datetime_from_iod(iod[23:43])
	position_deg = coord_format_to_deg_iod(iod[47:64], iod[44])
	epoch_chart = epoch_star_chart(iod[45]) if epoch_star_chart(iod[45]) != "JDATE" else julian_equinox_from_date(utc_date_time_uncertain[0])
	optical_behaviour = optical_behaviour_iod(iod[65])
	brightness_mag = apparent_mag_iod(iod[66:73])
	flash_period = flash_period_iod(iod[74:80])
	return(sat_info[0], sat_info[1], station_number, station_status, utc_date_time_uncertain, epoch_chart, position_deg, optical_behaviour, brightness_mag, flash_period)

def optical_behaviour_uk(status_code):
	'''
	Helper function.

	Args:
		status_code (char): One letter code for optical behaviour (from uk_one_line[79])

	Returns:
		Description string as per status_code explaining optical behaviour
	'''
	switcher = {
		ord('S'): "Steady magnitude",
		ord('I'): "Irregular variation in brightness",
		ord('R'): "Regular variation in brightness",
		ord('F'): "Flashing with constant flash period",
		ord('X'): "Flashing with irregular flash period",
		ord('E'): "Unusually faint because of eclipse transition"
	}
	return(switcher.get(ord(status_code), "NA"))

def time_standard_uk(time_standard):
	'''
	Helper function.

	Args:
		time_standard (char): One letter code for time standard (from uk_one_line[32])

	Returns:
		Description string as per time_standard stating the time standard used
	'''
	switcher = {
		ord('1'): "Radio Time Signal",
		ord('2'): "UK P.O. Speaking Clock",
		ord('3'): "BBC Time Pips"
	}
	return(switcher.get(ord(time_standard), "Radio Time Signal"))

def datetime_from_uk(clipped_uk):
	'''
	Extracts the temporal information from the input string as per the input format assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the UK reporting format by SatObs. For more information, visit http://www.satobs.org/position/UKformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_uk (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the UK reporting format by SatObs. (from uk_one_line[11:32])

	Returns:
		A tuple containing the following:
			utc_date_time (datetime object)(UTC): The time of observation in clipped_uk. (from clipped_uk[0:16])
			time_uncertain (double)(seconds): The uncertainty of measurement for utc_date_time in clipped_uk. (from clipped_uk[16:21])
			A number to check if time has been reported or not which will be helpful while printing the tuple
	'''
	# UTC+00:00 Datetime = YYMMDDHHMMSSssss
	utc_date_time = dt.datetime((2000 if int(clipped_uk[0:2]) < 57 else 1900) + int(clipped_uk[0:2]), int(clipped_uk[2:4]), int(clipped_uk[4:6]),\
		int(clipped_uk[6:8].replace(' ', '') if len(clipped_uk[6:8].replace(' ', '')) > 0 else "0"),\
		int(clipped_uk[8:10].replace(' ', '') if len(clipped_uk[8:10].replace(' ', '')) > 0 else "0"),\
		int(clipped_uk[10:12].replace(' ', '') if len(clipped_uk[10:12].replace(' ', '')) > 0 else "0"),\
		int(clipped_uk[12:16].replace(' ', '') if len(clipped_uk[12:16].replace(' ', '')) > 0 else "0") * (10 ** (6 - len(clipped_uk[12:16].replace(' ', '')))),\
		tzinfo=dt.timezone.utc)

	# uncertainty = Ttttt (seconds)
	time_uncertain = "NA" if (len(clipped_uk[16:21].replace(' ', '')) == 0) else ((clipped_uk[16].replace(' ', '') if len(clipped_uk[16].replace(' ', '')) > 0 else "0") + "." + (clipped_uk[17:21].replace(' ', '') if len(clipped_uk[17:21].replace(' ', '')) > 0 else "0"))
	return(utc_date_time, time_uncertain, len(clipped_uk[6:16].replace(' ', '')))

def flash_period_uk(clipped_uk):
	'''
	Extracts the flash period information from from the input string assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the UK reporting format by SatObs. For more information, visit http://www.satobs.org/position/UKformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_uk (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the UK reporting format by SatObs. (from uk_one_line[74:79])

	Returns:
		A pretty string for the input string containing flash period and units
	'''
	# flash period = SSSss (seconds)
	flash_period = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[3:5].replace(' ', '') if len(clipped_uk[3:5].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[3:5].replace(' ', '')))))
	return("NA" if len(clipped_uk.replace(' ', '')) == 0 else (str(flash_period) + "sec"))

def sat_info_uk(clipped_uk):
	'''
	Extracts the satellite's NORAD identification information from from the input string assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the UK reporting format by SatObs. For more information, visit http://www.satobs.org/position/UKformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_uk (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the UK reporting format by SatObs. (from uk_one_line[0:7])

	Returns:
		A pretty string for the input string with NORAD identification
	'''
	return("NA" if (len(clipped_uk[0:7].replace(' ', '')) == 0 or clipped_uk[0:7].replace(' ', '') == "9900000") else (("20" if int(clipped_uk[0:2].replace(' ', '')) < 57 else "19") + clipped_uk[0:2].replace(' ', '') + '-' + clipped_uk[2:5].replace(' ', '') + chr(64 + int(clipped_uk[5:7].replace(' ', '')))))

def coord_format_to_deg_uk(clipped_uk, pos_format):
	'''
	Extracts the positional information from the input string as per the input format assuming that
	the input string is the clipped substring of a single positional observation (one line of 80 characters) from a file
	that follows the UK reporting format by SatObs. For more information, visit http://www.satobs.org/position/UKformat.html
	and read detailed documentation (recommended).

	Args:
		clipped_uk (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the UK reporting format by SatObs. (from uk_one_line[34:54])
		pos_format: The format, units and resolution assumed by clipped_uk. (from uk_one_line[33])

	Returns:
		A tuple containing the following:
			ra_or_az (double)(degrees): The right acension or azimuth value in clipped_uk. (from clipped_uk[0:8])
			dec_or_el (double)(degrees): The declination or elevation value in clipped_uk. (from clipped_uk[8:16])
			coord_uncertain (double)(degrees): The uncertainty of measurement for ra_or_az and dec_or_el. (from clipped_uk[16:20])
	'''
	ra_or_az = 0.0
	dec_or_el = 0.0
	coord_uncertain = 0.0
	# RA/DEC = HHMMSSss+DDMMSSsSSSs (seconds of arc)
	if(pos_format is '1'):
		ra_or_az = (15.0 * float(clipped_uk[0:2].replace(' ', '') if len(clipped_uk[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_uk[2:4].replace(' ', '') if len(clipped_uk[2:4].replace(' ', '')) > 0 else "0"))\
			+ (((15.0 / 60.0) / 60.0) * float(clipped_uk[4:6].replace(' ', '') if len(clipped_uk[4:6].replace(' ', '')) > 0 else "0"))\
			+ (((15.0 / 60.0) / 60.0) * float(clipped_uk[6:8].replace(' ', '') if len(clipped_uk[6:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[6:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_uk[13:15].replace(' ', '') if len(clipped_uk[13:15].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[15].replace(' ', '') if len(clipped_uk[15].replace(' ', '')) > 0 else "0"))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			(((1.0 / 60.0) / 60.0) * float(clipped_uk[16:19].replace(' ', '') if len(clipped_uk[16:19].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[19].replace(' ', '') if len(clipped_uk[19].replace(' ', '')) > 0 else "0"))\
			)

	# RA/DEC = HHMMmmmm+DDMMmmmMMmm (minutes of arc)
	if(pos_format is '2'):
		ra_or_az = (15.0 * float(clipped_uk[0:2].replace(' ', '') if len(clipped_uk[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_uk[2:4].replace(' ', '') if len(clipped_uk[2:4].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_uk[4:8].replace(' ', '') if len(clipped_uk[4:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[4:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[13:16].replace(' ', '') if len(clipped_uk[13:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[13:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			((1.0 / 60.0) * float(clipped_uk[16:18].replace(' ', '') if len(clipped_uk[16:18].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[18:20].replace(' ', '') if len(clipped_uk[18:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[18:20].replace(' ', '')))))\
			)

	# RA/DEC = HHMMmmmm+DDdddddDddd (degrees of arc)
	if(pos_format is '3'):
		ra_or_az = (15.0 * float(clipped_uk[0:2].replace(' ', '') if len(clipped_uk[0:2].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_uk[2:4].replace(' ', '') if len(clipped_uk[2:4].replace(' ', '')) > 0 else "0"))\
			+ ((15.0 / 60.0) * float(clipped_uk[4:8].replace(' ', '') if len(clipped_uk[4:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[4:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[11:16].replace(' ', '') if len(clipped_uk[11:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[11:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			float(clipped_uk[16].replace(' ', '') if len(clipped_uk[16].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[17:20].replace(' ', '') if len(clipped_uk[17:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[17:20].replace(' ', '')))))\
			)

	# AZ/EL = DDDMMSSs+DDMMSSsSSSs (seconds of arc) (elevation corrected for refraction)
	if(pos_format is '4'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[3:5].replace(' ', '') if len(clipped_uk[3:5].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_uk[5:7].replace(' ', '') if len(clipped_uk[5:7].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[7].replace(' ', '') if len(clipped_uk[7].replace(' ', '')) > 0 else "0"))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_uk[13:15].replace(' ', '') if len(clipped_uk[13:15].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[15].replace(' ', '') if len(clipped_uk[15].replace(' ', '')) > 0 else "0"))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			(((1.0 / 60.0) / 60.0) * float(clipped_uk[16:19].replace(' ', '') if len(clipped_uk[16:19].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[19].replace(' ', '') if len(clipped_uk[19].replace(' ', '')) > 0 else "0"))\
			)

	# AZ/EL = DDDMMmmm+DDMMmmmMMmm (minutes of arc) (elevation corrected for refraction)
	if(pos_format is '5'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[3:5].replace(' ', '') if len(clipped_uk[3:5].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[5:8].replace(' ', '') if len(clipped_uk[5:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[5:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[13:16].replace(' ', '') if len(clipped_uk[13:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[13:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			((1.0 / 60.0) * float(clipped_uk[16:18].replace(' ', '') if len(clipped_uk[16:18].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[18:20].replace(' ', '') if len(clipped_uk[18:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[18:20].replace(' ', '')))))\
			)

	# AZ/EL = DDDddddd+DDdddddDddd (degrees of arc) (elevation corrected for refraction)
	if(pos_format is '6'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
		+ (float(clipped_uk[3:8].replace(' ', '') if len(clipped_uk[3:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[3:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[11:16].replace(' ', '') if len(clipped_uk[11:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[11:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			float(clipped_uk[16].replace(' ', '') if len(clipped_uk[16].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[17:20].replace(' ', '') if len(clipped_uk[17:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[17:20].replace(' ', '')))))\
			)

	# AZ/EL = DDDMMSSs+DDMMSSsSSSs (seconds of arc) (elevation not corrected for refraction)
	if(pos_format is '7'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[3:5].replace(' ', '') if len(clipped_uk[3:5].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_uk[5:7].replace(' ', '') if len(clipped_uk[5:7].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[7].replace(' ', '') if len(clipped_uk[7].replace(' ', '')) > 0 else "0"))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ (((1.0 / 60.0) / 60.0) * float(clipped_uk[13:15].replace(' ', '') if len(clipped_uk[13:15].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[15].replace(' ', '') if len(clipped_uk[15].replace(' ', '')) > 0 else "0"))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			(((1.0 / 60.0) / 60.0) * float(clipped_uk[16:19].replace(' ', '') if len(clipped_uk[16:19].replace(' ', '')) > 0 else "0"))\
			+ ((((1.0 / 60.0) / 60.0) / 10.0) * float(clipped_uk[19].replace(' ', '') if len(clipped_uk[19].replace(' ', '')) > 0 else "0"))\
			)

	# AZ/EL = DDDMMmmm+DDMMmmmMMmm (minutes of arc) (elevation not corrected for refraction)
	if(pos_format is '8'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[3:5].replace(' ', '') if len(clipped_uk[3:5].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[5:8].replace(' ', '') if len(clipped_uk[5:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[5:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ ((1.0 / 60.0) * float(clipped_uk[11:13].replace(' ', '') if len(clipped_uk[11:13].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[13:16].replace(' ', '') if len(clipped_uk[13:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[13:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			((1.0 / 60.0) * float(clipped_uk[16:18].replace(' ', '') if len(clipped_uk[16:18].replace(' ', '')) > 0 else "0"))\
			+ ((1.0 / 60.0) * float(clipped_uk[18:20].replace(' ', '') if len(clipped_uk[18:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[18:20].replace(' ', '')))))\
			)

	# AZ/EL = DDDddddd+DDdddddDddd (degrees of arc) (elevation not corrected for refraction)
	if(pos_format is '9'):
		ra_or_az = float(clipped_uk[0:3].replace(' ', '') if len(clipped_uk[0:3].replace(' ', '')) > 0 else "0")\
		+ (float(clipped_uk[3:8].replace(' ', '') if len(clipped_uk[3:8].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[3:8].replace(' ', '')))))
		dec_or_el = (-1 if clipped_uk[8] is '-' else 1)\
			* (\
			float(clipped_uk[9:11].replace(' ', '') if len(clipped_uk[9:11].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[11:16].replace(' ', '') if len(clipped_uk[11:16].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[11:16].replace(' ', '')))))\
			)
		coord_uncertain = 0.0 if len(clipped_uk[16:20].replace(' ', '')) == 0 else (\
			float(clipped_uk[16].replace(' ', '') if len(clipped_uk[16].replace(' ', '')) > 0 else "0")\
			+ (float(clipped_uk[17:20].replace(' ', '') if len(clipped_uk[17:20].replace(' ', '')) > 0 else "0") * (10.0 ** (-1 * len(clipped_uk[17:20].replace(' ', '')))))\
			)
	
	return(ra_or_az, dec_or_el, coord_uncertain)

def range_val_uncertain_uk(clipped_uk):
	'''
	Extracts the range information from the input string assuming that the input string is the clipped substring
	of a single positional observation (one line of 80 characters) from a file that follows the UK reporting format by SatObs.
	For more information, visit http://www.satobs.org/position/UKformat.html and read detailed documentation (recommended).

	Args:
		clipped_uk (string): The clipped substring of a single positional observation (one line of 80 characters)
		from a file that follows the UK reporting format by SatObs. (from uk_one_line[55:68])

	Returns:
		A tuple containing the following:
			range_val (double)(kilometers): The range value in clipped_uk. (from clipped_uk[0:8])
			range_uncertain (double)(kilometers): The uncertainty of measurement for range_val. (from clipped_uk[8:13])
	'''
	# Range = NNNNNnnnNNnnn (kilometers)
	range_val = (-1.0 if len(clipped_uk[0:5].replace(' ', '')) == 0 else float(clipped_uk[0:5].replace(' ', '')))\
		+ (0.0 if len(clipped_uk[5:8].replace(' ', '')) == 0 else (float(clipped_uk[5:8].replace(' ', '')) * (10.0 ** (-1 * len(clipped_uk[5:8].replace(' ', ''))))))
	range_uncertain = (-1.0 if len(clipped_uk[8:10].replace(' ', '')) == 0 else float(clipped_uk[8:10].replace(' ', '')))\
		+ (0.0 if len(clipped_uk[10:13].replace(' ', '')) == 0 else (float(clipped_uk[10:13].replace(' ', '')) * (10.0 ** (-1 * len(clipped_uk[10:13].replace(' ', ''))))))
	return(range_val, range_uncertain)

def range_to_str_uk(range_val_uncertain):
	'''
	Converts the tuple returned by range_val_uncertain_uk(...) to pretty string.

	Args:
		range_val_uncertain (tuple): The tuple returned by range_val_uncertain_uk(...)

	Returns:
		A pretty string for the input tuple
	'''
	return(("NA" if range_val_uncertain[0] < 0.0 else str(range_val_uncertain[0]))\
		+ ("" if range_val_uncertain[1] < 0.0 else (" " + u"\u00B1" + str("%.3f" % range_val_uncertain[1])))\
		+ ("" if range_val_uncertain[0] < 0.0 else "Km")
		)

def parse_uk(uk):
	'''
	Extracts the all the positional information from from the input string assuming that
	the input string is a single positional observation (one line of 80 characters) from a file
	that follows the UK reporting format by SatObs. For more information, visit http://www.satobs.org/position/UKformat.html
	and read detailed documentation (recommended).

	Args:
		uk (string): A single positional observation (one line of 80 characters) from
		a file that follows the UK reporting format by SatObs.

	Returns:
		A tuple that contains the following:
			sat_info (string): NORAD international designator
			site_num (string): Station number
			utc_date_time_uncertain (tuple): Returned by datetime_from_uk(...)
			time_standard (string): Assumed time standard
			epoch_chart (string): Assumed equinox
			position_deg (tuple): Returned by coord_format_to_deg_uk(...)
			range_val_uncertain (tuple): Returned by range_val_uncertain_uk(...)
			brightest_mag (string): Brightest apparent magnitude
			faintest_mag (string): Faintest apparent magnitude
			flash_period (string): Time period of flashing
			remarks (string): Optical behaviour
	'''
	if(len(uk) < 80):
		for i in range(80 - len(uk)):
			uk = uk + " "
	sat_info = sat_info_uk(uk[0:7])
	site_num = "NA" if len(uk[7:11].replace(' ', '')) == 0 else uk[7:11].replace(' ', '')
	utc_date_time_uncertain = datetime_from_uk(uk[11:32])
	time_standard = time_standard_uk(uk[32])
	position_deg = coord_format_to_deg_uk(uk[34:54], uk[33])
	epoch_chart = epoch_star_chart(uk[54]) if epoch_star_chart(uk[54]) != "JDATE" else julian_equinox_from_date(utc_date_time_uncertain[0])
	range_val_uncertain = range_val_uncertain_uk(uk[55:68])
	brightest_mag = "NA" if len(uk[68:71].replace(' ', '')) == 0 else (("+" if (uk[68] != "+" and uk[68] != "-") else "") + uk[68:70].replace(' ', '0') + "." + uk[70].replace(' ', '0'))
	faintest_mag = "NA" if (len(uk[71:74].replace(' ', '')) == 0 and uk[79] != 'S') else ("Constant apparent magnitude" if (len(uk[71:74].replace(' ', '')) == 0 and uk[79] == 'S') else ("Invisible" if uk[71:74] == "INV" else (("+" if (uk[71] != "+" and uk[71] != "-") else "") + uk[71:73].replace(' ', '0') + "." + uk[73].replace(' ', '0'))))
	flash_period = flash_period_uk(uk[74:79])
	remarks = optical_behaviour_uk(uk[79])
	return(sat_info, site_num, utc_date_time_uncertain, time_standard, epoch_chart, position_deg, range_val_uncertain, brightest_mag, faintest_mag, flash_period, remarks)

def print_iod(iod):
	'''
	Prints all the positional information, taking a single positional observation (one line of 80 characters)
	at a time from a file that follows the IOD reporting format by SatObs.
	For more information, visit http://www.satobs.org/position/IODformat.html and read detailed documentation (recommended).

	Args:
		iod (list): A list made of single positional observations (one line of 80 characters per observation)
		from a file that follows the IOD reporting format by SatObs.

	Returns:
		void
	'''
	iod_data_enum = ["Catalog Number", "International Designator", "Station Number", "Station Status", "Timestamp", "Equinox Star Chart", "Position(RA/AZ DEC/EL)", "Optical Behaviour", "Apparent Magnitude", "Flash Period"]
	print("\n***************Retrieved IOD Data***************\n")
	for i in range(0, len(iod)):
		iod_data = parse_iod(iod[i])
		for j in range(0, len(iod_data_enum)):
			if(j is 4):
				print(iod_data_enum[j] + ": " + datetime_to_str(iod_data[j]))
			elif(j is 6):
				position_deg_str = pos_coord_to_str(iod_data[j])
				print("Position(" + ("AZ" if (ord(iod[i][44]) > 51 and ord(iod[i][44]) < 55) else "RA") + "): " + re.split(" ", position_deg_str)[0] + ((" " + re.split(" ", position_deg_str)[1]) if len(re.split(" ", position_deg_str)) > 2 else ""))
				print("Position(" + ("EL" if (ord(iod[i][44]) > 51 and ord(iod[i][44]) < 55) else "DEC") + "): " + re.split(" ", position_deg_str)[2 if len(re.split(" ", position_deg_str)) > 2 else 1] + ((" " + re.split(" ", position_deg_str)[3]) if len(re.split(" ", position_deg_str)) > 2 else ""))
			else:
				print(iod_data_enum[j] + ": " + str(iod_data[j]))
		print("")

def print_uk(uk):
	'''
	Prints all the positional information, taking a single positional observation (one line of 80 characters)
	at a time from a file that follows the UK reporting format by SatObs.
	For more information, visit http://www.satobs.org/position/UKformat.html and read detailed documentation (recommended).

	Args:
		uk (list): A list made of single positional observations (one line of 80 characters per observation)
		from a file that follows the UK reporting format by SatObs.

	Returns:
		void
	'''
	uk_data_enum = ["International Designator", "Site Number", "Timestamp", "Time Standard", "Equinox Star Chart", "Position(RA/AZ DEC/EL)", "Range", "Brightest Apparent Magnitude", "Faintest Apparent Magnitude", "Flash Period", "Remarks"]
	print("\n***************Retrieved UK Data***************\n")
	for i in range(0, len(uk)):
		uk_data = parse_uk(uk[i])
		for j in range(0, len(uk_data_enum)):
			if(j is 2):
				print(uk_data_enum[j] + ": " + datetime_to_str(uk_data[j]))
			elif(j is 5):
				position_deg_str = pos_coord_to_str(uk_data[j])
				print("Position(" + ("AZ" if (ord(uk[i][33]) > 51) else "RA") + "): " + re.split(" ", position_deg_str)[0] + ((" " + re.split(" ", position_deg_str)[1]) if len(re.split(" ", position_deg_str)) > 2 else ""))
				print("Position(" + ("EL" if (ord(uk[i][33]) > 51) else "DEC") + "): " + re.split(" ", position_deg_str)[2 if len(re.split(" ", position_deg_str)) > 2 else 1] + ((" " + re.split(" ", position_deg_str)[3]) if len(re.split(" ", position_deg_str)) > 2 else ""))
			elif(j is 6):
				print(uk_data_enum[j] + ": " + range_to_str_uk(uk_data[j]))
			else:
				print(uk_data_enum[j] + ": " + str(uk_data[j]))
		print("")

if(__name__ == "__main__"):
	iod_file = []
	iod_file.append("23794 96 010A   2701 G 20040506012614270 17 25 1100114-184298 38 I+020 10 01189") # Includes everything
	# iod_file.append("90019 03 790B   2701 G 20040506020755480 17 25 0929080-203364 48")
	# iod_file.append("90019 03 790B   2701 G 20040506020932610 17 45 1029694-220449 67")
	# iod_file.append("90019 03 790B   2701 G 20040506021046340 17 25 1115711-225466 37")
	# iod_file.append("90019 03 790B   2701 G 20040506021115210 17 45 1133310-231006 78")
	# iod_file.append("23794 96 010A   2701 P 20040506061610940 17 25 1045488+105544 19 I-010 10")
	# iod_file.append("23794 96 010A   2701 P 20040506061636730 17 25 1252114+021122 29 I-020 10")
	# iod_file.append("23794 96 010A   2701 P 20040506061641360 17 25 1334003-010390 58 I-020 10")
	# iod_file.append("23794 96 010A   2701 P 20040506061735610 17 25 1909776-205541 97 I")
	print_iod(iod_file)

	uk_file = []
	uk_file.append("9701201201803101520195542  01   12172038  +15585   1  50451923400667+6 +8   190R") # Includes everything
	# uk_file.append("0401401267504050320170296  010  14102706  +36412   50 5")
	# uk_file.append("0401401267504050320171054  010  12102406  +41279   50 5")
	# uk_file.append("0401402267504050320192783  020  12100282  +21570   20 5")
	# uk_file.append("0401402267504050320200763  020  12090786  +47320  100 5")
	# uk_file.append("9906701267504050320381348  020  12114955  +16154   15 5")
	# uk_file.append("9607201267504050320421970  010  12153504  +26400   60 5")
	# uk_file.append("9607201267504050320423927  010  12162845  +41547   40 5")
	# uk_file.append("9107603267504050320472151  010  12135798  +22002   50 5")
	# uk_file.append("9107604267504050320473073  010  12135931  +21233   50 5")
	# uk_file.append("9107603267504050320480282  010  12144997  +28369   10 5")
	# uk_file.append("9107605267504050320483131  010  12152189  +32549   20 5")
	print_uk(uk_file)

	# rde_file = []
	# rde_file.append("2420 0309 0.211 1204")
	# rde_file.append("03")
	# rde_file.append("0300203 195141.71 005932+313611 4.4 4.4 0 S")
	# rde_file.append("8502002 195853.74 010744+311633 5.1 5.1 0 S")
	# rde_file.append("04")
	# rde_file.append("7704401 015519.00 212317+765245 6.3 7.5 4 R")
	# rde_file.append("7704401 015559.35 221342+565110 6.0 7.2 4 R")
	# rde_file.append("8700602 020411.32 151747+741235 4.8 4.8 0 S")
	# rde_file.append("8205102 021426.21 021503+282843 4.2 4.2 0 S")
	# rde_file.append("999")
