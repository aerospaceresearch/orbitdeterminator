"""Converts coordinates in TEME frame to ECEF frame."""

from datetime import datetime, timezone
import numpy as np

def conv_to_ecef(coords):
    """Converts coordinates in TEME frame to ECEF frame.

       Args:
           coords(nx4 numpy array): list of coordinates in the format [t,x,y,z]

       Returns:
           nx4 numpy array: list of coordinates in the format
                            [t, latitude, longitude, altitude]

                            Note that these coordinates are with respect to the
                            surface of the Earth. Latitude, longitude are in degrees.
    """

    t = coords[:,0]
    x = coords[:,1]
    y = coords[:,2]
    z = coords[:,3]

    alt = (x**2+y**2+z**2)**0.5
    lat = np.degrees(np.arcsin(z/alt))
    lng = np.degrees(np.arctan2(y,x)%(2*np.pi))

    midnight = datetime.fromtimestamp(t[0],tz=timezone.utc)
    midnight = midnight.replace(hour=0,minute=0,second=0,microsecond=0)
    t_mid = midnight.timestamp()

    J2000 = 946728000
    Tu = (t_mid-J2000)/86400/36525
    tg0h = 24110.54841 + 8640184.812866*Tu + 0.093104*Tu**2 - 6.2e-6*Tu**3
    we = 1.00273790935
    tgt = tg0h + we*(t-t_mid)
    era = (tgt%86400)*360/86400
    lng = lng-era
    return np.column_stack((t,lat,lng,alt))

if __name__ == "__main__":
    ecef_coords = conv_to_ecef(np.array([[1521562500,768.281,5835.68,2438.076],[1521562500,768.281,5835.68,2438.076],[1521562500,768.281,5835.68,2438.076]]))
    print(ecef_coords)
