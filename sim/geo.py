from math import radians, sin, cos, acos
import numpy as np


###########################################################
# Calculate approximate projection based on region latitude

# We set the latitude here for the region
LAT_DEG = 30.6
LAT_RAD = np.deg2rad(LAT_DEG)

# Source:
# https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude
METERS_PER_DEG_LAT = 111132.92 - 559.82 * cos(2 * LAT_RAD) + 1.175 * cos(4 * LAT_RAD) - 0.0023 * cos(6*LAT_RAD)
METERS_PER_DEG_LNG = 111412.84 * cos(LAT_RAD) - 93.5 * cos(3*LAT_RAD) + 0.118 * cos(5*LAT_RAD)


def great_circle_distance(lat1, lng1, lat2, lng2):
    """Returns great circle distance in meters."""
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    return 6371.0088 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2)))
