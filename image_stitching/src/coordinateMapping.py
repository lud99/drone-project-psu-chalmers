import numpy as np
import math


def pixel_to_gps(
    pixel, camera_location, altitude, orientation=0, fov=83.0, resolution=(1920, 1080)
):
    """
    Converts a pixel position to GPS coordinates, based on camera location, altitude, and FoV.
    """
    EARTH_RADIUS = 6378137  # in meters  # noqa: N806
    lat, lon = camera_location
    width, height = resolution
    x, y = pixel

    # Calculate the field of view in vertical direction based on image aspect ratio
    fov_width, fov_height = fov, fov * (height / width)

    # Calculate the ground distance covered by the image
    ground_height = 2 * (np.tan(np.radians(fov_height / 2)) * altitude)
    ground_width = 2 * (np.tan(np.radians(fov_width / 2)) * altitude)

    # Calculate the ground distance covered by each pixel
    pixel_width = ground_width / width
    pixel_height = ground_height / height

    # Calculate offsets for the given pixel from the image center
    x_offset = pixel_width * (x - width / 2)
    y_offset = pixel_height * (height / 2 - y)  # Invert Y axis

    # Rotate the offsets by the orientation
    orientation_rad = np.radians(orientation)
    x_offset_rotated = x_offset * np.cos(orientation_rad) + y_offset * np.sin(
        orientation_rad
    )
    y_offset_rotated = -x_offset * np.sin(orientation_rad) + y_offset * np.cos(
        orientation_rad
    )

    # Calculate new coordinates
    delta_lat = (y_offset_rotated / EARTH_RADIUS) * (180 / np.pi)
    delta_long = (x_offset_rotated / (EARTH_RADIUS * np.cos(np.radians(lat)))) * (
        180 / np.pi
    )

    return lat + delta_lat, lon + delta_long


def gps_to_delta_meters(origin_coord, coord):
    """
    Calculates object's (x,y) meter offsets from drone location using
    object's gps coordinate and drone's gps coordinate

    Useful for testing gps calculation accuracy
    """
    meters_per_deg_lat = 111111

    delta_lat = coord[0] - origin_coord[0]
    delta_long = coord[1] - origin_coord[1]

    # Convert differences to meters (approximation)
    delta_lat_meters = delta_lat * meters_per_deg_lat
    delta_long_meters = (
        delta_long
        * meters_per_deg_lat
        * math.cos(math.radians((origin_coord[0] + coord[0]) / 2))
    )

    return delta_long_meters, delta_lat_meters


def offset_from_drone(pixel, resolution, altitude, fov):
    resolution_width, resolution_height = resolution
    x, y = pixel

    fov_width, fov_height = fov, fov * (resolution_height / resolution_width)

    ground_height = 2 * (np.tan(np.radians(fov_height / 2)) * altitude)
    ground_width = 2 * (np.tan(np.radians(fov_width / 2)) * altitude)

    pixel_width = ground_width / resolution_width
    pixel_height = ground_height / resolution_height

    center_x = resolution_width / 2
    center_y = resolution_height / 2

    pixel_offset_x = x - center_x
    pixel_offset_y = center_y - y

    x_offset = pixel_offset_x * pixel_width
    y_offset = pixel_offset_y * pixel_height

    return x_offset, y_offset
