import numpy as np
import math

def pixelToGps(pixel, cameraLocation, altitude,
               orientation=0, fov=83.0, resolution=(1920, 1080)):
    """
    Converts a pixel position to GPS coordinates, based on camera location, altitude, and FoV.
    """
    EARTH_RADIUS = 6378137  # in meters
    lat, lon = cameraLocation
    width, height = resolution
    x, y = pixel

    # Calculate the field of view in vertical direction based on image aspect ratio
    fovWidth, fovHeight = fov, fov * (height / width)

    # Calculate the ground distance covered by the image
    groundHeight = 2 * (np.tan(np.radians(fovHeight / 2)) * altitude)
    groundWidth = 2 * (np.tan(np.radians(fovWidth / 2)) * altitude)

    # Calculate the ground distance covered by each pixel
    pixelWidth = groundWidth / width
    pixelHeight = groundHeight / height

    # Calculate offsets for the given pixel from the image center
    xOffset = pixelWidth * (x - width / 2)
    yOffset = pixelHeight * (height / 2 - y)  # Invert Y axis

    # Rotate the offsets by the orientation
    orientationRad = np.radians(orientation)
    xOffsetRotated = xOffset * np.cos(orientationRad) + yOffset * np.sin(orientationRad)
    yOffsetRotated = -xOffset * np.sin(orientationRad) + yOffset * np.cos(orientationRad)

    # Calculate new coordinates
    deltaLat = (yOffsetRotated / EARTH_RADIUS) * (180 / np.pi)
    deltaLon = (xOffsetRotated / (EARTH_RADIUS * np.cos(np.radians(lat)))) * (180 / np.pi)

    return lat + deltaLat, lon + deltaLon

def gpsDeltaToMeters(originCoord, coord):
    '''
    Calculates object's (x,y) meter offsets from drone location using
    object's gps coordinate and drone's gps coordinate

    Useful for testing gps calculation accuracy
    '''
    metersPerDegLat = 111111

    delatLat = coord[0] - originCoord[0]
    deltaLon = coord[1] - originCoord[1]

    # Convert differences to meters (approximation)
    deltaLatMeters = delatLat * metersPerDegLat
    deltaLonMeters = deltaLon * metersPerDegLat * math.cos(math.radians((originCoord[0] + coord[0]) / 2))

    return deltaLonMeters, deltaLatMeters

def offsetFromDrone(pixel, resolution, altitude, fov):
    resolutionWidth, resolutionHeight = resolution
    x, y = pixel

    fovWidth, fovHeight = fov, fov * (resolutionHeight / resolutionWidth)

    groundHeight = 2 * (np.tan(np.radians(fovHeight / 2)) * altitude)
    groundWidth = 2 * (np.tan(np.radians(fovWidth / 2)) * altitude)

    pixelWidth = groundWidth / resolutionWidth
    pixelHeight = groundHeight / resolutionHeight

    centerX = resolutionWidth / 2
    centerY = resolutionHeight / 2

    pixelOffsetX = x - centerX
    pixelOffsetY = centerY - y

    xOffset = pixelOffsetX * pixelWidth
    yOffset = pixelOffsetY * pixelHeight

    return xOffset, yOffset