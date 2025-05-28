import numpy as np
from scipy.spatial import ConvexHull

# Exception classes

class HeightError(Exception):
    def __init__(self, message="Height exceeds Swedish regulations"):
        super().__init__(message)

class ProximityError(Exception):
    def __init__(self, message="Does not take more than one drone and overlap over 90 percent"):
        super().__init__(message)

class Coordinate:
    def __init__(self, lat, lng, alt=0):
        self.lat = lat
        self.lng = lng
        self.alt = alt

    def __str__(self): 
        return f"Coordinate(lat={self.lat}, lng={self.lng}, alt={self.alt})"

    def __repr__(self): 
        return f"Coordinate(lat={self.lat}, lng={self.lng}, alt={self.alt})"

# Main function to calculate drone locations

def calculate_Height(area: float) -> float:
    """Calculates the height that the drone needs to fly at to cover a certain 16:9 area."""
    theta = (82.6/2)*(np.pi/180)
    x = np.sqrt(area/(16*9))
    y = (16*x)/4
    radius = np.sqrt((2*y)**2+(1.5*y)**2)
    height = radius / np.tan(theta)
    height = round(height)
    if height < 99:
        return height
    else:
        raise HeightError()

def getDronesLoc(
        coordslist: dict[str, list[Coordinate]], 
        droneOrigin: Coordinate, 
        n_drones: int=2, 
        overlap: float=0.5
        ) -> tuple[list[Coordinate], float]:
    """
    Calculates the drone coverage area and returns the coordinates for the drones to fly to.
    
    Args:
        coordslist (dict): Dictionary of trajectory coordinates for each vehicle.
        droneOrigin (Coordinate): The origin coordinate of the test.
        n_drones (int): Number of drones to be used in the test.
        overlap (float): The overlap percentage between the drones.
        
    Returns:
        tuple: A tuple containing a list of coordinates for the drones to fly to and the angle of the rectangle.
    """

    # Overlap has to be between 0 and 1
    if not (0 <= overlap <= 1):
        raise ValueError("Overlap must be between 0 and 1 (inclusive).")
    
    # Proximity error if more than 2 drones and overlap is greater than 0.9
    if n_drones >= 2 and overlap >= 0.9:
        raise ProximityError()
    coords = []

    # Flatten the list of coordinates into an array
    for coordList in coordslist.values():
        for coord in coordList:
            coords.append([coord.lng, coord.lat])
    coords = np.array(coords)

    # Class to represent a rectangle
    class Rectangle:
        def __init__(self):
            self.center = np.array([0.0, 0.0])
            self.axis = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
            self.extent = [0.0, 0.0]
            self.area = float('inf')

    # Helper functions
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalizes a vector."""
        if np.linalg.norm(v) == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return v / np.linalg.norm(v)

    def perp(v: np.ndarray) -> np.ndarray:
        """Returns a perpendicular vector."""
        if v.shape != (2,):
            raise ValueError("Input vector must be 2D.")
        return np.array([-v[1], v[0]])

    def dot(v1: np.ndarray, v2: np.ndarray) -> float:
        """Returns the dot product of two vectors."""
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same shape.")
        return np.dot(v1, v2)

    def min_area_rectangle_of_hull(polygon: list) -> Rectangle:
        """Computes the oriented bounding box that encloses the convex hull of the trajectory points."""
        # Initialize the minimum rectangle
        min_rect = Rectangle()
        n = len(polygon)

        # Iterate through each edge of the polygon
        for i0 in range(n):

            # Get the next vertex in the polygon
            i1 = (i0 + 1) % n

            # Calculate the origin and the two axes of the rectangle
            origin = polygon[i0]
            U0 = normalize(polygon[i1] - origin)
            U1 = perp(U0)

            # Initialize min and max values for the rectangle
            min0, max0 = 0, 0
            max1 = 0

            # Project all points onto the axes and find the min/max
            for j in range(n):
                D = polygon[j] - origin
                dot0 = dot(U0, D)
                min0 = min(min0, dot0)
                max0 = max(max0, dot0)
                dot1 = dot(U1, D)
                max1 = max(max1, dot1)
            area = (max0 - min0) * max1

            # Update the minimum rectangle if the area is smaller
            if area < min_rect.area:
                min_rect.center = origin + ((min0 + max0) / 2) * U0 + (max1 / 2) * U1
                min_rect.axis[0] = U0
                min_rect.axis[1] = U1
                min_rect.extent[0] = (max0 - min0) / 2
                min_rect.extent[1] = max1 / 2
                min_rect.area = area
        return min_rect

    def compute_convex_hull(points: np.ndarray) -> list:
        """Computes the convex hull of a set of points."""
        hull = ConvexHull(points)
        return [points[i] for i in hull.vertices]

    def are_colinear(points: np.ndarray, tol: float=1e-9) -> bool:
        """Checks if a set of points are collinear."""
        if len(points) < 3:
            return True
        x0, y0 = points[0]
        x1, y1 = points[1]
        for x, y in points[2:]:
            cp = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
            if abs(cp) > tol:
                return False
        return True

    if are_colinear(coords):
        rect = Rectangle()
        rect.center = np.mean(coords, axis=0)
        sorted_coords = sorted(coords, key=lambda p: p[0])
        start_coord = sorted_coords[0]
        end_coord = sorted_coords[-1]
        direction = end_coord - rect.center
        U0 = normalize(direction)
        U1 = perp(U0)
        extent_long = np.linalg.norm(direction)
        rect.extent[1] = float(extent_long / 2)
        rect.extent[0] = float(extent_long)
        rect.axis[0] = U0
        rect.axis[1] = U1
        rect.area = 4 * rect.extent[0] * rect.extent[1]
    else:
        rect = min_area_rectangle_of_hull(compute_convex_hull(coords))

    axis = np.array(rect.axis)
    center = np.array(rect.center)
    extent = np.array(rect.extent)

    if extent[0] > extent[1]:
        split_axis = axis[0]
        angle_axis = axis[1]
    else:
        split_axis = axis[1]
        angle_axis = axis[0]

    # Calculate the total area of the rectangle
    total_area = 4 * extent[0] * extent[1]

    # Calculate the dimensions of the 16:9 drone squares
    aspect_ratio = 16 / 9
    height_r = np.sqrt(total_area / (n_drones * aspect_ratio))
    width = height_r * aspect_ratio

    # Adjust the split offset to ensure the drone squares cover the entire rectangle
    split_offset = width * (1 + (1 - 2 * overlap))

    drone_centers = [center + (i - (n_drones - 1) / 2) * split_offset * split_axis for i in range(int(n_drones))]

    height = calculate_Height(width * height_r)

    if height < 30:

        height = 30

        theta = (82.6 / 2) * (np.pi / 180)
        radius = (height * np.tan(theta))*1.4

        aspect_ratio = 16 / 9   
        norm_factor = np.sqrt(aspect_ratio**2 + 1)

        width = radius * (aspect_ratio / norm_factor)
        height_r = radius * (1 / norm_factor)

        split_offset = 2 * width * (1 - overlap)
        drone_centers = [center + (i - (n_drones - 1) / 2) * split_offset * split_axis for i in range(n_drones)]

    flyTo_coords = []
    for drone_center in drone_centers:
        delta_lat = drone_center[0] / 6371000 * (180 / np.pi)
        delta_long = (drone_center[1] / (6371000 * np.cos(droneOrigin.lat * np.pi / 180))) * (180 / np.pi)
        lat = droneOrigin.lat + delta_lat
        long = droneOrigin.lng + delta_long
        flyTo_coords.append(Coordinate(lat, long, int(height)))

    angle = round(np.arctan2(angle_axis[1], angle_axis[0]))
    return flyTo_coords, np.degrees(angle)
