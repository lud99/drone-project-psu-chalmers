import numpy as np
from scipy.spatial import ConvexHull

# Exception classes

MIN_ALTITUDE = 10
MAX_ALTITUDE = 80


class HeightError(Exception):
    def __init__(self, message="Height exceeds Swedish regulations"):
        super().__init__(message)


class ProximityError(Exception):
    def __init__(
        self, message="Does not take more than one drone and overlap over 90 percent"
    ):
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


def calculate_altitude(width: float, height: float, diagonal_fov: float) -> float:
    # Diagonal of the area we want to see
    diagonal = np.sqrt(width**2 + height**2)
    # diagonal_fov is the camera diagonal Field of View (FOV) in degrees
    theta = (diagonal_fov / 2) * (np.pi / 180)

    # Height = (Diagonal / 2) / tan(FOV / 2)
    alt = (diagonal / 2) / np.tan(theta)
    alt = round(alt)
    if alt < MIN_ALTITUDE:
        return MIN_ALTITUDE

    if alt < MAX_ALTITUDE:
        return alt
    else:
        return MAX_ALTITUDE


# ENU projection
def latlon_to_local(lat, lon, origin_lat, origin_lon):
    R = 6371000
    dlat = np.radians(lat - origin_lat)
    dlon = np.radians(lon - origin_lon)

    x = dlon * R * np.cos(np.radians(origin_lat))  # East
    y = dlat * R  # North
    return np.array([x, y])


def get_drones_location(
    corner_coords: list[Coordinate],
    drone_origin: Coordinate,
    diagonal_fov: float,
    n_drones: int = 2,
    overlap: float = 0.5,
    aspect_ratio: float = 16 / 9,
) -> tuple[
    list[Coordinate], float, list[tuple[float, float]], list[tuple[float, float]]
]:
    """
    Calculates the drone coverage area and returns the coordinates for the drones to fly to.

    Args:
        corner_coords (dict): Dictionary of trajectory coordinates for each vehicle.
        drone_origin (Coordinate): The origin coordinate of the test.
        diagonal_fov (float): Camera diagonal Field of View in degrees.
        n_drones (int): Number of drones to be used in the test.
        overlap (float): The overlap percentage between the drones.
        aspect_ratio (float): Camera aspect ratio (default 16/9).

    Returns:
        tuple: A tuple containing:
            - list[Coordinate]: coordinates for the drones to fly to
            - float: angle of the rectangle in degrees
            - list[list[Coordinate]]: coverage corners for each drone
            - list[Coordinate]: the minimum bounding rectangle corners for the original hull
    """

    # Overlap has to be between 0 and 1
    if not (0 <= overlap <= 1):
        raise ValueError("Overlap must be between 0 and 1 (inclusive).")

    # Proximity error if more than 2 drones and overlap is greater than 0.9
    if n_drones >= 2 and overlap >= 0.9:
        raise ProximityError()

    # Flatten the list of coordinates into an array
    coords = np.array(
        [
            latlon_to_local(c.lat, c.lng, drone_origin.lat, drone_origin.lng)
            for c in corner_coords
        ]
    )

    # Class to represent a rectangle
    class Rectangle:
        def __init__(self):
            self.center = np.array([0.0, 0.0])
            self.axis = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
            self.extent = [0.0, 0.0]
            self.area = float("inf")

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
        min_rect = Rectangle()
        n = len(polygon)

        for i0 in range(n):
            i1 = (i0 + 1) % n
            origin = polygon[i0]
            U0 = normalize(polygon[i1] - origin)
            U1 = perp(U0)

            min0, max0 = 0, 0
            min1, max1 = 0, 0

            for j in range(n):
                D = polygon[j] - origin
                dot0 = dot(U0, D)
                min0 = min(min0, dot0)
                max0 = max(max0, dot0)
                dot1 = dot(U1, D)
                min1 = min(min1, dot1)
                max1 = max(max1, dot1)
            area = (max0 - min0) * (max1 - min1)

            if area < min_rect.area:
                min_rect.center = (
                    origin + ((min0 + max0) / 2) * U0 + ((min1 + max1) / 2) * U1
                )
                min_rect.axis[0] = U0
                min_rect.axis[1] = U1
                min_rect.extent[0] = (max0 - min0) / 2
                min_rect.extent[1] = (max1 - min1) / 2
                min_rect.area = area
        return min_rect

    def compute_convex_hull(points: np.ndarray) -> list:
        """Computes the convex hull of a set of points."""
        hull = ConvexHull(points)
        return [points[i] for i in hull.vertices]

    def are_colinear(points: np.ndarray, tol: float = 1e-9) -> bool:
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

    # Check what altitude this would require
    curr_w = rect.extent[0] * 2
    curr_h = rect.extent[1] * 2

    # Scale to aspect ratio
    if curr_w / curr_h > aspect_ratio:
        scaled_w = curr_w
        scaled_h = scaled_w / aspect_ratio
    else:
        scaled_h = curr_h
        scaled_w = scaled_h * aspect_ratio

    # 2. Extract its current dimensions and orientation
    # rect.extent stores half-widths/heights
    curr_w = rect.extent[0] * 2
    curr_h = rect.extent[1] * 2
    axis = rect.axis  # This ensures Green rotation = Red rotation
    center = rect.center

    # 3. Scale dimensions to fit aspect ratio without changing rotation
    if curr_w / curr_h > aspect_ratio:
        # Too wide for the aspect ratio; expand height
        width = curr_w
        height = width / aspect_ratio
    else:
        # Too tall for the aspect ratio; expand width
        height = curr_h
        width = height * aspect_ratio

    if curr_w > curr_h:
        split_axis = axis[0]
        angle_axis = axis[1]
    else:
        split_axis = axis[1]
        angle_axis = axis[0]

    # 3. Figure out the physical dimensions each drone needs to cover
    coverage_factor = n_drones - overlap * (n_drones - 1)

    if curr_w > curr_h:
        split_axis = axis[0]
        angle_axis = axis[1]
        req_w = curr_w / coverage_factor
        req_h = curr_h
    else:
        split_axis = axis[1]
        angle_axis = axis[0]
        req_w = curr_w
        req_h = curr_h / coverage_factor

    # 4. Pad the required dimensions to fit the camera's Aspect Ratio
    if req_w / req_h > aspect_ratio:
        cam_w = req_w
        cam_h = cam_w / aspect_ratio
    else:
        cam_h = req_h
        cam_w = cam_h * aspect_ratio

    altitude = calculate_altitude(cam_w, cam_h, diagonal_fov)

    theta = (diagonal_fov / 2) * (np.pi / 180)

    safety_buffer = 1.0
    radius = altitude * np.tan(theta) * safety_buffer

    # Compute full dimensions from clamped altitude and camera aspect ratio
    norm_factor = np.sqrt(aspect_ratio**2 + 1)

    width = 2 * radius * (aspect_ratio / norm_factor)
    height = 2 * radius * (1 / norm_factor)

    # Half extents after altitude clamping
    half_w = width / 2
    half_h = height / 2

    # Recalculate centers so they don't have gaps between them
    if curr_w > curr_h:
        split_offset = cam_w * (1 - overlap)
    else:
        split_offset = cam_h * (1 - overlap)

    # Correct spacing (full width)
    split_offset = width * (1 - overlap)

    drone_centers = [
        center + (i - (n_drones - 1) / 2) * split_offset * split_axis
        for i in range(n_drones)
    ]

    fly_to_coords = []
    for drone_center in drone_centers:
        delta_lat = drone_center[1] / 6371000 * (180 / np.pi)
        delta_long = (
            drone_center[0] / (6371000 * np.cos(drone_origin.lat * np.pi / 180))
        ) * (180 / np.pi)

        lat = drone_origin.lat + delta_lat
        long = drone_origin.lng + delta_long

        fly_to_coords.append(Coordinate(lat, long, int(altitude)))

    coverage_corners = []
    for drone_center in drone_centers:
        rect_corners_local = np.array(
            [
                drone_center + half_w * axis[0] + half_h * axis[1],
                drone_center + half_w * axis[0] - half_h * axis[1],
                drone_center - half_w * axis[0] - half_h * axis[1],
                drone_center - half_w * axis[0] + half_h * axis[1],
            ]
        )

        corners_lat_lng = []
        for x, y in rect_corners_local:
            delta_lat = y / 6371000 * (180 / np.pi)
            delta_long = (x / (6371000 * np.cos(np.radians(drone_origin.lat)))) * (
                180 / np.pi
            )

            lat = drone_origin.lat + delta_lat
            lng = drone_origin.lng + delta_long

            corners_lat_lng.append((lat, lng))

        coverage_corners.append(corners_lat_lng)

    min_rect_corners_local = np.array(
        [
            rect.center + rect.extent[0] * rect.axis[0] + rect.extent[1] * rect.axis[1],
            rect.center + rect.extent[0] * rect.axis[0] - rect.extent[1] * rect.axis[1],
            rect.center - rect.extent[0] * rect.axis[0] - rect.extent[1] * rect.axis[1],
            rect.center - rect.extent[0] * rect.axis[0] + rect.extent[1] * rect.axis[1],
        ]
    )

    min_rect_corners = []
    for x, y in min_rect_corners_local:
        delta_lat = y / 6371000 * (180 / np.pi)
        delta_long = (x / (6371000 * np.cos(np.radians(drone_origin.lat)))) * (
            180 / np.pi
        )

        lat = drone_origin.lat + delta_lat
        lng = drone_origin.lng + delta_long
        min_rect_corners.append((lat, lng))

    angle_radians = np.arctan2(angle_axis[1], angle_axis[0])
    return (
        fly_to_coords,
        round(np.degrees(angle_radians)),
        coverage_corners,
        min_rect_corners,
    )
