import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import optimize


class Coordinate:
    def __init__(self, lat, lng, alt=0):
        self.lat = lat
        self.lng = lng
        self.alt = alt


def calculate_height(area):
    theta = (82.6 / 2) * (np.pi / 180)
    x = np.sqrt(area / (16 * 9))
    y = (16 * x) / 4
    radius = np.sqrt((2 * y) ** 2 + (1.5 * y) ** 2)
    height = radius / np.tan(theta)
    height = round(height)
    if height < 99:
        return height
    else:
        print("The height exceeds swedish regulations")
        return 99


class ProximityError(Exception):
    def __init__(
        self, message="Does not take more than one drone and overlap over 90 percent"
    ):
        super().__init__(message)


def get_drones_location(coordslist, drone_origin, n_drones=2, overlap=0.5):
    if not (0 <= overlap <= 1):
        raise ValueError("Overlap must be between 0 and 1 (inclusive).")
    if n_drones >= 2 and overlap >= 0.9:
        raise ProximityError("Overlap too high for multiple drones.")
    coords = []
    for coord_list in coordslist.values():
        for coord in coord_list:
            coords.append([coord.lng, coord.lat])
    coords = np.array(coords)

    class Rectangle:
        def __init__(self):
            self.center = np.array([0.0, 0.0])
            self.axis = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
            self.extent = [0.0, 0.0]
            self.area = float("inf")

    def normalize(v):
        return v / np.linalg.norm(v)

    def perp(v):
        return np.array([-v[1], v[0]])

    def dot(v1, v2):
        return np.dot(v1, v2)

    def min_area_rectangle_of_hull(polygon):
        # ruff: disable[N806]
        min_rect = Rectangle()
        n = len(polygon)
        for i0 in range(n):
            i1 = (i0 + 1) % n
            origin = polygon[i0]
            U0 = normalize(polygon[i1] - origin)
            U1 = perp(U0)
            min0, max0 = 0, 0
            max1 = 0
            for j in range(n):
                D = polygon[j] - origin
                dot0 = dot(U0, D)
                min0 = min(min0, dot0)
                max0 = max(max0, dot0)
                dot1 = dot(U1, D)
                max1 = max(max1, dot1)
            area = (max0 - min0) * max1
            if area < min_rect.area:
                min_rect.center = origin + ((min0 + max0) / 2) * U0 + (max1 / 2) * U1
                min_rect.axis[0] = U0
                min_rect.axis[1] = U1
                min_rect.extent[0] = (max0 - min0) / 2
                min_rect.extent[1] = max1 / 2
                min_rect.area = area
        return min_rect

    # ruff: enable[N806]

    def compute_convex_hull(points):
        hull = ConvexHull(points)
        return [points[i] for i in hull.vertices]

    def are_colinear(points, tol=1e-9):
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
        # ruff: disable[N806]
        U0 = normalize(direction)
        U1 = perp(U0)
        # ruff: enable[N806]
        extent_long = np.linalg.norm(direction)
        rect.extent[1] = extent_long / 2
        rect.extent[0] = extent_long
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

    diagonal = max(extent)
    width = (
        diagonal * (16 / 9) / np.sqrt((16 / 9) ** 2 + 1)
    )  # Initialize width based on aspect ratio
    aspect_ratio = 16 / 9
    norm_factor = np.sqrt(aspect_ratio**2 + 1)
    split_offset = width * (1 - overlap)
    height_r = diagonal * (1 / norm_factor)
    split_offset = width * (1 + (1 - 2 * overlap))
    print(n_drones)
    drone_centers = [
        center + (i - (n_drones - 1) / 2) * split_offset * split_axis
        for i in range(int(n_drones))
    ]
    height = calculate_height(width * height_r)

    if height < 30:
        height = 30
        height_r = optimize.root_scalar(
            lambda x: calculate_height(x) - height, x0=20, method="newton"
        ).root
    elif height == 99:
        height = 99
        height_r = optimize.root_scalar(
            lambda x: calculate_height(x) - height, x0=20, method="newton"
        ).root
        print("Cannot ensure full coverage with current drone amount")

    print(height)

    fly_to_coords = []
    for drone_center in drone_centers:
        delta_lat = drone_center[0] / 6371000 * (180 / np.pi)
        delta_long = (
            drone_center[1] / (6371000 * np.cos(drone_origin.lat * np.pi / 180))
        ) * (180 / np.pi)
        lat = drone_origin.lat + delta_lat
        long = drone_origin.lng + delta_long
        fly_to_coords.append(Coordinate(lat, long, height))

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], color="blue", label="Data Points")

    rect_corners = np.array(
        [
            center + extent[0] * axis[0] + extent[1] * axis[1],
            center + extent[0] * axis[0] - extent[1] * axis[1],
            center - extent[0] * axis[0] - extent[1] * axis[1],
            center - extent[0] * axis[0] + extent[1] * axis[1],
            center + extent[0] * axis[0] + extent[1] * axis[1],
        ]
    )
    plt.plot(rect_corners[:, 0], rect_corners[:, 1], "r-", label="Min Area Rectangle")

    for drone_center in drone_centers:
        rect_corners = np.array(
            [
                drone_center + width * axis[0] + height_r * axis[1],
                drone_center + width * axis[0] - height_r * axis[1],
                drone_center - width * axis[0] - height_r * axis[1],
                drone_center - width * axis[0] + height_r * axis[1],
                drone_center + width * axis[0] + height_r * axis[1],
            ]
        )
        plt.plot(
            rect_corners[:, 0],
            rect_corners[:, 1],
            "g-",
            label="Drone Coverage" if drone_center is drone_centers[0] else "",
        )

    plt.legend()
    plt.quiver(
        center[0],
        center[1],
        axis[1][0],
        axis[1][1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="cyan",
        label="Axis 1",
    )
    plt.xlabel("ATOS x-coordinate")
    plt.ylabel("ATOS y-coordinate")
    print(drone_centers)
    for drone_center in drone_centers:
        plt.scatter(
            drone_center[0],
            drone_center[1],
            color="orange",
            label="Drone Center" if drone_center is drone_centers[0] else "",
        )
    plt.title("Drone Coverage Area")
    plt.grid()
    plt.axis("equal")
    plt.show()

    angle = np.arctan2(angle_axis[1], angle_axis[0])
    return fly_to_coords, np.degrees(angle)


# Example usage
vehicle_trajectories = {
    "vehicle_1": [
        Coordinate(-30.4033, -21.2788, 0),
        Coordinate(-154.55, 14.9422, 0),
    ]
}
drone_origin = Coordinate(0, 0, 0)
flyto = get_drones_location(vehicle_trajectories, drone_origin, n_drones=2, overlap=0.5)
