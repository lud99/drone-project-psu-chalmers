import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Coordinate:
    def __init__(self, lat, lng, alt=0):
        self.lat = lat
        self.lng = lng
        self.alt = alt


class HeightError(Exception):
    def __init__(self, height, message="Height exceeds Swedish regulations"):
        super().__init__(message + " " + str(height) + "m")


def calculate_altitude_old(area: float, aspect_ratio: float = 16 / 9) -> float:
    """Calculates the height that the drone needs to fly at to cover a certain 16:9 area."""
    theta = (82.6 / 2) * (np.pi / 180)
    x = np.sqrt(area / aspect_ratio)
    y = (16 * x) / 4
    radius = np.sqrt((2 * y) ** 2 + (1.5 * y) ** 2)
    height = radius / np.tan(theta)
    height = round(height)
    if height < 99:
        return height
    else:
        return 20
        # raise HeightError(height)


def calculate_altitude(width: float, height: float) -> float:
    # Diagonal of the area we want to see
    diagonal = np.sqrt(width**2 + height**2)
    # 82.6 is the diagonal Field of View (FOV)
    theta = (82.6 / 2) * (np.pi / 180)

    # Height = (Diagonal / 2) / tan(FOV / 2)
    alt = (diagonal / 2) / np.tan(theta)
    alt = round(alt)
    if alt < 99:
        return alt
    else:
        return 99


class ProximityError(Exception):
    def __init__(
        self, message="Does not take more than one drone and overlap over 90 percent"
    ):
        super().__init__(message)


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
    n_drones: int = 2,
    overlap: float = 0.5,
    aspect_ratio: float = 16 / 9,
) -> tuple[list[Coordinate], float]:
    """
    Calculates the drone coverage area and returns the coordinates for the drones to fly to.

    Args:
        corner_coords (dict): Dictionary of trajectory coordinates for each vehicle.
        drone_origin (Coordinate): The origin coordinate of the test.
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

    def find_optimal_rectangle(coords, aspect_ratio, max_altitude=99):
        """Find the rectangle orientation that minimizes coverage cost when altitude is constrained."""
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]

        # Calculate camera footprint at max altitude
        theta = (82.6 / 2) * (np.pi / 180)
        radius = max_altitude * np.tan(theta)
        norm_factor = np.sqrt(aspect_ratio**2 + 1)
        camera_width = 2 * radius * (aspect_ratio / norm_factor)
        camera_height = 2 * radius * (1 / norm_factor)

        best = {
            "cost": float("inf"),
            "center": None,
            "axis": None,
            "width": None,
            "height": None,
        }

        # Try orientations based on hull edges
        for i in range(len(hull_pts)):
            p0 = hull_pts[i]
            p1 = hull_pts[(i + 1) % len(hull_pts)]

            edge = p1 - p0
            U0 = edge / np.linalg.norm(edge)
            U1 = np.array([-U0[1], U0[0]])

            # Project points onto these axes
            proj0 = coords @ U0
            proj1 = coords @ U1

            min0, max0 = proj0.min(), proj0.max()
            min1, max1 = proj1.min(), proj1.max()

            rect_width = max0 - min0
            rect_height = max1 - min1

            # Calculate how many camera footprints we need in each direction
            # when altitude is capped
            n_width = max(1, np.ceil(rect_width / camera_width))
            n_height = max(1, np.ceil(rect_height / camera_height))

            # Total coverage area needed
            total_coverage_area = n_width * camera_width * n_height * camera_height

            # Also consider the aspect ratio constraint
            if rect_width / rect_height > aspect_ratio:
                final_width = rect_width
                final_height = final_width / aspect_ratio
            else:
                final_height = rect_height
                final_width = final_height * aspect_ratio

            # The "cost" is the total coverage area - lower is better
            cost = total_coverage_area

            if cost < best["cost"]:
                center = ((min0 + max0) / 2) * U0 + ((min1 + max1) / 2) * U1
                best.update(
                    {
                        "cost": cost,
                        "center": center,
                        "axis": [U0, U1],
                        "width": final_width,
                        "height": final_height,
                    }
                )

        return best

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

    # 1. Get the tightest bounding box (The Red Box)
    rect = min_area_rectangle_of_hull(compute_convex_hull(coords))

    # Check if we'll need to cap altitude
    curr_w = rect.extent[0] * 2
    curr_h = rect.extent[1] * 2
    test_altitude = calculate_altitude(curr_w, curr_h)

    # If altitude will be capped, find a better orientation
    if test_altitude >= 99:
        optimal_rect = find_optimal_rectangle(coords, aspect_ratio, max_altitude=99)
        if optimal_rect["cost"] < float("inf"):
            rect.center = optimal_rect["center"]
            rect.axis = optimal_rect["axis"]
            rect.extent = [optimal_rect["width"] / 2, optimal_rect["height"] / 2]

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

    # 4. Final half-extents for plotting
    half_w = width / 2
    half_h = height / 2

    if curr_w > curr_h:
        split_axis = axis[0]
        angle_axis = axis[1]
    else:
        split_axis = axis[1]
        angle_axis = axis[0]

    # if width > height:
    #     split_axis = axis[0]
    #     angle_axis = axis[1]
    # else:
    #     split_axis = axis[1]
    #     angle_axis = axis[0]

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

    # 4. Pad the required dimensions to fit the camera's 16:9 Aspect Ratio
    if req_w / req_h > aspect_ratio:
        cam_w = req_w
        cam_h = cam_w / aspect_ratio
    else:
        cam_h = req_h
        cam_w = cam_h * aspect_ratio

    half_w = cam_w / 2
    half_h = cam_h / 2

    # 5. Set the distance between drones based on the camera size
    if curr_w > curr_h:
        split_offset = cam_w * (1 - overlap)
    else:
        split_offset = cam_h * (1 - overlap)

    drone_centers = [
        center + (i - (n_drones - 1) / 2) * split_offset * split_axis
        for i in range(int(n_drones))
    ]

    altitude = calculate_altitude(cam_w, cam_h)

    if altitude < 30:
        altitude = 30  # Enforce minimum 30m flight limit

        theta = (82.6 / 2) * (np.pi / 180)

        # NOTE: Removed the '* 1.4' so the green box accurately represents
        # the camera footprint at 30 meters. Put it back if you want a safety buffer.
        safety_buffer = 1.0
        radius = altitude * np.tan(theta) * safety_buffer

        # Compute full dimensions from aspect ratio
        norm_factor = np.sqrt(aspect_ratio**2 + 1)

        width = 2 * radius * (aspect_ratio / norm_factor)
        height = 2 * radius * (1 / norm_factor)

        # Half extents (used everywhere else)
        half_w = cam_w / 2
        half_h = cam_h / 2

        # Correct spacing (full width)
        split_offset = width * (1 - overlap)

        drone_centers = [
            center + (i - (n_drones - 1) / 2) * split_offset * split_axis
            for i in range(n_drones)
        ]

    if altitude >= 99:  # Example max altitude
        altitude = 99

        # Calculate what the camera ACTUALLY sees at 120m
        theta = (82.6 / 2) * (np.pi / 180)
        radius = altitude * np.tan(theta)
        norm_factor = np.sqrt(aspect_ratio**2 + 1)

        # Shrink the plotting variables to match physical reality
        cam_w = 2 * radius * (aspect_ratio / norm_factor)
        cam_h = 2 * radius * (1 / norm_factor)
        half_w = cam_w / 2
        half_h = cam_h / 2

        # Recalculate centers so they don't have gaps between them
        if curr_w > curr_h:
            split_offset = cam_w * (1 - overlap)
        else:
            split_offset = cam_h * (1 - overlap)

        drone_centers = [
            center + (i - (n_drones - 1) / 2) * split_offset * split_axis
            for i in range(int(n_drones))
        ]

    fly_to_coords = []
    for drone_center in drone_centers:
        delta_lat = drone_center[1] / 6371000 * (180 / np.pi)
        delta_long = (
            drone_center[0] / (6371000 * np.cos(drone_origin.lat * np.pi / 180))
        ) * (180 / np.pi)

        # delta_lat = drone_center[0] / 6371000 * (180 / np.pi)
        # delta_long = (
        #     drone_center[1] / (6371000 * np.cos(drone_origin.lat * np.pi / 180))
        # ) * (180 / np.pi)
        lat = drone_origin.lat + delta_lat
        long = drone_origin.lng + delta_long

        fly_to_coords.append(Coordinate(lat, long, int(altitude)))

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], color="blue", label="Data Points")

    # 1. Convert the Fly-To objects back to local meters
    fly_to_local = np.array(
        [
            latlon_to_local(c.lat, c.lng, drone_origin.lat, drone_origin.lng)
            for c in fly_to_coords
        ]
    )

    # Loop through each drone to plot the altitude text
    for i, (coord, local_pt) in enumerate(zip(fly_to_coords, fly_to_local)):
        plt.text(
            local_pt[0],
            local_pt[1]
            + 2.5,  # Offset +1.5m on the y-axis so text doesn't overlap the 'X'
            f"{coord.alt}m",  # The altitude string
            color="blue",
            fontweight="bold",
            fontsize=10,
            ha="center",  # Center the text horizontally on the point
            bbox=dict(
                facecolor="white", alpha=0.6, edgecolor="none"
            ),  # Optional: white background for readability
        )

    # 3. Optional: Add text labels for drone numbers
    # for i, point in enumerate(fly_to_local):
    #     plt.text(point[0] + 1, point[1] + 1, f"Drone {i + 1}", color="red", fontsize=9)

    rect_corners = np.array(
        [
            rect.center + rect.extent[0] * rect.axis[0] + rect.extent[1] * rect.axis[1],
            rect.center + rect.extent[0] * rect.axis[0] - rect.extent[1] * rect.axis[1],
            rect.center - rect.extent[0] * rect.axis[0] - rect.extent[1] * rect.axis[1],
            rect.center - rect.extent[0] * rect.axis[0] + rect.extent[1] * rect.axis[1],
            rect.center + rect.extent[0] * rect.axis[0] + rect.extent[1] * rect.axis[1],
        ]
    )
    plt.plot(
        rect_corners[:, 0],
        rect_corners[:, 1],
        "r--",
        label="Min Area Bounding Box",
        zorder=10,
    )
    print(rect_corners)

    for drone_center in drone_centers:
        rect_corners = np.array(
            [
                drone_center + half_w * axis[0] + half_h * axis[1],
                drone_center + half_w * axis[0] - half_h * axis[1],
                drone_center - half_w * axis[0] - half_h * axis[1],
                drone_center - half_w * axis[0] + half_h * axis[1],
                drone_center + half_w * axis[0] + half_h * axis[1],
            ]
        )

        plt.plot(
            rect_corners[:, 0],
            rect_corners[:, 1],
            "g-",
            alpha=0.5,  # Add transparency
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
    plt.xlabel("x-coordinate relative to origin")
    plt.ylabel("y-coordinate relative to origin")
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


# ==========================================
# TEST CASES
# ==========================================


def run_test_case(name, points, n_drones, aspect_ratio=16 / 9, overlap=0.0):
    print(f"\n{'-' * 10} {name} {'-' * 10}")
    print(
        f"Drones: {n_drones} | Overlap: {overlap * 100}% | Aspect Ratio: {aspect_ratio}"
    )

    hull_points = [Coordinate(lat=p["lat"], lng=p["lon"]) for p in points]
    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)
    origin = Coordinate(lat=center_lat, lng=center_lon, alt=30)

    try:
        fly_to_coords, angle = get_drones_location(
            corner_coords=hull_points,
            drone_origin=origin,
            n_drones=n_drones,
            aspect_ratio=aspect_ratio,
            overlap=overlap,
        )
        for i, coord in enumerate(fly_to_coords):
            print(
                f"  Drone {i + 1} Target -> Lat: {coord.lat:.6f}, Lng: {coord.lng:.6f}, Alt: {coord.alt}m"
            )
        print(f"  Bounding Box Angle: {angle:.2f} degrees")
    except Exception as e:
        print(f"  [!] Error/Exception triggered: {type(e).__name__} - {e}")


# Fotbool field near campus
tc0_points = [
    {"lat": 57.68465596441495, "lon": 11.978838813919419},
    {"lat": 57.684988616504526, "lon": 11.979697539453221},
    {"lat": 57.68424874821773, "lon": 11.98067433974799},
    {"lat": 57.6839103538125, "lon": 11.979869284559996},
    {"lat": 57.68393329590591, "lon": 11.979890752698344},
]

run_test_case(
    "Test Case 0: Footbool field near chalmers",
    tc0_points,
    n_drones=1,
    aspect_ratio=16 / 9,
)


# Small triangle
tc1_5_points = [
    {"lat": 57.684844802309264, "lon": 11.97926434117621},
    {"lat": 57.68491936213738, "lon": 11.979628875286148},
    {"lat": 57.68478744849098, "lon": 11.979489494597036},
]

run_test_case(
    "Test Case 1.5: Small Triangle", tc1_5_points, n_drones=1, aspect_ratio=16 / 9
)


# ---------------------------------------------------------
# Test Case 1: Large Standard Area (1 Drone)
# Expected: A single drone flies high enough to capture a large roughly square area.
# ---------------------------------------------------------
tc1_points = [
    {"lat": 57.685, "lon": 11.970},
    {"lat": 57.685, "lon": 11.980},
    {"lat": 57.680, "lon": 11.980},
    {"lat": 57.680, "lon": 11.970},
]
run_test_case("Test Case 1: Large Square", tc1_points, n_drones=1, aspect_ratio=16 / 9)


# ---------------------------------------------------------
# Test Case 2: Long East-West Corridor (2 Drones)
# Expected: Two drones splitting a wide horizontal area (e.g. a road).
# The drone targets should be placed side-by-side horizontally.
# ---------------------------------------------------------
tc2_points = [
    {"lat": 57.684, "lon": 11.970},
    {"lat": 57.684, "lon": 11.990},  # Very wide longitude
    {"lat": 57.683, "lon": 11.990},
    {"lat": 57.683, "lon": 11.970},
]
run_test_case(
    "Test Case 2: Horizontal Corridor",
    tc2_points,
    n_drones=2,
    aspect_ratio=16 / 9,
    overlap=0.1,
)


# ---------------------------------------------------------
# Test Case 3: Long Diagonal Area (3 Drones)
# Expected: The algorithm should orient the bounding box diagonally
# and split the coverage into 3 distinct overlapping drone viewpoints.
# ---------------------------------------------------------
tc3_points = [
    {"lat": 57.680, "lon": 11.970},
    {"lat": 57.682, "lon": 11.974},
    {"lat": 57.684, "lon": 11.978},
    {"lat": 57.685, "lon": 11.980},
    {"lat": 57.684, "lon": 11.982},
    {"lat": 57.679, "lon": 11.972},
]
run_test_case(
    "Test Case 3: Diagonal Mapping",
    tc3_points,
    n_drones=3,
    aspect_ratio=16 / 9,
    overlap=0.2,
)


# ---------------------------------------------------------
# Test Case 4: Perfectly Collinear Points (2 Drones)
# Expected: Triggers the `are_colinear` fallback branch in your code.
# Fits a rectangle exactly along a straight line.
# ---------------------------------------------------------
tc4_points = [
    {"lat": 57.680, "lon": 11.970},
    {"lat": 57.681, "lon": 11.972},
    {"lat": 57.682, "lon": 11.974},
    {"lat": 57.683, "lon": 11.976},
]
run_test_case(
    "Test Case 4: Collinear Points",
    tc4_points,
    n_drones=2,
    aspect_ratio=1.0,
    overlap=0.0,
)


# ---------------------------------------------------------
# Test Case 5: Intentional Proximity Error (2 Drones)
# Expected: Triggers the custom `ProximityError` because the
# overlap is >= 0.9 with more than 1 drone.
# ---------------------------------------------------------
tc5_points = [
    {"lat": 57.685, "lon": 11.975},
    {"lat": 57.686, "lon": 11.976},
    {"lat": 57.685, "lon": 11.977},
]
run_test_case(
    "Test Case 5: Too Much Overlap",
    tc5_points,
    n_drones=2,
    aspect_ratio=16 / 9,
    overlap=0.95,
)
