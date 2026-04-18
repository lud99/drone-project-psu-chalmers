from typing import Any, Dict, List, Tuple, Optional
import math


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm to check if a point is inside a polygon.
    Polygon should be a list of (lat, lon) tuples, closed (first == last).
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class Geofence:
    def __init__(self, polygon: Optional[List[Tuple[float, float]]] = None, margin_m: float = 0.0):
        self.polygon = polygon
        self.margin_m = margin_m
        # Circular fence
        self.circle_center: Optional[Tuple[float, float]] = None  # (lat, lon)
        self.circle_radius_m: Optional[float] = None

    # ------------------------------------------------------------------ common

    def has_active_fence(self) -> bool:
        """Return True if at least one fence type is configured."""
        return self.polygon is not None or self.circle_center is not None

    def is_point_inside(self, lat: float, lon: float) -> bool:
        """Return True if the point satisfies ALL active keep-in fences."""
        if self.polygon is not None:
            if not point_in_polygon((lat, lon), self.polygon):
                return False
        if self.circle_center is not None and self.circle_radius_m is not None:
            dist = haversine_distance(
                lat, lon, self.circle_center[0], self.circle_center[1])
            if dist > self.circle_radius_m:
                return False
        return True

    # ------------------------------------------------------------------ polygon

    def validate_polygon(self, polygon: List[Tuple[float, float]]) -> bool:
        if len(polygon) < 4:  # min 3 + closing
            return False
        for lat, lon in polygon:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return False
        return True

    def set_polygon(self, polygon: List[Tuple[float, float]]) -> bool:
        if not self.validate_polygon(polygon):
            return False
        self.polygon = polygon
        return True

    def clear_polygon(self) -> None:
        self.polygon = None

    def get_polygon(self) -> Optional[List[Tuple[float, float]]]:
        return self.polygon

    # ------------------------------------------------------------------ circle

    def set_circle(self, lat: float, lon: float, radius_m: float) -> bool:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        if radius_m <= 0:
            return False
        self.circle_center = (lat, lon)
        self.circle_radius_m = radius_m
        return True

    def clear_circle(self) -> None:
        self.circle_center = None
        self.circle_radius_m = None

    def get_circle(self) -> Optional[Dict[str, Any]]:
        if self.circle_center is None or self.circle_radius_m is None:
            return None
        return {
            "latitude": self.circle_center[0],
            "longitude": self.circle_center[1],
            "radius_m": self.circle_radius_m,
        }
