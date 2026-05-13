from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from communication_software.convex_hull_scalable import Coordinate, get_drones_location


# Use a non-interactive backend so this test works in headless CI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 20260506
N_RANDOM_TESTS = 100
RESULTS_DIR = Path("convex_hull_test_results")


def _offset_meters_to_lat_lng(
    origin: Coordinate, east_m: float, north_m: float
) -> tuple[float, float]:
    earth_radius = 6371000.0
    delta_lat = (north_m / earth_radius) * (180.0 / np.pi)
    delta_lng = (east_m / (earth_radius * np.cos(np.radians(origin.lat)))) * (
        180.0 / np.pi
    )
    return origin.lat + delta_lat, origin.lng + delta_lng


def _generate_points(
    rng: np.random.Generator, origin: Coordinate, n_points: int
) -> list[Coordinate]:
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_points))
    radii = rng.uniform(5.0, 30.0, size=n_points)

    points: list[Coordinate] = []
    for angle, radius in zip(angles, radii):
        east = float(radius * np.cos(angle))
        north = float(radius * np.sin(angle))
        lat, lng = _offset_meters_to_lat_lng(origin, east, north)
        points.append(Coordinate(lat, lng))
    return points


def _latlon_to_meters(
    lat: float, lng: float, origin: Coordinate
) -> tuple[float, float]:
    """Convert a lat/lng coordinate to local ENU metres relative to origin."""
    earth_radius = 6371000.0
    east = np.radians(lng - origin.lng) * earth_radius * np.cos(np.radians(origin.lat))
    north = np.radians(lat - origin.lat) * earth_radius
    return float(east), float(north)


def _plot_case(
    idx: int,
    points: list[Coordinate],
    origin: Coordinate,
    min_rect_corners: list[tuple[float, float]],
    coverage_corners: list[list[tuple[float, float]]],
    diagonal_fov: float,
    aspect_ratio: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Input points (already generated in ENU space, convert back for display) ──
    pts_m = np.array([_latlon_to_meters(p.lat, p.lng, origin) for p in points])
    ax.scatter(
        pts_m[:, 0], pts_m[:, 1], color="black", s=30, label="Input points", zorder=3
    )

    # ── OBB ──
    obb_m = np.array(
        [_latlon_to_meters(lat, lng, origin) for lat, lng in min_rect_corners]
        + [_latlon_to_meters(min_rect_corners[0][0], min_rect_corners[0][1], origin)]
    )
    ax.plot(
        obb_m[:, 0],
        obb_m[:, 1],
        linestyle="--",
        linewidth=2,
        color="tab:red",
        label="Calculated OBB",
        zorder=2,
    )

    # ── Coverage footprint per drone ──
    cmap = plt.cm.get_cmap("tab10", max(1, len(coverage_corners)))
    for drone_idx, corners in enumerate(coverage_corners):
        poly_m = np.array(
            [_latlon_to_meters(lat, lng, origin) for lat, lng in corners]
            + [_latlon_to_meters(corners[0][0], corners[0][1], origin)]
        )
        color = cmap(drone_idx)
        ax.fill(
            poly_m[:, 0],
            poly_m[:, 1],
            alpha=0.18,
            color=color,
            label=f"Drone {drone_idx + 1} coverage",
            zorder=1,
        )
        ax.plot(poly_m[:, 0], poly_m[:, 1], color=color, linewidth=1.5, zorder=1)

    params_text = (
        f"fov={diagonal_fov:.1f} deg\n"
        f"aspect_ratio={aspect_ratio:.3f}\n"
        f"n_points={len(points)}\n"
        "n_drones=1"
    )

    ax.set_title(f"Convex Hull Visual Test {idx + 1}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.98,
        params_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"random_case_{idx + 1}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def test_generate_reproducible_random_visual_cases() -> None:
    rng = np.random.default_rng(RNG_SEED)
    origin = Coordinate(lat=57.70887, lng=11.97456)

    generated_files: list[Path] = []
    for idx in range(N_RANDOM_TESTS):
        n_points = int(rng.integers(2, 10))
        diagonal_fov = float(rng.uniform(65.0, 115.0))
        aspect_ratio = float(rng.uniform(1.1, 2.3))
        n_drones = 1
        overlap = 0.0

        points = _generate_points(rng=rng, origin=origin, n_points=n_points)
        _fly_to, _angle, coverage_corners, min_rect_corners = get_drones_location(
            corner_coords=points,
            drone_origin=origin,
            diagonal_fov=diagonal_fov,
            n_drones=n_drones,
            overlap=overlap,
            aspect_ratio=aspect_ratio,
        )

        out_path = _plot_case(
            idx=idx,
            points=points,
            origin=origin,
            min_rect_corners=min_rect_corners,
            coverage_corners=coverage_corners,
            diagonal_fov=diagonal_fov,
            aspect_ratio=aspect_ratio,
        )
        generated_files.append(out_path)

    assert len(generated_files) == N_RANDOM_TESTS
    for path in generated_files:
        assert path.exists(), f"Expected plot output file does not exist: {path}"
        assert path.stat().st_size > 0, f"Expected non-empty plot file: {path}"
