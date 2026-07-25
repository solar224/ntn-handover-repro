from __future__ import annotations

import argparse
import datetime as dt
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import dataset_path, load_config, metadata_path
from .deps import require_numpy
from .geometry import (
    elevation_angle_deg,
    interpolate_route,
    synthetic_satellite_position,
    wrap_lon_deg,
)
from .utils import set_seed, sha256_text, write_json


ROUTES = [
    ((51.47, -0.45), (40.64, -73.78)),   # London Heathrow -> JFK
    ((35.55, 139.78), (37.62, -122.38)), # Tokyo Haneda -> SFO
    ((25.25, 55.36), (1.36, 103.99)),    # Dubai -> Singapore
    ((48.35, 11.79), (34.05, -118.24)),  # Munich -> Los Angeles
    ((-33.94, 151.18), (22.31, 113.92)), # Sydney -> Hong Kong
    ((-23.43, -46.47), (40.49, -3.57)),  # Sao Paulo -> Madrid
    ((30.19, -97.67), (47.45, -122.31)), # Austin -> Seattle
    ((-26.13, 28.24), (25.25, 55.36)),   # Johannesburg -> Dubai
]


def fetch_tle(url: str, timeout_s: int = 20) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError):
        return None


def parse_tle_records(raw: str, limit: int) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    records: list[tuple[str, str, str]] = []
    idx = 0
    while idx + 2 < len(lines) and len(records) < limit:
        if lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            records.append((lines[idx], lines[idx + 1], lines[idx + 2]))
            idx += 3
        elif lines[idx].startswith("1 ") and lines[idx + 1].startswith("2 "):
            records.append((f"SAT-{len(records):02d}", lines[idx], lines[idx + 1]))
            idx += 2
        else:
            idx += 1
    return records


def propagate_with_skyfield(records: list[tuple[str, str, str]], step_seconds: int, num_steps: int):
    try:
        from skyfield.api import EarthSatellite, load
    except Exception:
        return None

    np = require_numpy()
    ts = load.timescale()
    start = dt.datetime.now(dt.UTC).replace(microsecond=0)
    times = [start + dt.timedelta(seconds=i * step_seconds) for i in range(num_steps)]
    sf_times = ts.from_datetimes(times)
    sat_pos = np.zeros((num_steps, len(records), 3), dtype=np.float32)
    satellites = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in records]
    for n, sat in enumerate(satellites):
        subpoints = sat.at(sf_times).subpoint()
        sat_pos[:, n, 0] = subpoints.latitude.degrees
        sat_pos[:, n, 1] = subpoints.longitude.degrees
        sat_pos[:, n, 2] = subpoints.elevation.km
    return sat_pos


def generate_synthetic_satellites(cfg: dict[str, Any]):
    np = require_numpy()
    data_cfg = cfg["data"]
    num_steps = int(data_cfg["num_steps"])
    num_satellites = int(data_cfg["num_satellites"])
    step_seconds = int(data_cfg["step_seconds"])
    altitude = float(data_cfg["synthetic_satellite_altitude_km"])
    sat_pos = np.zeros((num_steps, num_satellites, 3), dtype=np.float32)
    for t in range(num_steps):
        elapsed = t * step_seconds
        for n in range(num_satellites):
            sat_pos[t, n, :] = synthetic_satellite_position(n, elapsed, num_satellites, altitude)
    return sat_pos


def generate_airplanes(cfg: dict[str, Any]):
    np = require_numpy()
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    data_cfg = cfg["data"]
    num_steps = int(data_cfg["num_steps"])
    num_airplanes = int(data_cfg["num_airplanes"])
    alt_base = float(data_cfg["airplane_altitude_km"])
    jitter = float(data_cfg["route_jitter_deg"])
    plane_pos = np.zeros((num_steps, num_airplanes, 3), dtype=np.float32)
    demand = np.zeros((num_steps, num_airplanes), dtype=np.float32)
    for k in range(num_airplanes):
        start, end = ROUTES[k % len(ROUTES)]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        demand_phase = rng.uniform(0.0, 2.0 * math.pi)
        for t in range(num_steps):
            cycle = ((t + k * 173) % num_steps) / max(1, num_steps - 1)
            fraction = 0.5 - 0.5 * math.cos(2.0 * math.pi * cycle)
            lat, lon = interpolate_route(start, end, fraction)
            lat += jitter * math.sin(4.0 * math.pi * fraction + phase)
            lon = wrap_lon_deg(lon + jitter * math.cos(3.0 * math.pi * fraction + phase))
            alt = alt_base + 0.4 * math.sin(2.0 * math.pi * cycle + phase)
            plane_pos[t, k, :] = (lat, lon, alt)
            demand_wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * cycle * 3.0 + demand_phase)
            demand_noise = 0.08 * rng.random()
            demand[t, k] = data_cfg["demand_min"] + (data_cfg["demand_max"] - data_cfg["demand_min"]) * min(
                1.0, max(0.0, 0.75 * demand_wave + demand_noise)
            )
    return plane_pos, demand


def compute_links(cfg: dict[str, Any], sat_pos, plane_pos):
    np = require_numpy()
    data_cfg = cfg["data"]
    num_steps, num_airplanes = plane_pos.shape[:2]
    num_satellites = sat_pos.shape[1]
    elevation = np.zeros((num_steps, num_airplanes, num_satellites), dtype=np.float32)
    for t in range(num_steps):
        for k in range(num_airplanes):
            plat, plon, palt = map(float, plane_pos[t, k])
            for n in range(num_satellites):
                slat, slon, salt = map(float, sat_pos[t, n])
                elevation[t, k, n] = elevation_angle_deg(plat, plon, palt, slat, slon, salt)
    coverage = elevation >= float(data_cfg["theta_min_deg"])
    return elevation, coverage


def generate_congestion(cfg: dict[str, Any]):
    np = require_numpy()
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 17)
    data_cfg = cfg["data"]
    num_steps = int(data_cfg["num_steps"])
    num_satellites = int(data_cfg["num_satellites"])
    lo = float(data_cfg["congestion_base_min"])
    hi = float(data_cfg["congestion_base_max"])
    congestion = np.zeros((num_steps, num_satellites), dtype=np.float32)
    for n in range(num_satellites):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        drift = rng.uniform(0.6, 1.4)
        for t in range(num_steps):
            wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / num_steps * drift + phase)
            congestion[t, n] = lo + (hi - lo) * (0.7 * wave + 0.3 * rng.random())
    return np.clip(congestion, 0.0, 0.95)


def build_dataset(config: str | Path, fast: bool = False, prefer_celestrak: bool | None = None) -> Path:
    overrides: dict[str, Any] = {}
    if fast:
        overrides = {"data": {"num_steps": 400, "num_airplanes": 3}}
    cfg = load_config(config, overrides)
    set_seed(int(cfg["project"]["seed"]))
    np = require_numpy()
    data_cfg = cfg["data"]
    if prefer_celestrak is None:
        prefer_celestrak = bool(data_cfg.get("prefer_celestrak", True))

    tle_raw = None
    tle_records: list[tuple[str, str, str]] = []
    sat_pos = None
    source = "synthetic"
    if prefer_celestrak:
        tle_raw = fetch_tle(str(data_cfg["celestrak_url"]))
        if tle_raw:
            tle_records = parse_tle_records(tle_raw, int(data_cfg["num_satellites"]))
            if len(tle_records) >= int(data_cfg["num_satellites"]):
                propagated = propagate_with_skyfield(
                    tle_records,
                    int(data_cfg["step_seconds"]),
                    int(data_cfg["num_steps"]),
                )
                if propagated is not None:
                    sat_pos = propagated
                    source = "celestrak_tle_skyfield"

    if sat_pos is None:
        sat_pos = generate_synthetic_satellites(cfg)
        if not tle_records:
            tle_records = [(f"SYNTH-{i:02d}", "", "") for i in range(int(data_cfg["num_satellites"]))]

    plane_pos, demand = generate_airplanes(cfg)
    elevation, coverage = compute_links(cfg, sat_pos, plane_pos)
    congestion = generate_congestion(cfg)
    timestamps_s = np.arange(int(data_cfg["num_steps"]), dtype=np.int32) * int(data_cfg["step_seconds"])

    out = dataset_path(cfg)
    np.savez_compressed(
        out,
        timestamps_s=timestamps_s,
        sat_pos=sat_pos.astype(np.float32),
        plane_pos=plane_pos.astype(np.float32),
        elevation=elevation.astype(np.float32),
        coverage=coverage.astype(np.bool_),
        congestion=congestion.astype(np.float32),
        demand=demand.astype(np.float32),
    )

    metadata = {
        "source": source,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config_path": str(config),
        "num_steps": int(data_cfg["num_steps"]),
        "step_seconds": int(data_cfg["step_seconds"]),
        "num_satellites": int(data_cfg["num_satellites"]),
        "num_airplanes": int(data_cfg["num_airplanes"]),
        "theta_min_deg": float(data_cfg["theta_min_deg"]),
        "demand_min": float(data_cfg["demand_min"]),
        "demand_max": float(data_cfg["demand_max"]),
        "tle_url": str(data_cfg["celestrak_url"]),
        "tle_sha256": sha256_text(tle_raw or ""),
        "satellites": [{"name": rec[0], "line1": rec[1], "line2": rec[2]} for rec in tle_records],
        "dataset_file": str(out),
        "dataset_sha256": sha256_text(str(out.stat().st_size) + str(out.stat().st_mtime_ns)),
        "fallback_note": "Synthetic constellation used when CelesTrak or Skyfield propagation is unavailable.",
    }
    write_json(metadata_path(cfg), metadata)
    return out


def load_dataset(cfg: dict[str, Any]):
    np = require_numpy()
    path = dataset_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python -m ntn_repro.data build")
    return np.load(path, allow_pickle=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the NTN handover dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Generate the synthetic/TLE-backed dataset.")
    build.add_argument("--config", default="configs/paper_result.yaml")
    build.add_argument("--fast", action="store_true", help="Generate a small smoke-test dataset.")
    build.add_argument("--synthetic-only", action="store_true", help="Skip CelesTrak and use deterministic synthetic satellites.")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        out = build_dataset(args.config, fast=args.fast, prefer_celestrak=not args.synthetic_only)
        print(f"Wrote dataset to {out}")


if __name__ == "__main__":
    main()

