from __future__ import annotations

import argparse
import datetime as dt
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import dataset_path, load_config, metadata_path, project_root
from .deps import require_numpy
from .geometry import (
    EARTH_RADIUS_KM,
    elevation_angle_deg,
    interpolate_route,
    synthetic_satellite_position,
    wrap_lon_deg,
)
from .utils import read_json, set_seed, sha256_text, write_json


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


class TLEFetchError(RuntimeError):
    pass


def fetch_tle(url: str, timeout_s: int = 30, retries: int = 3) -> str:
    """Download CelesTrak TLE text with an identifiable HTTP client.

    CelesTrak rejects Python's default urllib user agent with HTTP 403. Use a
    descriptive user agent, retry transient failures, and preserve the final
    error instead of silently switching data sources.
    """

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ntn-handover-repro/0.1 (+https://celestrak.org/)",
            "Accept": "text/plain",
        },
    )
    last_error: Exception | None = None
    attempts_made = 0
    for attempt in range(1, max(1, retries) + 1):
        attempts_made = attempt
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
            if not raw.strip():
                raise TLEFetchError("CelesTrak returned an empty response.")
            return raw
        except urllib.error.HTTPError as exc:
            last_error = exc
            # CelesTrak uses HTTP 403 for its download-frequency block.
            # Retrying immediately only extends the abuse pattern.
            if exc.code == 403:
                break
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            ValueError,
            TLEFetchError,
        ) as exc:
            last_error = exc
            if attempt < max(1, retries):
                time.sleep(0.5 * attempt)
    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
    raise TLEFetchError(f"Failed to download CelesTrak TLE after {attempts_made} attempt(s) ({detail}).")


def parse_tle_records(raw: str, limit: int | None = None) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    records: list[tuple[str, str, str]] = []
    idx = 0
    while idx + 2 < len(lines) and (limit is None or len(records) < limit):
        if lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            records.append((lines[idx], lines[idx + 1], lines[idx + 2]))
            idx += 3
        elif lines[idx].startswith("1 ") and lines[idx + 1].startswith("2 "):
            records.append((f"SAT-{len(records):02d}", lines[idx], lines[idx + 1]))
            idx += 2
        else:
            idx += 1
    return records


def tle_catalog_id(record: tuple[str, str, str]) -> int:
    return int(record[1][2:7])


def tle_altitude_km(record: tuple[str, str, str]) -> float:
    """Approximate semi-major-axis altitude from the TLE mean motion."""

    mean_motion_rev_day = float(record[2][52:63])
    angular_rate = mean_motion_rev_day * 2.0 * math.pi / 86400.0
    orbital_radius = (398600.4418 / (angular_rate * angular_rate)) ** (1.0 / 3.0)
    return orbital_radius - EARTH_RADIUS_KM


def filter_tle_records_by_altitude(
    records: list[tuple[str, str, str]],
    minimum_km: float,
    maximum_km: float,
) -> list[tuple[str, str, str]]:
    """Keep a reproducible operational-shell proxy and reject lowering orbits."""

    filtered = [
        record
        for record in records
        if minimum_km <= tle_altitude_km(record) <= maximum_km
    ]
    return sorted(filtered, key=tle_catalog_id)


def configured_propagation_start(data_cfg: dict[str, Any]) -> dt.datetime:
    raw = data_cfg.get("propagation_start_utc")
    if not raw:
        return dt.datetime.now(dt.UTC).replace(microsecond=0)
    value = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("data.propagation_start_utc must include a UTC offset.")
    return value.astimezone(dt.UTC).replace(microsecond=0)


def propagate_with_skyfield(
    records: list[tuple[str, str, str]],
    step_seconds: int,
    num_steps: int,
    start: dt.datetime | None = None,
):
    try:
        from skyfield.api import EarthSatellite, load
    except Exception as exc:
        raise RuntimeError(
            "Skyfield TLE propagation is unavailable. Install requirements.txt in the active environment."
        ) from exc

    np = require_numpy()
    ts = load.timescale()
    start = start or dt.datetime.now(dt.UTC).replace(microsecond=0)
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


def compute_elevation_matrix(plane_pos, sat_pos):
    """Vectorized elevation for [time, airplane, 3] and [time, satellite, 3]."""

    np = require_numpy()
    plane_lat = np.radians(plane_pos[:, :, 0])
    plane_lon = np.radians(plane_pos[:, :, 1])
    plane_radius = EARTH_RADIUS_KM + plane_pos[:, :, 2]
    plane_cos_lat = np.cos(plane_lat)
    observer = np.stack(
        [
            plane_radius * plane_cos_lat * np.cos(plane_lon),
            plane_radius * plane_cos_lat * np.sin(plane_lon),
            plane_radius * np.sin(plane_lat),
        ],
        axis=-1,
    )

    sat_lat = np.radians(sat_pos[:, :, 0])
    sat_lon = np.radians(sat_pos[:, :, 1])
    sat_radius = EARTH_RADIUS_KM + sat_pos[:, :, 2]
    sat_cos_lat = np.cos(sat_lat)
    satellite = np.stack(
        [
            sat_radius * sat_cos_lat * np.cos(sat_lon),
            sat_radius * sat_cos_lat * np.sin(sat_lon),
            sat_radius * np.sin(sat_lat),
        ],
        axis=-1,
    )

    relative = satellite[:, None, :, :] - observer[:, :, None, :]
    relative_norm = np.linalg.norm(relative, axis=-1)
    up = observer / np.linalg.norm(observer, axis=-1, keepdims=True)
    sin_elevation = np.sum(relative * up[:, :, None, :], axis=-1) / relative_norm
    return np.degrees(np.arcsin(np.clip(sin_elevation, -1.0, 1.0))).astype(np.float32)


def generate_dynamic_satellite_candidates(
    cfg: dict[str, Any],
    records: list[tuple[str, str, str]],
    plane_pos,
):
    """Select the highest-elevation candidate slots from a larger TLE pool.

    Candidate slots may change over time, so the returned NORAD IDs are the
    persistent satellite identity used by the handover environment.
    """

    try:
        from skyfield.api import EarthSatellite, load
    except Exception as exc:
        raise RuntimeError(
            "Skyfield TLE propagation is unavailable. Install requirements.txt in the active environment."
        ) from exc

    np = require_numpy()
    data_cfg = cfg["data"]
    num_steps, num_airplanes = plane_pos.shape[:2]
    num_candidates = int(data_cfg["num_satellites"])
    chunk_size = max(1, int(data_cfg.get("candidate_chunk_size", 32)))
    step_seconds = int(data_cfg["step_seconds"])

    ts = load.timescale()
    start = configured_propagation_start(data_cfg)
    datetimes = [start + dt.timedelta(seconds=i * step_seconds) for i in range(num_steps)]
    sf_times = ts.from_datetimes(datetimes)

    best_elevation = np.full(
        (num_steps, num_airplanes, num_candidates),
        -90.0,
        dtype=np.float32,
    )
    best_position = np.zeros(
        (num_steps, num_airplanes, num_candidates, 3),
        dtype=np.float32,
    )
    best_ids = np.full(
        (num_steps, num_airplanes, num_candidates),
        -1,
        dtype=np.int32,
    )

    for start_index in range(0, len(records), chunk_size):
        chunk = records[start_index : start_index + chunk_size]
        chunk_pos = np.zeros((num_steps, len(chunk), 3), dtype=np.float32)
        chunk_ids = np.asarray([tle_catalog_id(record) for record in chunk], dtype=np.int32)
        for chunk_index, (name, line1, line2) in enumerate(chunk):
            subpoints = EarthSatellite(line1, line2, name, ts).at(sf_times).subpoint()
            chunk_pos[:, chunk_index, 0] = subpoints.latitude.degrees
            chunk_pos[:, chunk_index, 1] = subpoints.longitude.degrees
            chunk_pos[:, chunk_index, 2] = subpoints.elevation.km

        chunk_elevation = compute_elevation_matrix(plane_pos, chunk_pos)
        expanded_position = np.broadcast_to(
            chunk_pos[:, None, :, :],
            (num_steps, num_airplanes, len(chunk), 3),
        )
        expanded_ids = np.broadcast_to(
            chunk_ids[None, None, :],
            (num_steps, num_airplanes, len(chunk)),
        )

        merged_elevation = np.concatenate([best_elevation, chunk_elevation], axis=2)
        merged_position = np.concatenate([best_position, expanded_position], axis=2)
        merged_ids = np.concatenate([best_ids, expanded_ids], axis=2)
        top_indices = np.argpartition(
            merged_elevation,
            -num_candidates,
            axis=2,
        )[:, :, -num_candidates:]
        selected_elevation = np.take_along_axis(merged_elevation, top_indices, axis=2)
        selected_position = np.take_along_axis(
            merged_position,
            top_indices[:, :, :, None],
            axis=2,
        )
        selected_ids = np.take_along_axis(merged_ids, top_indices, axis=2)
        order = np.argsort(-selected_elevation, axis=2)
        best_elevation = np.take_along_axis(selected_elevation, order, axis=2)
        best_position = np.take_along_axis(
            selected_position,
            order[:, :, :, None],
            axis=2,
        )
        best_ids = np.take_along_axis(selected_ids, order, axis=2)

    coverage = best_elevation >= float(data_cfg["theta_min_deg"])
    return best_position, best_elevation, coverage, best_ids, start


def generate_dynamic_congestion(cfg: dict[str, Any], sat_ids):
    """Deterministic congestion keyed by timestep and persistent NORAD ID."""

    np = require_numpy()
    data_cfg = cfg["data"]
    num_steps = sat_ids.shape[0]
    lo = float(data_cfg["congestion_base_min"])
    hi = float(data_cfg["congestion_base_max"])
    valid_ids = np.maximum(sat_ids, 0).astype(np.float64)
    phase = np.mod(valid_ids * 0.6180339887498949, 1.0) * 2.0 * math.pi
    drift = 0.6 + 0.8 * np.mod(valid_ids * 0.4142135623730950, 1.0)
    timestep = np.arange(num_steps, dtype=np.float64)[:, None, None]
    wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * timestep / num_steps * drift + phase)
    noise = np.mod(
        np.sin(valid_ids * 12.9898 + timestep * 78.233) * 43758.5453,
        1.0,
    )
    congestion = lo + (hi - lo) * (0.7 * wave + 0.3 * noise)
    congestion[sat_ids < 0] = 1.0
    return np.clip(congestion, 0.0, 0.95).astype(np.float32)


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


def build_dataset(
    config: str | Path,
    fast: bool = False,
    prefer_celestrak: bool | None = None,
    allow_synthetic_fallback: bool | None = None,
) -> Path:
    overrides: dict[str, Any] = {}
    if fast:
        overrides = {"data": {"num_steps": 400, "num_airplanes": 3}}
    cfg = load_config(config, overrides)
    set_seed(int(cfg["project"]["seed"]))
    np = require_numpy()
    data_cfg = cfg["data"]
    if prefer_celestrak is None:
        prefer_celestrak = bool(data_cfg.get("prefer_celestrak", True))
    if allow_synthetic_fallback is None:
        allow_synthetic_fallback = bool(data_cfg.get("allow_synthetic_fallback", True))
    selection_mode = str(data_cfg.get("satellite_selection_mode", "fixed_first_n"))
    dynamic_candidates = prefer_celestrak and selection_mode == "dynamic_top_elevation"
    record_limit = None if dynamic_candidates else int(data_cfg["num_satellites"])

    tle_raw = None
    tle_records: list[tuple[str, str, str]] = []
    candidate_pool_records: list[tuple[str, str, str]] = []
    sat_pos = None
    sat_ids = None
    elevation = None
    coverage = None
    congestion = None
    propagation_start: dt.datetime | None = None
    source = "synthetic"
    tle_error: str | None = None
    tle_cache = dataset_path(cfg).parent / str(data_cfg.get("tle_cache_file", "celestrak_starlink.tle"))
    snapshot_value = data_cfg.get("tle_snapshot_file")
    tle_snapshot = project_root(cfg) / str(snapshot_value) if snapshot_value else None
    refresh_seconds = int(data_cfg.get("tle_refresh_seconds", 7200))
    if prefer_celestrak:
        cache_is_fresh = (
            tle_cache.exists()
            and time.time() - tle_cache.stat().st_mtime <= refresh_seconds
        )
        if cache_is_fresh:
            tle_raw = tle_cache.read_text(encoding="utf-8")
            tle_records = parse_tle_records(tle_raw, record_limit)
            if len(tle_records) >= int(data_cfg["num_satellites"]):
                source = "celestrak_tle_cache_skyfield"
                print("Using recent CelesTrak TLE cache; live download skipped.", flush=True)
            else:
                tle_records = []
        if not tle_records:
            try:
                tle_raw = fetch_tle(str(data_cfg["celestrak_url"]))
                tle_records = parse_tle_records(tle_raw, record_limit)
                if len(tle_records) < int(data_cfg["num_satellites"]):
                    raise TLEFetchError(
                        f"CelesTrak response contained only {len(tle_records)} valid TLE records; "
                        f"{data_cfg['num_satellites']} are required."
                    )
                tle_cache.write_text(tle_raw, encoding="utf-8")
                source = "celestrak_tle_live_skyfield"
            except (TLEFetchError, OSError) as exc:
                tle_error = str(exc)
                for candidate, candidate_source in [
                    (tle_cache, "celestrak_tle_cache_skyfield"),
                    (tle_snapshot, "celestrak_tle_snapshot_skyfield"),
                ]:
                    if candidate is None or not candidate.exists():
                        continue
                    candidate_raw = candidate.read_text(encoding="utf-8")
                    candidate_records = parse_tle_records(candidate_raw, record_limit)
                    if len(candidate_records) >= int(data_cfg["num_satellites"]):
                        tle_raw = candidate_raw
                        tle_records = candidate_records
                        source = candidate_source
                        if candidate != tle_cache:
                            tle_cache.write_text(candidate_raw, encoding="utf-8")
                        print(
                            f"WARNING: live CelesTrak download unavailable; using verified "
                            f"CelesTrak {'cache' if candidate == tle_cache else 'snapshot'}: {tle_error}",
                            flush=True,
                        )
                        break
        if not dynamic_candidates and len(tle_records) >= int(data_cfg["num_satellites"]):
            try:
                propagation_start = configured_propagation_start(data_cfg)
                propagated = propagate_with_skyfield(
                    tle_records,
                    int(data_cfg["step_seconds"]),
                    int(data_cfg["num_steps"]),
                    propagation_start,
                )
                sat_pos = propagated
            except RuntimeError as exc:
                tle_error = str(exc)

    plane_pos, demand = generate_airplanes(cfg)
    if dynamic_candidates and len(tle_records) >= int(data_cfg["num_satellites"]):
        minimum_altitude = float(data_cfg.get("candidate_pool_altitude_min_km", 500.0))
        maximum_altitude = float(data_cfg.get("candidate_pool_altitude_max_km", 600.0))
        candidate_pool_records = filter_tle_records_by_altitude(
            tle_records,
            minimum_altitude,
            maximum_altitude,
        )
        if len(candidate_pool_records) < int(data_cfg["num_satellites"]):
            tle_error = (
                f"Only {len(candidate_pool_records)} Starlink records remain after the "
                f"{minimum_altitude:g}-{maximum_altitude:g} km candidate-pool filter."
            )
        else:
            try:
                sat_pos, elevation, coverage, sat_ids, propagation_start = (
                    generate_dynamic_satellite_candidates(
                        cfg,
                        candidate_pool_records,
                        plane_pos,
                    )
                )
                congestion = generate_dynamic_congestion(cfg, sat_ids)
            except RuntimeError as exc:
                tle_error = str(exc)

    if sat_pos is None and prefer_celestrak and not allow_synthetic_fallback:
        raise RuntimeError(
            "Paper-result dataset requires CelesTrak TLE propagation, but it was unavailable. "
            f"Reason: {tle_error or 'no valid TLE records were returned'}. "
            "Check network access and the skyfield dependency. Use "
            "configs/synthetic_smoke.yaml for a trend-only experiment."
        )

    if sat_pos is None:
        sat_pos = generate_synthetic_satellites(cfg)
        source = "synthetic"
        tle_records = [(f"SYNTH-{i:02d}", "", "") for i in range(int(data_cfg["num_satellites"]))]
        candidate_pool_records = []

    if elevation is None or coverage is None:
        elevation, coverage = compute_links(cfg, sat_pos, plane_pos)
    if congestion is None:
        congestion = generate_congestion(cfg)
    timestamps_s = np.arange(int(data_cfg["num_steps"]), dtype=np.int32) * int(data_cfg["step_seconds"])
    visible_counts = coverage.sum(axis=2)
    no_visible_fraction = float((visible_counts == 0).mean())

    if source == "synthetic":
        print(
            "WARNING: using a synthetic constellation. Results are trend-only and "
            "must not be compared numerically with the paper.",
            flush=True,
        )
    if no_visible_fraction > 0.05:
        print(
            f"WARNING: no satellite is visible in {no_visible_fraction:.2%} of airplane-timesteps.",
            flush=True,
        )
    maximum_outage = data_cfg.get("max_no_visible_fraction")
    if (
        dynamic_candidates
        and maximum_outage is not None
        and no_visible_fraction > float(maximum_outage)
    ):
        raise RuntimeError(
            f"Dynamic candidate geometry failed: no-visible fraction "
            f"{no_visible_fraction:.2%} exceeds the configured "
            f"{float(maximum_outage):.2%} limit."
        )

    out = dataset_path(cfg)
    dataset_arrays = {
        "timestamps_s": timestamps_s,
        "sat_pos": sat_pos.astype(np.float32),
        "plane_pos": plane_pos.astype(np.float32),
        "elevation": elevation.astype(np.float32),
        "coverage": coverage.astype(np.bool_),
        "congestion": congestion.astype(np.float32),
        "demand": demand.astype(np.float32),
    }
    if sat_ids is not None:
        dataset_arrays["sat_ids"] = sat_ids.astype(np.int32)
    np.savez_compressed(out, **dataset_arrays)

    selected_catalog_ids = (
        sorted(int(value) for value in np.unique(sat_ids) if int(value) >= 0)
        if sat_ids is not None
        else []
    )
    selected_catalog_id_set = set(selected_catalog_ids)
    metadata_records = (
        [
            record
            for record in candidate_pool_records
            if tle_catalog_id(record) in selected_catalog_id_set
        ]
        if sat_ids is not None
        else tle_records
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
        "satellite_selection_mode": selection_mode,
        "candidate_pool_size": len(candidate_pool_records),
        "candidate_pool_altitude_min_km": float(
            data_cfg.get("candidate_pool_altitude_min_km", 500.0)
        ),
        "candidate_pool_altitude_max_km": float(
            data_cfg.get("candidate_pool_altitude_max_km", 600.0)
        ),
        "selected_catalog_ids": selected_catalog_ids,
        "propagation_start_utc": propagation_start.isoformat() if propagation_start else None,
        "demand_min": float(data_cfg["demand_min"]),
        "demand_max": float(data_cfg["demand_max"]),
        "congestion_base_min": float(data_cfg["congestion_base_min"]),
        "congestion_base_max": float(data_cfg["congestion_base_max"]),
        "allow_synthetic_fallback": bool(allow_synthetic_fallback),
        "no_visible_fraction": no_visible_fraction,
        "max_no_visible_fraction": (
            float(maximum_outage) if maximum_outage is not None else None
        ),
        "visible_satellites_mean": float(visible_counts.mean()),
        "visible_satellites_max": int(visible_counts.max()),
        "tle_url": str(data_cfg["celestrak_url"]),
        "tle_cache_file": str(tle_cache),
        "tle_snapshot_file": str(tle_snapshot) if tle_snapshot else None,
        "tle_error": tle_error,
        "tle_sha256": sha256_text(tle_raw or ""),
        "satellites": [
            {
                "name": rec[0],
                "catalog_id": tle_catalog_id(rec) if rec[1] else None,
                "line1": rec[1],
                "line2": rec[2],
            }
            for rec in metadata_records
        ],
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


def validate_dataset_provenance(cfg: dict[str, Any]) -> None:
    """Reject trend-only data when a config claims a paper-result run."""

    if bool(cfg["data"].get("allow_synthetic_fallback", True)):
        return
    path = metadata_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {path}. Rebuild the dataset.")
    metadata = read_json(path)
    source = str(metadata.get("source", "unknown"))
    if source not in {
        "celestrak_tle_live_skyfield",
        "celestrak_tle_cache_skyfield",
        "celestrak_tle_snapshot_skyfield",
    }:
        raise RuntimeError(
            f"Paper-result config requires CelesTrak/Skyfield data, but metadata reports "
            f"source={source!r}. Rebuild after fixing TLE propagation."
        )
    expected_selection = str(
        cfg["data"].get("satellite_selection_mode", "fixed_first_n")
    )
    actual_selection = str(
        metadata.get("satellite_selection_mode", "fixed_first_n")
    )
    if actual_selection != expected_selection:
        raise RuntimeError(
            f"Dataset selection mode is {actual_selection!r}, but the config "
            f"requires {expected_selection!r}. Rebuild the dataset."
        )
    for key in (
        "num_steps",
        "step_seconds",
        "num_satellites",
        "num_airplanes",
        "theta_min_deg",
        "demand_min",
        "demand_max",
        "congestion_base_min",
        "congestion_base_max",
    ):
        if key not in metadata:
            raise RuntimeError(
                f"Dataset metadata does not record {key!r}. Rebuild the dataset."
            )
        expected = cfg["data"][key]
        actual = metadata[key]
        matches = (
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
            if isinstance(expected, float)
            else actual == expected
        )
        if not matches:
            raise RuntimeError(
                f"Dataset metadata {key}={actual!r}, but the config requires "
                f"{expected!r}. Rebuild the dataset."
            )
    configured_start = cfg["data"].get("propagation_start_utc")
    if configured_start is not None:
        expected_start = configured_propagation_start(cfg["data"]).isoformat()
        actual_start = metadata.get("propagation_start_utc")
        if actual_start != expected_start:
            raise RuntimeError(
                f"Dataset propagation_start_utc={actual_start!r}, but the config "
                f"requires {expected_start!r}. Rebuild the dataset."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the NTN handover dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Generate the synthetic/TLE-backed dataset.")
    build.add_argument("--config", default="configs/paper_result.yaml")
    build.add_argument("--fast", action="store_true", help="Generate a small smoke-test dataset.")
    build.add_argument("--synthetic-only", action="store_true", help="Skip CelesTrak and use deterministic synthetic satellites.")
    build.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow a trend-only synthetic fallback if TLE propagation fails.",
    )
    args = parser.parse_args(argv)
    if args.cmd == "build":
        out = build_dataset(
            args.config,
            fast=args.fast,
            prefer_celestrak=not args.synthetic_only,
            allow_synthetic_fallback=args.synthetic_only or args.allow_synthetic_fallback or None,
        )
        print(f"Wrote dataset to {out}")


if __name__ == "__main__":
    main()

