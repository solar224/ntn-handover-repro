from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0
EARTH_ROTATION_RAD_S = 7.2921159e-5
MU_EARTH_KM3_S2 = 398600.4418


def wrap_lon_deg(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    radius = EARTH_RADIUS_KM + alt_km
    cos_lat = math.cos(lat)
    return (
        radius * cos_lat * math.cos(lon),
        radius * cos_lat * math.sin(lon),
        radius * math.sin(lat),
    )


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    radius = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(z / radius))
    lon = math.degrees(math.atan2(y, x))
    return lat, wrap_lon_deg(lon), radius - EARTH_RADIUS_KM


def elevation_angle_deg(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_km: float,
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_km: float,
) -> float:
    obs = geodetic_to_ecef(observer_lat_deg, observer_lon_deg, observer_alt_km)
    sat = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_km)
    rx = sat[0] - obs[0]
    ry = sat[1] - obs[1]
    rz = sat[2] - obs[2]
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    on = math.sqrt(obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2])
    ux, uy, uz = obs[0] / on, obs[1] / on, obs[2] / on
    sin_el = max(-1.0, min(1.0, (rx * ux + ry * uy + rz * uz) / rn))
    return math.degrees(math.asin(sin_el))


def synthetic_satellite_position(
    sat_index: int,
    elapsed_s: float,
    num_satellites: int,
    altitude_km: float,
    inclination_deg: float = 53.0,
) -> tuple[float, float, float]:
    """Deterministic Walker-like circular LEO position.

    This is a fallback when live TLE propagation is unavailable. It is not
    intended to be orbital truth; it gives reproducible high-mobility NTN
    geometry with realistic time scales.
    """

    planes = max(1, int(round(math.sqrt(num_satellites))))
    sats_per_plane = max(1, math.ceil(num_satellites / planes))
    plane = sat_index % planes
    slot = sat_index // planes
    r = EARTH_RADIUS_KM + altitude_km
    mean_motion = math.sqrt(MU_EARTH_KM3_S2 / (r**3))
    inc = math.radians(inclination_deg)
    raan = 2.0 * math.pi * plane / planes
    phase = 2.0 * math.pi * slot / sats_per_plane + 0.35 * plane
    u = mean_motion * elapsed_s + phase

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)
    x_eci = math.cos(raan) * x_orb - math.sin(raan) * math.cos(inc) * y_orb
    y_eci = math.sin(raan) * x_orb + math.cos(raan) * math.cos(inc) * y_orb
    z_eci = math.sin(inc) * y_orb

    theta = EARTH_ROTATION_RAD_S * elapsed_s
    x_ecef = math.cos(theta) * x_eci + math.sin(theta) * y_eci
    y_ecef = -math.sin(theta) * x_eci + math.cos(theta) * y_eci
    z_ecef = z_eci
    return ecef_to_geodetic(x_ecef, y_ecef, z_ecef)


def interpolate_route(
    start: tuple[float, float],
    end: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    lat = start[0] + (end[0] - start[0]) * fraction
    dlon = wrap_lon_deg(end[1] - start[1])
    lon = wrap_lon_deg(start[1] + dlon * fraction)
    return lat, lon

