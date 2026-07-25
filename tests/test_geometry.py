from ntn_repro.geometry import elevation_angle_deg, geodetic_to_ecef, wrap_lon_deg


def test_wrap_lon():
    assert wrap_lon_deg(190.0) == -170.0
    assert wrap_lon_deg(-190.0) == 170.0


def test_ecef_equator_radius():
    x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert round(x, 1) == 6371.0
    assert round(y, 1) == 0.0
    assert round(z, 1) == 0.0


def test_overhead_satellite_has_high_elevation():
    elevation = elevation_angle_deg(0.0, 0.0, 0.0, 0.0, 0.0, 1200.0)
    assert elevation > 89.0

