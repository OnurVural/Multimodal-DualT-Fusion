"""Heliographic -> pixel projection for cropping Helioviewer AIA JP2s to the
SWAN-SF active region.

SWAN-SF LAT_MIN/LAT_MAX/LON_MIN/LON_MAX are Stonyhurst degrees, so the bounding
box centre is projected with B0 = CRLT_OBS. verify_geom.py checks the result
against the HC_ANGLE column stored with each record.

Projection (orthographic; finite-distance error is ~0.3% of a solar radius and
far below the crop margin):

    x_px = CRPIX1 + R_SUN * cos(lat) * sin(lon)
    y_px = CRPIX2 - R_SUN * (sin(lat) * cos(B0) - cos(lat) * cos(lon) * sin(B0))
"""
import re
import numpy as np

XML_KEYS = ("CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "R_SUN", "RSUN_OBS",
            "CRLN_OBS", "CRLT_OBS", "CROTA2", "NAXIS1", "NAXIS2", "DATE-OBS")


def read_jp2_wcs(path, nbytes=300000):
    """Pull the WCS keywords out of the JP2's embedded XML box."""
    raw = open(path, "rb").read(nbytes)
    i = raw.find(b"<?xml")
    if i < 0:
        raise ValueError("no XML box in %s" % path)
    j = raw.find(b"</meta", i)
    seg = raw[i:(j if j > 0 else i + 20000)].decode("utf8", "replace")
    out = {}
    for k in XML_KEYS:
        m = re.search(r"<%s>([^<>]*)</%s>" % (re.escape(k), re.escape(k)), seg)
        if m is None:
            continue
        v = m.group(1).strip()
        if k == "DATE-OBS":
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def heliographic_to_pixel(lat_deg, lon_deg, b0_deg, wcs):
    """Stonyhurst (lat, lon) -> (x_px, y_px). Arrays or scalars."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    b0 = np.deg2rad(float(b0_deg))
    r = wcs.get("R_SUN")
    if r is None:
        r = wcs["RSUN_OBS"] / wcs["CDELT1"]
    x = r * np.cos(lat) * np.sin(lon)
    y = r * (np.sin(lat) * np.cos(b0) - np.cos(lat) * np.cos(lon) * np.sin(b0))
    return wcs["CRPIX1"] + x, wcs["CRPIX2"] - y


def heliocentric_angle(lat_deg, lon_deg, b0_deg):
    """Angular distance from disk centre, in degrees (compare to HC_ANGLE)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    b0 = np.deg2rad(float(b0_deg))
    c = np.sin(b0) * np.sin(lat) + np.cos(b0) * np.cos(lat) * np.cos(lon)
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def ar_bbox_pixels(row, wcs, margin=1.4, min_px=256):
    """Pixel bbox for one MVTS row's AR, with a margin, clamped to the frame.

    Projects a grid over the lat/lon box (not just corners) because the
    projection is nonlinear near the limb.
    """
    b0 = float(row["CRLT_OBS"])
    lats = np.linspace(float(row["LAT_MIN"]), float(row["LAT_MAX"]), 5)
    lons = np.linspace(float(row["LON_MIN"]), float(row["LON_MAX"]), 5)
    LA, LO = np.meshgrid(lats, lons)
    xs, ys = heliographic_to_pixel(LA.ravel(), LO.ravel(), b0, wcs)

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    half_w = max((xs.max() - xs.min()) * margin / 2.0, min_px / 2.0)
    half_h = max((ys.max() - ys.min()) * margin / 2.0, min_px / 2.0)
    half = max(half_w, half_h)  # square crop so the resize does not distort

    nx = wcs.get("NAXIS1", 4096.0)
    ny = wcs.get("NAXIS2", 4096.0)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    # shift (rather than clip) a box that runs off the frame, to keep it square
    if x0 < 0:
        x1 -= x0; x0 = 0.0
    if y0 < 0:
        y1 -= y0; y0 = 0.0
    if x1 > nx:
        x0 -= (x1 - nx); x1 = nx
    if y1 > ny:
        y0 -= (y1 - ny); y1 = ny
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))), (cx, cy)
