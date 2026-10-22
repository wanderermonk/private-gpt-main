"""Pure coordinate parsing/validation helpers (no deps). Offline-safe.

Used as a deterministic fallback/complement to LLM extraction: regexes pull
explicit decimal coordinates straight out of document text, so even if the
model misses some, clearly-written lat/long pairs are still mapped.
"""

from __future__ import annotations

import re


def valid_lat(lat: float) -> bool:
    return -90.0 <= lat <= 90.0


def valid_lon(lon: float) -> bool:
    return -180.0 <= lon <= 180.0


def valid_coord(lat: float, lon: float) -> bool:
    return valid_lat(lat) and valid_lon(lon)


# "12.9716, 77.5946"  /  "12.9716 N, 77.5946 E"  /  "-33.86, 151.21"
_DECIMAL_PAIR = re.compile(
    r"(-?\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,; ]\s*"
    r"(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?"
)


def _signed(value: str, hemi: str | None, negative_letter: str) -> float:
    v = float(value)
    if hemi and hemi.lower() == negative_letter:
        v = -abs(v)
    return v


def extract_decimal_pairs(text: str) -> list[tuple[float, float]]:
    """Find decimal lat/lon pairs in free text. Returns [(lat, lon), ...].

    Conservative: only accepts pairs that validate as real coordinates, so
    stray number pairs (prices, dates) are mostly rejected.
    """
    out: list[tuple[float, float]] = []
    for m in _DECIMAL_PAIR.finditer(text):
        lat_s, lat_h, lon_s, lon_h = m.groups()
        try:
            lat = _signed(lat_s, lat_h, "s")
            lon = _signed(lon_s, lon_h, "w")
        except ValueError:
            continue
        # Require at least one decimal point somewhere to avoid "1, 2" noise.
        if "." not in lat_s and "." not in lon_s:
            continue
        if valid_coord(lat, lon):
            out.append((round(lat, 6), round(lon, 6)))
    return out
