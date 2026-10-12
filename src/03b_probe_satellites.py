"""
Does modis-13Q1-061 carry one satellite or two?

    python src/03b_probe_satellites.py

MOD13Q1 is a 16-day product, so one growing season should return
about 15 composites. The catalogue returned 29, spaced 8 days apart,
which only happens if Terra (MOD) and Aqua (MYD) are both in the
collection, offset from each other by 8 days.

This matters because the two instruments cross the equator at
different times, Terra around 10:30 and Aqua around 13:30. They see
different illumination and different canopy water status, so their
NDVI differs systematically. Interleaved, that produces a sawtooth
between alternating observations: harmless for a seasonal mean,
damaging for any feature built on the rate of change, and green-up
rate is one of the features planned.
"""

from collections import Counter

import planetary_computer as pc
import pystac_client

STORY_COUNTY_BBOX = [-93.70, 41.86, -93.32, 42.21]

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

items = list(
    cat.search(
        collections=["modis-13Q1-061"],
        bbox=STORY_COUNTY_BBOX,
        datetime="2020-04-01/2020-10-31",
    ).items()
)

print(f"items returned: {len(items)}")
print(f"by product prefix: {dict(Counter(i.id[:3] for i in items))}")

print("\nfirst twelve, in date order:")
for i in sorted(items, key=lambda x: x.properties["start_datetime"])[:12]:
    print(f"  {i.properties['start_datetime'][:10]}  {i.id}")

terra = [i for i in items if i.id.startswith("MOD")]
aqua = [i for i in items if i.id.startswith("MYD")]

print(f"\nTerra composites in the season: {len(terra)}")
print(f"Aqua  composites in the season: {len(aqua)}")

if terra:
    d = sorted(i.properties["start_datetime"][:10] for i in terra)
    print(f"\nTerra alone, spacing check: {d[0]} to {d[-1]}, {len(d)} dates")
    print(f"  {d}")
