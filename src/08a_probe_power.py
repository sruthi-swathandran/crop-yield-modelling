"""
One request to NASA POWER, to see the response shape before writing
a script that makes 742 of them.

    python src/08a_probe_power.py

No API key needed. Story County, Iowa, ten days in July 2020.

WHAT TO LOOK FOR
----------------
  status 200
  the parameter block keyed by date as YYYYMMDD
  shortwave in MJ/m^2/day, somewhere around 15 to 30 in July
  the fill value for missing days, which the full script must treat
  as missing rather than as a number
"""

import json

import requests

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {
    "parameters": "ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,PRECTOTCORR",
    "community": "AG",
    "longitude": -93.46,          # Story County centroid, roughly
    "latitude": 42.04,
    "start": "20200701",
    "end": "20200710",
    "format": "JSON",
}


def main() -> None:
    r = requests.get(URL, params=PARAMS, timeout=120)
    print(f"status: {r.status_code}")
    r.raise_for_status()
    d = r.json()

    print(f"top-level keys: {list(d)}")
    print("\n--- header ---")
    print(json.dumps(d.get("header", {}), indent=2)[:1500])

    print("\n--- parameter metadata (units) ---")
    print(json.dumps(d.get("parameters", {}), indent=2)[:1500])

    print("\n--- values ---")
    for name, series in d["properties"]["parameter"].items():
        print(f"{name}:")
        for day, v in series.items():
            print(f"   {day}  {v}")

    print(f"\ngeometry returned: {d.get('geometry')}")


if __name__ == "__main__":
    main()
