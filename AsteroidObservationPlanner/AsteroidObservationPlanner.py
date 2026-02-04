import os
import gzip
import requests
import numpy as np
import pandas as pd
from datetime import timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import TimeDelta


ASTEROID_NAMES = {}

# ============================================================
# USER CONFIG
# ============================================================

LOCATION = {
    'lat': 0,     # degrees
    'lon': 0,     # degrees East
    'elevation': 480    # meters
}

# Time Zone
TZ = timezone(timedelta(hours=10.5))

# Observation nights (Time is UTC mid-session) 
NIGHTS = {
    "2026-02-4": Time("2026-01-25 11:30").jd,
    "2026-02-5": Time("2026-01-26 11:30").jd   
}

# MPC OBS code submission requires submission of objects fainter than 16 in apparent magnitude
# Apparently magnitude of 16 calculates to an absolute magnitude of 
# M= m- 5 log10 (d) + 5

# Absolute magnitude limits
MAG_MIN = 16
MAG_MAX = 19

# Desired final count
TARGET_COUNT = 6

# Batch size for processing - best not to change this
BATCH_SIZE = 100

# Number of concurrent API calls - best not to change this
MAX_WORKERS = 10

# Telescopius horizon mask (AZ°, ALT°)
HORIZON_POINTS = np.array([
    [172, 60], [186, 53], [249, 45], [271, 36], [290, 26],
    [306, 31], [330, 43], [357, 50], [29, 41],  [53, 32],
    [73, 32],  [101, 30], [119, 31], [130, 54], [129, 63]
])

# ============================================================
# HORIZON FUNCTION
# ============================================================

HORIZON_POINTS[:, 0] %= 360
HORIZON_POINTS = HORIZON_POINTS[np.argsort(HORIZON_POINTS[:, 0])]

def horizon_alt(az):
    return np.interp(az, HORIZON_POINTS[:,0], HORIZON_POINTS[:,1], period=360)

# ============================================================
# MPCORB HANDLING
# ============================================================

NEO_URL = "https://www.minorplanetcenter.net/iau/MPCORB/NEA.txt"
NEO_FILE = "NEA.txt"

def ensure_mpcorb():
    if os.path.exists(NEO_FILE):
        return

    print("Downloading NEA.txt...")
    r = requests.get(NEO_URL, stream=True)
    with open("NEA.txt", "wb") as f:
        f.write(r.content)

    #with gzip.open("NEA.txt.gz", "rb") as f_in:
    #    with open(NEO_FILE, "wb") as f_out:
    #        f_out.write(f_in.read())

    #os.remove("MPCORB.tar.gz")

# ============================================================
# Helper Functions
# ============================================================

def ra_to_hms(ra_deg):
    a = Angle(ra_deg * u.deg)
    return a.to_string(unit=u.hourangle, sep=' ', precision=2, pad=True)

def dec_to_dms(dec_deg):
    a = Angle(dec_deg * u.deg)
    return a.to_string(unit=u.deg, sep=' ', precision=1, alwayssign=True, pad=True)

def hour_angle_hours(ra_deg, time_utc):
    lst = time_utc.sidereal_time(
        'apparent',
        longitude=LOCATION['lon'] * u.deg
    )
    ra = Angle(ra_deg * u.deg).to(u.hourangle)
    ha = (lst - ra).wrap_at(12 * u.hourangle)
    return abs(ha.hour)

def compute_transit_time(ra_deg, date_utc):
    """
    Compute exact upper transit time near local night.
    Returns astropy Time (UTC).
    """
    ra = Angle(ra_deg * u.deg).to(u.hourangle)

    # Search window: 10pm–2am local ≈ 10:30–14:30 UTC
    t0 = Time(date_utc.iso.split()[0] + " 10:30")
    times = t0 + TimeDelta(np.linspace(0, 4, 241) * u.hour)

    lsts = times.sidereal_time(
        'apparent',
        longitude=LOCATION['lon'] * u.deg
    )

    ha = (lsts - ra).wrap_at(12 * u.hourangle).hour
    idx = np.argmin(np.abs(ha))

    return times[idx]


# ============================================================
# STAGE 1: PARSE & PRE-FILTER
# ============================================================

def load_candidate_numbers():
    print("Parsing MPCORB...")
    candidates = []

    with open(NEO_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue

            try:
                num = int(line[0:7])
                aM   = float(line[8:13])
                #a   = float(line[92:103])   # semi-major axis
                #e   = float(line[70:79])    # eccentricity

                name = line[166:194].strip()
                if not name:
                    name = f"({num})"
            except:
                continue

            # Store name lookup
            ASTEROID_NAMES[num] = name
            candidates.append(num)

    print(f"Candidate MBAs after MAG filter: {len(candidates)}")
    return candidates


# ============================================================
# STAGE 3.1: QUERY SINGLE ASTEROID
# ============================================================

def query_single_asteroid(num):
    """Query a single asteroid and check if it passes constraints."""
    
    epochs = list(NIGHTS.values())
    
    try:
        obj = Horizons(id=num, location=LOCATION, epochs=epochs)
        eph = obj.ephemerides()
        
        nightly = {}
        passes = True
        
        for i, night in enumerate(NIGHTS.keys()):
            mag = eph['V'][i]
            alt = eph['EL'][i]
            az  = eph['AZ'][i]
            ra  = eph['RA'][i]
            
            # Check constraints
            if not (MAG_MIN <= mag <= MAG_MAX):
                passes = False
                break
            
            # Check asteroid is visible above horizon at observation time
            if alt < horizon_alt(az):
                passes = False
                break
            
            transit = compute_transit_time(ra, Time(NIGHTS[night], format='jd'))
            
            nightly[night] = {
                "RA_J2000": ra,
                "DEC_J2000": eph['DEC'][i],
                "Vmag": mag,
                "Alt": alt,
                "Az": az,
                "Transit_UTC": transit
            }
        
        if passes:
            return (num, nightly)
        else:
            return None
            
    except Exception as e:
        return None


# ============================================================
# STAGE 3: PARALLEL BATCH QUERY
# ============================================================

def batch_query_horizons(asteroid_numbers):
    """Query multiple asteroids in parallel."""

    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_asteroid = {executor.submit(query_single_asteroid, num): num 
                              for num in asteroid_numbers}
        
        for future in as_completed(future_to_asteroid):
            num = future_to_asteroid[future]
            try:
                result = future.result()
                if result:
                    asteroid_num, nightly = result
                    results[asteroid_num] = nightly
                    print(f"{asteroid_num} passes")
            except Exception as e:
                print(f"✗ {num} failed: {e}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_mpcorb()
    candidates = load_candidate_numbers()

    print(f"Candidates is {len(candidates)}")
    
    all_results = {}
    
    # Process candidates in batches until we have enough
    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(candidates))
        batch = candidates[batch_start:batch_end]
        
        print(f"\nSearching asteroids {batch[0]} to {batch[-1]}...")
        
        batch_results = batch_query_horizons(batch)
        all_results.update(batch_results)
        
        print(f"Found {len(all_results)} passing asteroids so far")
        
        # Stop if we have enough
        if len(all_results) >= TARGET_COUNT:
            print(f"Reached target count of {TARGET_COUNT}")
            break
    
    if len(all_results) < TARGET_COUNT:
        print(f"\nOnly found {len(all_results)} objects after checking {batch_end} candidates.")
        print("Consider adjusting magnitude/H limits or increasing search range.")
    
    # Take top N results
    final_results = list(all_results.items())[:TARGET_COUNT]
    
    # Format output
    rows = []
    for num, data in final_results:
        for night, info in data.items():
            rows.append({
                "Asteroid_Number": num,
                "Asteroid_Name": ASTEROID_NAMES.get(num, ""),
                "Date": night,
                "RA": ra_to_hms(info["RA_J2000"]),
                "DEC": dec_to_dms(info["DEC_J2000"]),
                "Vmag": round(info["Vmag"], 2),
                "Alt": round(info["Alt"], 1),
                "Az": round(info["Az"], 1),
                "Transit_Local": info["Transit_UTC"].to_datetime(timezone=TZ).strftime("%H:%M")
            })
    
    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("FINAL OBSERVING LIST")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    df.to_csv("asteroid_targets.csv", index=False)
    print("\nSaved to asteroid_targets.csv")

if __name__ == "__main__":
    main()
