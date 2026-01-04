#!/usr/bin/env python3
"""
Calculate asteroid position and viewable times.
Usage: python asteroid_position.py <asteroid_id>
Example: python asteroid_position.py 1 Ceres
"""

import sys
import numpy as np
from datetime import timezone, timedelta
from astroquery.jplhorizons import Horizons
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle
import astropy.units as u

# ============================================================
# OBSERVER CONFIGURATION
# ============================================================

LOCATION = {
    'lat': 00.00,           # degrees
    'lon': 00.00,           # degrees East
    'elevation': 441,       # meters
    'utc_offset': 10.5      # hours (e.g., 10.5 for ACDT, -5 for EST, 0 for UTC)
}

# Telescopius horizon mask (AZ°, ALT°)
HORIZON_POINTS = np.array([
    [172, 60], [186, 53], [249, 45], [271, 36], [290, 26],
    [306, 31], [330, 43], [357, 50], [29, 41],  [53, 32],
    [73, 32],  [101, 30], [119, 31], [130, 54], [129, 63]
])

# ============================================================
# DERIVED CONFIGURATION (DO NOT EDIT)
# ============================================================

# Timezone based on UTC offset
TZ = timezone(timedelta(hours=LOCATION['utc_offset']))

# ============================================================
# HORIZON MASK
# ============================================================

# Sort and prepare horizon
HORIZON_POINTS[:, 0] %= 360
HORIZON_POINTS = HORIZON_POINTS[np.argsort(HORIZON_POINTS[:, 0])]

def horizon_alt(az):
    """Get minimum altitude at given azimuth based on horizon mask."""
    return np.interp(az, HORIZON_POINTS[:,0], HORIZON_POINTS[:,1], period=360)

# ============================================================
# COORDINATE FORMATTING
# ============================================================

def ra_to_hms(ra_deg):
    """Convert RA in degrees to HHh MMm SS.sss format."""
    a = Angle(ra_deg * u.deg)
    hms = a.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    h, m, s = hms.split(':')
    return f"{h}h {m}m {s}s"

def dec_to_dms(dec_deg):
    """Convert DEC in degrees to ±DD° MM' SS.s" format."""
    a = Angle(dec_deg * u.deg)
    dms = a.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True)
    d, m, s = dms.split(':')
    return f"{d}° {m}' {s}\""

# ============================================================
# TIME CALCULATIONS
# ============================================================

def compute_transit_time(ra_deg, date):
    """
    Compute upper transit (culmination) time for given RA.
    Returns astropy Time in UTC.
    """
    ra = Angle(ra_deg * u.deg).to(u.hourangle)
    
    # Search window: evening through night (6pm to 6am local)
    base_date = date.iso.split()[0]
    # Convert 6pm local to UTC
    local_6pm_utc_offset = 18.0 - LOCATION['utc_offset']
    
    # Convert decimal hours to HH:MM:SS format
    hours = int(local_6pm_utc_offset)
    minutes = int((local_6pm_utc_offset - hours) * 60)
    seconds = int(((local_6pm_utc_offset - hours) * 60 - minutes) * 60)
    
    # Handle negative hours (wrap to previous day)
    if hours < 0:
        hours += 24
    
    t0 = Time(base_date + f" {hours:02d}:{minutes:02d}:{seconds:02d}")
    times = t0 + TimeDelta(np.linspace(0, 12, 721) * u.hour)
    
    lsts = times.sidereal_time('apparent', longitude=LOCATION['lon'] * u.deg)
    ha = (lsts - ra).wrap_at(12 * u.hourangle).hour
    idx = np.argmin(np.abs(ha))
    
    return times[idx]

def find_viewable_window(asteroid_id, date):
    """
    Find the time window when asteroid is above horizon.
    Returns (rise_time, set_time, transit_time) or None if never visible.
    """
    base_time = Time(date.iso.split()[0] + " 00:00")
    end_time = base_time + TimeDelta(24 * u.hour)
    
    try: 
        obj = Horizons(id=asteroid_id, location=LOCATION, 
                   epochs={'start': base_time.iso, 
                           'stop': end_time.iso,
                           'step': '10m'})  # 10 minute steps
        eph = obj.ephemerides()
    except Exception as e:
        print(f"Error querying asteroid: {e}")
        return None
    
    # Create Time objects from Julian dates in the ephemeris
    times = Time(eph['datetime_jd'], format='jd')
    
    # Check which times are above horizon
    visible = []
    for i in range(len(eph)):
        alt = eph['EL'][i]
        az = eph['AZ'][i]
        if alt > horizon_alt(az):
            visible.append(i)
    
    if not visible:
        return None
    
    # Find rise and set times
    rise_idx = visible[0]
    set_idx = visible[-1]
    
    rise_time = times[rise_idx]
    set_time = times[set_idx]
    
    # Get RA for transit calculation
    mid_idx = visible[len(visible)//2]
    ra = eph['RA'][mid_idx]
    transit_time = compute_transit_time(ra, date)
    
    return (rise_time, set_time, transit_time)

# ============================================================
# MAIN CALCULATION
# ============================================================

def calculate_asteroid_position(asteroid_id):
    """Calculate current position and viewable times for an asteroid."""
    
    # Use current time
    now = Time.now()
    
    print(f"\n{'='*70}")
    print(f"ASTEROID POSITION CALCULATOR")
    print(f"{'='*70}")
    print(f"Target: {asteroid_id}")
    print(f"Observer: (lat={LOCATION['lat']:.3f}°, lon={LOCATION['lon']:.3f}°)")
    print(f"Date/Time: {now.to_datetime(timezone=TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"{'='*70}\n")
    
    # Query current position
    try:
        obj = Horizons(id=asteroid_id, location=LOCATION, epochs=now.jd)
        eph = obj.ephemerides()
    except Exception as e:
        print(f"Error: Could not query asteroid '{asteroid_id}'")
        print(f"Details: {e}")
        return
    
    # Extract current data
    ra = eph['RA'][0]
    dec = eph['DEC'][0]
    mag = eph['V'][0]
    alt = eph['EL'][0]
    az = eph['AZ'][0]
    
    # Current visibility
    min_alt = horizon_alt(az)
    is_visible = alt > min_alt
    
    print("CURRENT POSITION:")
    print(f"  RA (J2000):  {ra_to_hms(ra)}")
    print(f"  DEC (J2000): {dec_to_dms(dec)}")
    print(f"  Magnitude:   V = {mag:.1f}")
    print(f"  Altitude:    {alt:.1f}°")
    print(f"  Azimuth:     {az:.1f}°")
    print(f"  Status:      {'✓ VISIBLE' if is_visible else '✗ Below horizon'}")
    if not is_visible:
        print(f"               (needs {min_alt:.1f}° altitude at this azimuth)")
    
    # Calculate viewable window for tonight
    print(f"\n{'='*70}")
    print("TONIGHT'S VIEWABLE WINDOW:")
    print(f"{'='*70}")
    
    window = find_viewable_window(asteroid_id, now)
    
    if window:
        rise_time, set_time, transit_time = window
        
        rise_local = rise_time.to_datetime(timezone=TZ)
        set_local = set_time.to_datetime(timezone=TZ)
        transit_local = transit_time.to_datetime(timezone=TZ)
        
        duration = (set_time - rise_time).to_value(u.hour)
        
        print(f"  Rises:       {rise_local.strftime('%H:%M %Z')}")
        print(f"  Transit:     {transit_local.strftime('%H:%M %Z')} (best viewing)")
        print(f"  Sets:        {set_local.strftime('%H:%M %Z')}")
        print(f"  Duration:    {duration:.1f} hours")
        
        # Get position at transit
        obj_transit = Horizons(id=asteroid_id, location=LOCATION, epochs=transit_time.jd)
        eph_transit = obj_transit.ephemerides()
        transit_ra = eph_transit['RA'][0]
        transit_dec = eph_transit['DEC'][0]
        transit_alt = eph_transit['EL'][0]
        transit_az = eph_transit['AZ'][0]
        
        print(f"\n  At transit:")
        print(f"    RA (J2000):  {ra_to_hms(transit_ra)}")
        print(f"    DEC (J2000): {dec_to_dms(transit_dec)}")
        print(f"    Altitude:    {transit_alt:.1f}°")
        print(f"    Azimuth:     {transit_az:.1f}°")
        
    else:
        print("  Not visible tonight from this location")
        print("     (Never rises above horizon mask)")
    
    print(f"{'='*70}\n")

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python asteroid_position.py <asteroid_id>")
        print("\nExamples:")
        print("  python asteroid_position.py 1          # Ceres")
        print("  python asteroid_position.py 433        # Eros")
        print("  python asteroid_position.py 'Vesta'    # By name")
        sys.exit(1)
    
    # Handle asteroid ID (can be number or name)
    asteroid_id = ' '.join(sys.argv[1:])
    
    # Try to convert to int if it's a number
    try:
        asteroid_id = int(asteroid_id)
    except ValueError:
        pass  # Keep as string (name)
    
    calculate_asteroid_position(asteroid_id)

if __name__ == "__main__":
    main()