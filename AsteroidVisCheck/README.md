Asteroid Visibility Check
===========================

This tool takes your location, horizon and a numbered asteroid designation and returns if the asteroid is visible from your location as well as the best time to obsderve it. 

How to use this tool
--------------------------
This tool takes only one argument which is the asteroid ID, but there are many parameters to set in the script itself. These are all set in the Observer Configuration section. 

#### Configuration

- Location: Set your observation location longitude, latitude and altitude.
- Time Zone: Set the hours= to your time zone in UTC offset. EG Adelaide is 10:30.
- Horizon: This parameter is an array of points over 360 degrees to create a horizon mask. It can be done easily using [gyrocam](rkinnett.github.io/gyrocam) and your phone! 


#### Example
Replace xxxx with an asteroid designation number such as 2116

Run:

`python3 AsteroidVisCheck.py xxxx`

Output:

```
======================================================================
ASTEROID POSITION CALCULATOR
======================================================================
Target: 1907
Observer: (lat=-xx.xx°, lon=xx.xx°)
Date/Time: 2026-01-04 15:51 UTC+10:30
======================================================================

CURRENT POSITION:
  RA (J2000):  01h xxm 20.51s
  DEC (J2000): +07° xx' 52.9"
  Magnitude:   V = 17.1
  Altitude:    15.9°
  Azimuth:     68.7°
  Status:      ✗ Below horizon
               (needs 32.0° altitude at this azimuth)

======================================================================
TONIGHT'S VIEWABLE WINDOW:
======================================================================
  Rises:       17:39 UTC+10:30
  Transit:     20:08 UTC+10:30 (best viewing)
  Sets:        22:50 UTC+10:30
  Duration:    5.2 hours

  At transit:
    RA (J2000):  01h xxm 25.43s
    DEC (J2000): +07° xx' 34.8"
    Altitude:    47.6°
    Azimuth:     0.7°
======================================================================
```