Asteroid Observation Planner
===========================

This tool is designed to generate a list of asteroids to observe based on your location and horizon mask, for observation over two nights. Target magnitude is set to be between 16 and 19, but is user configurable. 

Initially this tool was created to easily generate a list of asteroids to observe for a minor planet centre observer code submission.

How it works
--------------------------
The tool first tdownloads a data file from the Minor Planet Centre which contains number designations and names as well as orbital parameters of all reasonably observible minor planets. This file is a one off download done automatically if it isn't found on your machine already.

The data file is then parsed and asteroid targets are filtered to suit required magnitude and ensure they're main belt astreoids. This helps to limit the number of JPL Horizon API requests we send. The API requests are a large bottleneck in processing times so parallel execution of requests has been implemented. Note that this could possibly cause rate limiting issues but it's not been tested or investigated yet. 

The astreoid list is then queried against JPL Horizon API to ger precise ephemerides at our requested observation times. The results are filtered against our horizon mask and the RA and DEC positions as well as transit time and Vmag is output to the window. It also produces a csv file for future reference. 

How to use this tool
--------------------------
This tool takes only one argument which is the asteroid ID, but there are many parameters to set in the script itself. These are all set in the Observer Configuration section. 

#### Configuration

- LOCATION: Set your observation location longitude, latitude and altitude
- TZ: Set the hours= to your time zone in UTC offset. EG Adelaide is 10:30
- HORIZON_POINTS: This parameter is an array of points over 360 degrees to create a horizon mask. It can be done easily using [gyrocam](rkinnett.github.io/gyrocam) and your phone! 
- CULMINATION_UTC: The time in UTC at which the asteroid is at it's highest orbit path relative to your position
- MAX_HA_HOURS: Search for asteroids between +- x hours culmination
- TARGET_COUNT: Number of asteroids to observe
- MAG_MIN: Asteroid minimum magnitude 
- MAG_MAX: Asteroid maximum magnitude 


#### Example
Run:

`python3 AsteroidObservationPlanner.py`

Output
(some values obfuscated here to protect creator's location)

```
================================================================================
FINAL OBSERVING LIST
================================================================================
 Asteroid_Number  Asteroid_Name       Date          RA         DEC  Vmag  Alt    Az Transit_Local
             682    (682) Hagar 2026-01-xx xx 35 06.71 xx 50 06.0 17.11 59.4 237.5         22:00
             689     (689) Zita 2026-01-xx xx 16 09.98 xx 26 29.1 16.43 45.1 106.1         01:53
             827 (827) Wolfiana 2026-01-xx xx 27.03 20 xx 13.8 17.56 69.9 181.8            22:56
             822   (822) Lalage 2026-01-xx xx 03 14.11 xx 53 13.8 17.29 67.7 153.1         23:41
             864     (864) Aase 2026-01-xx xx 38 51.62 xx 30 29.7 17.54 58.2 230.6         22:00
             935   (935) Clivia 2026-01-xx xx 30 59.16 xx 03 15.4 17.55 65.9 174.4         23:08
             956    (956) Elisa 2026-01-xx xx 06 37.12 xx 37 36.9 16.23 70.0 146.4         23:44
================================================================================

Saved to asteroid_targets.csv
```