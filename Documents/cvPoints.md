# cvPoints — Cal/Val GPS Ground-Truth Points

Reads GPS-derived reference velocities, converts them to polar-stereographic
coordinates, and computes statistics against a `nisarVel` or `nisarVelSeries`
map.  Supports both static point files and time-varying GPS files.

---

## Construction

```python
from nisardev import cvPoints

# Static file (all points at a single epoch)
cv = cvPoints('gps_points.dat', epsg=3413)

# Time-varying: list of per-point GPS files
cv = cvPoints(['pt1.dat', 'pt2.dat', 'pt3.dat'], epsg=3413)
```

**File format** (static `.dat`): space-delimited columns —
`lat lon elevation vx vy vz [weight]`

**File format** (time-varying, comma-delimited):
`date, site, lat, lon, vx, vx_sigma, vy, vy_sigma, vv, vv_sigma`

---

## I/O

| Method | Description |
|--------|-------------|
| `readCVs(cvFile=None)` | (Re-)read CV points from file. |
| `writeCVs(cvFileOut, comment=None)` | Write non-culled points to a file. |
| `applyCullFile(cullFile)` | Load a cull file and mark the listed points for exclusion. |
| `setNoCull(noCull)` | Directly supply a boolean array marking points to keep. |

---

## Point selection

| Method | Description |
|--------|-------------|
| `allCVs()` | Bool array: all valid points (speed ≥ 0). |
| `zeroCVs()` | Bool array: effectively stationary points (speed < 0.00001 m/yr). |
| `vRangeCVs(minv, maxv)` | Bool array: points with speed in `[minv, maxv)`. |
| `NallCVs()` | Count of all valid points. |
| `NzeroCVs()` | Count of zero-speed points. |
| `NVRangeCVs(minv, maxv)` | Count of points in speed range. |

---

## Coordinates

| Method | Description |
|--------|-------------|
| `lltoxy(lat, lon, units='m')` | Convert lat/lon arrays to polar-stereographic x/y. |
| `xyAll(units='m')` | x, y of all points. |
| `xyNoCull(units='m')` | x, y of uncullled points. |
| `xyZero(units='m')` | x, y of zero-speed points. |
| `xyVRange(minv, maxv, units='m')` | x, y of points in speed range `[minv, maxv)`. |
| `boundingBox(units='m', pad=10000.)` | Bounding box of all points with padding. |

---

## Differences (map vs GPS)

| Method | Description |
|--------|-------------|
| `cvDifferences(x, y, iPts, vel, units='m', date=None)` | Interpolate `vel` at the given x/y points and return `(dvx, dvy)` — the map minus the GPS values. |
| `vAllData(vel, units='m', date=None)` | Interpolate `vel` at all GPS point locations. |
| `vRangeData(vel, minv, maxv, units='m', date=None)` | Interpolate `vel` at points in speed range `[minv, maxv)`. |

---

## Statistics

| Method | Description |
|--------|-------------|
| `vRangeStats(vel, minv, maxv, date=None, table=False)` | Mean, sigma, RMS of `vel` − GPS differences for points in `[minv, maxv)`.  Set `table=True` to receive a Pandas DataFrame. |
| `noCullStats(vel, date=None, table=False)` | Same, but for all non-culled points. |
| `timeSeriesStats(myVelSeries, minv, maxv)` | Point-by-point stats across every time layer of `myVelSeries`.  Returns a DataFrame + summary dict. |
| `statsStyle(styler, thresh=1.0, caption=None)` | Apply colour-coded formatting to a stats DataFrame `styler` (RMS > thresh shown in red). |

---

## Plotting locations

| Method | Description |
|--------|-------------|
| `plotAllCVLocs(units='m', vel=None, ax=None, **kwargs)` | Plot locations of all CV points on a map. |
| `plotVRangeCVLocs(minv, maxv, units='m', vel=None, ax=None, **kwargs)` | Plot locations of points in speed range `[minv, maxv)`. |
| `plotOutlierLocs(minv, maxv, vel, units='m', nSig=3, ax=None, **kwargs)` | Plot locations of outlier points (residual > `nSig` × sigma). |
| `getOutlierLocs(minv, maxv, vel, units='m', nSig=3)` | Return x, y coordinates of outlier points. |

---

## Plotting differences

| Method | Description |
|--------|-------------|
| `plotVRangeCVDiffs(vel, minv, maxv, ax=None, xColor='r', yColor='b', ...)` | Plot `dvx` and `dvy` as a function of point index. |
| `plotVRangeHistDiffs(vel, minv, maxv, axes=None, ...)` | Histogram of `dvx` and `dvy` clipped to ±3σ. |
| `plotTimesSeriesData(myVelSeries, minv, maxv, bands=['vv'], ...)` | Time-series comparison plot: GPS points (circles) overlaid on velocity series values (stars) for each point. |

---

## Time-varying GPS

| Method | Description |
|--------|-------------|
| `velocityForDateRange(date1, date2)` | Recompute mean `vx`, `vy`, `lat`, `lon` for each point for the given date window.  Used internally by `timeSeriesDifferences`. |
| `timeSeriesDifferences(myVelSeries, minv, maxv)` | For each time layer in `myVelSeries`, match the GPS date window and accumulate differences for all points in the speed range.  Returns a dict of arrays keyed by `vx`, `vy`, `vv`, `dvx`, `dvy`, etc. |
