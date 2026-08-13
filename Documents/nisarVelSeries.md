# nisarVelSeries — Velocity Time Series

Stores a stack of velocity maps as a single xarray DataArray indexed by
`time` (midpoint), `time1` (start), and `time2` (end).  Bands are the same
as `nisarVel`: `vx`, `vy`, `vv`, `ex`, `ey`, `ev`, `dT`.

---

## Construction

```python
import nisardev as nisar

myVelSeries = nisar.nisarVelSeries(numWorkers=4)
```

**Parameters:**
- `numWorkers` — parallel dask download threads (default: 2; recommended: 4)
- `epsg` — EPSG code (auto-detected from files if omitted)
- `verbose` — print progress messages

---

## Key attributes

| Attribute | Description |
|-----------|-------------|
| `xr` | Full xarray DataArray (band × time × y × x).  Inspect with `myVelSeries.xr`. |
| `subset` | The currently active spatial/temporal view.  Inspect with `myVelSeries.subset`. |
| `time` | Array of midpoint dates (`numpy.datetime64`).  Iterate to loop over layers. |
| `time1`, `time2` | Arrays of start / end dates for each layer |
| `vx`, `vy`, `vv` | Velocity arrays — dask before `loadRemote()`, numpy after |
| `ex`, `ey`, `ev` | Error arrays |

---

## I/O

| Method | Description |
|--------|-------------|
| `readSeriesFromTiff(fileNames, url=False, useStack=True, readSpeed=False, useVelocity=True, useErrors=True, useDT=True, bbox=None, index1=3, index2=4, dateFormat='%d%b%y', overviewLevel=-1, chunkSize=2048)` | Build a series from a list of COG file paths or NSIDC https URLs.  `fileNames` typically comes from `grimp.cmrUrls(...).getCogs(replace='vv', removeTiff=True)`.  `readSpeed=False` is recommended — it computes `vv` from components on the fly. |
| `readSeriesFromNetCDF(cdfFile)` | Load a series from a NetCDF file previously saved by `toNetCDF()`. |
| `loadRemote()` | Download and materialise the lazy / dask-backed subset into memory.  Call after `subsetVel` to cache data locally for fast repeated access. |
| `toNetCDF(cdfFile)` | Save the *current subset* to a NetCDF file for later reloading.  If no subset has been applied this writes the full dataset (potentially hundreds of GB). |
| `writeSeriesToCOG(baseName, dateFormat='%d%b%y', suffix='V')` | Write each time layer as a separate Cloud-Optimised GeoTIFF. *(inherited from nisarBase2D)* |

**Typical remote-access pattern:**
```python
myVelSeries = nisar.nisarVelSeries(numWorkers=4)
myVelSeries.readSeriesFromTiff(myCogs, url=True, readSpeed=False, useStack=True)
myVelSeries.subsetVel(bbox)        # clip to region of interest
myVelSeries.loadRemote()           # download the subset
myVelSeries.toNetCDF('subset.nc')  # save for later

# Reload saved data
myVelReload = nisar.nisarVelSeries()
myVelReload.readSeriesFromNetCDF('subset.nc')
myVelReload.loadRemote()
```

---

## Subsetting

| Method | Description |
|--------|-------------|
| `subsetVel(bbox, useVelocity=True)` | Clip all layers to `{'minx', 'miny', 'maxx', 'maxy'}` in metres.  Can be called repeatedly to change the region; always re-clips from the full loaded extent. |
| `subsetData(bbox, useVelocity=True)` | Alias for `subsetVel`. |
| `timeSliceVel(date1, date2)` | Return a new `nisarVelSeries` with only layers between `date1` and `date2`. |
| `timeSliceData(date1, date2)` | Lower-level time-slice returning the base type. *(inherited from nisarBase2D)* |

---

## Accessing layers

| Method | Description |
|--------|-------------|
| `getMap(date, band='vv', returnXR=False)` | Extract the band array from the layer nearest `date`. |
| `inspect(band='vv', date=None, imgOpts={}, plotOpts={})` | Interactive Panel widget: map view + point-click time series. Pass display options through `imgOpts`, e.g. `imgOpts={'clim': (0, 2000), 'logz': True, 'cmap': 'hsv'}`. |

---

## Interpolation

| Method | Description |
|--------|-------------|
| `interp(x, y, date=None, units='m', returnXR=False, sourceEPSG=None, grid=False, **kwargs)` | Bilinear interpolation at `x`, `y`.  If `date=None`, interpolates all layers and returns results indexed by time.  Pass `sourceEPSG=4326` to supply lat/lon directly. |
| `interpGeo(x, y, myVars, date=None, units='m', returnXR=False)` | Lower-level interpolation for a named list of bands. *(inherited from nisarBase2D)* |

**Examples:**
```python
# Interpolate all dates at GPS points (metres)
vxAll, vyAll, vvAll = myVelSeries.interp(xGPS, yGPS, date=None, units='m')
# vvAll is shaped (nDates, nPoints)

# From lat/lon, returned as xarray
vPts = myVelSeries.interp(latGPS, lonGPS, date=None, returnXR=True, sourceEPSG=4326)
speed = vPts.sel(band='vv')   # shape: (time, nPoints)

# Single date
vx1, vy1, vv1 = myVelSeries.interp(xGPS, yGPS, date='2020-06-01', units='m')
```

---

## Statistics

| Method | Description |
|--------|-------------|
| `mean(skipna=True, squaredErrors=True)` | Temporal mean of each band.  Error bands are averaged in quadrature when `squaredErrors=True`. Returns a `nisarVelSeries` with one time layer. |
| `stdev(skipna=True)` | Temporal standard deviation. *(inherited from nisarBase2D)* |
| `meanXY(returnXR=False)` | Spatial mean (collapse x, y) for each band and time.  Pass `returnXR=True` for an xarray result. |
| `stdevXY(returnXR=False)` | Spatial standard deviation for each band and time. |
| `anomaly()` | Each layer minus the temporal mean. Returns a `nisarVelSeries`. |
| `numberValid()` | Count non-NaN pixels per band per layer. |

**Example — temporal stats:**
```python
velMean    = myVelSeries.mean()
velSigma   = myVelSeries.stdev()
velCount   = myVelSeries.numberValid()
velAnomaly = myVelSeries.anomaly()

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
velMean.displayVelForDate(ax=axes[0], units='km', colorBarLabel='Mean Speed (m/yr)')
velSigma.displayVelForDate(ax=axes[1], vmin=0, vmax=3, autoScale=False,
                            colorBarLabel='Sigma Speed (m/yr)')
velCount.displayVelForDate(ax=axes[2], vmin=0, vmax=7, autoScale=False,
                            colorBarLabel='N Valid', extend='neither')
```

**Example — spatial means over time:**
```python
meanXR = myVelSeries.meanXY(returnXR=True)
fig, ax = plt.subplots(figsize=(10, 7))
for band in meanXR.band:
    ax.plot(meanXR.time, meanXR.sel(band=band), '-o', label=band.item())
ax.legend()
```

---

## Geometry helpers *(inherited from nisarBase2D)*

Same as `nisarVel` — see [nisarVel.md](nisarVel.md) for the full list:
`boundingBox`, `size`, `pixSize`, `origin`, `bounds`, `extent`, `outline`,
`xyGrid`, `sizeInPixels`, `getDomain`.

---

## Date helpers *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `parseDate(date, ...)` | Convert `'YYYY-MM-DD'` string or `datetime`. |
| `datetime64ToDatetime(date64)` | Convert `numpy.datetime64` to Python `datetime`. |

---

## Display

### `displayVelForDate`

```python
myVelSeries.displayVelForDate(date=None, band='vv', ax=None,
                               units='m', midDate=True,
                               vmin=None, vmax=None, percentile=100,
                               autoScale=True, scale='linear',
                               cmap='RdYlBu_r', backgroundColor=None,
                               colorBar=True, colorBarLabel=None,
                               colorBarPosition='right', colorBarSize='5%', colorBarPad=0.05,
                               title=None, axisOff=False,
                               labelFontSize=10, plotFontSize=10, titleFontSize=12,
                               extend='both', wrap=None)
```

Display the velocity layer nearest to `date`.  If `date=None`, uses the first layer.

| Key parameter | Notes |
|---------------|-------|
| `date` | `'YYYY-MM-DD'` string, `datetime`, or `None` (first layer) |
| `band` | `'vx'`, `'vy'`, `'vv'`, `'ex'`, `'ey'`, `'ev'`, `'dT'` |
| `units` | `'m'` or `'km'` — controls axis tick units |
| `midDate` | `True` → title shows midpoint; `False` → shows date range |
| `scale` | `'linear'` or `'log'` |
| `percentile` | Clip colour range to this percentile (e.g. `99` ignores top 1%) |
| `autoScale` | `True` auto-scales; `False` uses `vmin`/`vmax` |
| `axisOff` | Remove x/y axis ticks |
| `colorBarPosition` | `'right'`, `'left'`, `'top'`, `'bottom'` |
| `extend` | `'both'`, `'min'`, `'max'`, `'neither'` |
| `labelFontSize` | Font size for axis tick labels |
| `plotFontSize` | Font size for axis titles/labels |
| `titleFontSize` | Font size for the figure title |

**Examples:**
```python
# Iterate over all dates
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for date, ax in zip(myVelSeries.time[0:6], axes.flatten()):
    myVelSeries.displayVelForDate(date=date, band='vv', ax=ax, units='km')
fig.tight_layout()

# Log scale with custom font sizes and no axis ticks
myVelSeries.displayVelForDate('2020-01-01', ax=ax, units='km',
                               scale='log', axisOff=True,
                               labelFontSize=10, plotFontSize=9, titleFontSize=14,
                               vmin=0, vmax=2000)
```

### Profile and point plots

| Method | Description |
|--------|-------------|
| `plotProfile(x, y, *args, band='vv', date=None, ax=None, units='m', midDate=True, **kwargs)` | Plot values along a transect for one date.  Extra `*args`/`**kwargs` go to `ax.plot`.  Typically called in a loop over `myVelSeries.time`. |
| `plotPoint(x, y, *args, band='vv', ax=None, units='m', sourceEPSG=None, **kwargs)` | Plot the time series at a single map location.  Pass `sourceEPSG=4326` to give lat/lon directly. |
| `labelProfilePlot(ax, band='vv', xLabel=None, yLabel=None, units='m', title=None, fontScale=1.0, plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a profile plot. |
| `labelPointPlot(ax, band='vv', xLabel=None, yLabel=None, title=None, plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a point-vs-time plot. |

**Example — profile for each year:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Profile
xprof = np.arange(200, 280, 0.25)   # km
yprof = np.full(xprof.shape, -1530.0)
for date in myVelSeries.time:
    myVelSeries.plotProfile(xprof, yprof, ax=axes[0], units='km', date=date)
myVelSeries.labelProfilePlot(axes[0], title='Speed Profile', fontScale=1.3)
axes[0].legend()

# Point time series
xpt, ypt = 250, -1530   # km
myVelSeries.plotPoint(xpt, ypt, 'r-*', ax=axes[1], units='km', markersize=12)
myVelSeries.labelPointPlot(axes[1], title='Speed at Point')
```

---

## Utility

| Method | Description |
|--------|-------------|
| `copy()` | Deep copy. *(inherited from nisarBase2D)* |
