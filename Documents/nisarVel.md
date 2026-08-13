# nisarVel — Single Velocity Map

Stores one velocity map with up to seven bands: `vx`, `vy`, `vv`, `ex`,
`ey`, `ev`, `dT` (component velocities, their errors, and the time-interval
length in days).  Data is held in an xarray DataArray (`self.xr`).

---

## Construction

```python
import nisardev as nisar

myVel = nisar.nisarVel(numWorkers=4)
```

**Parameters:**
- `numWorkers` — number of parallel dask threads for downloads (default: 2; recommended: 4)
- `epsg` — EPSG code for the projection (auto-detected from file if omitted)
- `verbose` — print progress messages

---

## Key attributes

| Attribute | Description |
|-----------|-------------|
| `xr` | The full xarray DataArray (bands × y × x) |
| `subset` | The currently active xarray view after subsetting |
| `vx`, `vy`, `vv` | Velocity component arrays (dask or numpy after `loadRemote()`) |
| `ex`, `ey`, `ev` | Error component arrays |
| `dT` | Time-interval length in days |
| `time` | The midpoint date (`numpy.datetime64`) |
| `time1`, `time2` | Start and end dates of the measurement window |
| `sx`, `sy` | Spatial size in the native units (pixels) |

---

## I/O

| Method | Description |
|--------|-------------|
| `readDataFromTiff(fileNameBase, url=False, useStack=True, readSpeed=False, useVelocity=True, useErrors=True, useDT=True, bbox=None, overviewLevel=-1, chunkSize=2048, masked=True, suffix='')` | Open a COG velocity product.  `fileNameBase` is the path/URL without the `.{band}.tif` suffix.  Pass `url=True` for NSIDC https links.  `readSpeed=False` (recommended) computes `vv` from `vx`/`vy` rather than reading a separate file. |
| `readDataFromNetCDF(cdfFile)` | Read a NetCDF file previously saved by `toNetCDF()`. |
| `loadRemote()` | Download and materialise the lazy-loaded (or dask-backed) data into memory.  Call after `subsetVel` to cache the subset locally. |
| `toNetCDF(cdfFile)` | Save the data to a NetCDF file. *(inherited from nisarBase2D)* |
| `writeCloudOptGeo(tiffRoot, full=False, myVars=None)` | Write as a Cloud-Optimised GeoTIFF. *(inherited from nisarBase2D)* |

---

## Subsetting

| Method | Description |
|--------|-------------|
| `subsetVel(bbox, useVelocity=True)` | Clip to a bounding box dict `{'minx', 'miny', 'maxx', 'maxy'}` in metres.  Can be called multiple times; always re-subsets from the full loaded extent. |
| `subsetData(bbox, useVelocity=True)` | Alias for `subsetVel`. |

---

## Interpolation

| Method | Description |
|--------|-------------|
| `interp(x, y, units='m', returnXR=False, sourceEPSG=None, **kwargs)` | Bilinear interpolation at coordinate arrays `x`, `y`.  Returns `[vx, vy, vv, ex, ey, ev, dT]` (only loaded bands).  Pass `sourceEPSG=4326` to supply lat/lon directly and skip manual reprojection.  Pass `returnXR=True` to receive an xarray instead of numpy arrays. |
| `interpGeo(x, y, myVars, date=None, units='m', returnXR=False)` | Lower-level interpolation for a named list of bands. *(inherited from nisarBase2D)* |

**Examples:**

```python
# From projected coordinates (metres)
vx, vy, vv = myVel.interp(xGPS, yGPS, units='m')

# From geographic coordinates (lat/lon)
vx, vy, vv = myVel.interp(latGPS, lonGPS, sourceEPSG=4326)

# Returned as xarray
velXR = myVel.interp(latGPS, lonGPS, sourceEPSG=4326, returnXR=True)
speed = velXR.sel(band='vv')
```

---

## Geometry helpers *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `boundingBox(units='m')` | Return `{'minx', 'miny', 'maxx', 'maxy'}` of the current extent. |
| `size(units='m')` | Return `(width, height)` of the spatial domain. |
| `pixSize(units='m')` | Return pixel size `(dx, dy)`. |
| `origin(units='m')` | Return lower-left corner `(x0, y0)`. |
| `bounds(units='m')` | Return `(minx, miny, maxx, maxy)` tuple. |
| `extent(units='m')` | Return `[minx, maxx, miny, maxy]` for matplotlib `imshow(extent=...)`. |
| `outline(units='m')` | Four corners as a closed polygon. |
| `xyGrid()` | `(X, Y)` meshgrid arrays for all pixel centres. |
| `sizeInPixels()` | Return `(nx, ny)`. |
| `getDomain(epsg)` | Return bounds reprojected to the given EPSG. |

---

## Statistics *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `mean(skipna=True, errors=[])` | Spatial mean of each band.  `errors` is a list of error band names to average in quadrature. |
| `stdev(skipna=True)` | Spatial standard deviation. |
| `meanXY(returnXR=False)` | Mean along x then y axes. |
| `stdevXY(returnXR=False)` | Stdev along x then y axes. |
| `anomaly()` | Data minus its spatial mean. |
| `numberValid()` | Count finite (non-NaN) pixels per band. |

---

## Date helpers *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `parseDate(date, defaultDate=True, returnString=False)` | Convert `'YYYY-MM-DD'` string or `datetime` to `datetime`. |
| `datetime64ToDatetime(date64)` | Convert `numpy.datetime64` to Python `datetime`. |
| `parseVelDatesFromFileName(fileNameBase, index1, index2, dateFormat)` | Parse `time1` and `time2` from the filename. |

---

## Display

### `displayVel`

```python
myVel.displayVel(band='vv', ax=None, units='m', midDate=True,
                 vmin=None, vmax=None, percentile=100,
                 autoScale=True, scale='linear',
                 cmap='RdYlBu_r', backgroundColor=None,
                 colorBar=True, colorBarLabel=None,
                 colorBarPosition='right', colorBarSize='5%', colorBarPad=0.05,
                 title=None, axisOff=False,
                 labelFontSize=10, plotFontSize=10, titleFontSize=12,
                 extend='both', wrap=None)
```

Plot one velocity band as a colour map.

| Key parameter | Notes |
|---------------|-------|
| `band` | One of `'vx'`, `'vy'`, `'vv'`, `'ex'`, `'ey'`, `'ev'`, `'dT'` |
| `units` | `'m'` or `'km'` for axis tick labels |
| `midDate` | `True` → title shows midpoint date; `False` → shows first–last date range |
| `scale` | `'linear'` or `'log'` (log colour scale for speed maps) |
| `percentile` | Clip colour range to this percentile (e.g. `99` ignores the top 1%) |
| `autoScale` | `True` → auto-compute colour limits from data; `False` → use `vmin`/`vmax` |
| `axisOff` | Remove x/y axis ticks and labels |
| `colorBarPosition` | `'right'` (default), `'left'`, `'top'`, or `'bottom'` |
| `extend` | Matplotlib colorbar extend: `'both'`, `'min'`, `'max'`, or `'neither'` |
| `backgroundColor` | Colour for masked / no-data pixels (RGB tuple or named colour) |

**Example:**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
myVel.displayVel(ax=axes[0], units='m',  midDate=True)
myVel.displayVel(ax=axes[1], units='km', midDate=False)
```

### Profile and point plots

| Method | Description |
|--------|-------------|
| `plotProfile(x, y, *args, band='vv', ax=None, date=None, units='m', midDate=True, **kwargs)` | Plot values along a transect as a function of distance.  `x`, `y` are coordinate arrays.  Extra `*args`/`**kwargs` are passed to `ax.plot`. |
| `plotPoint(x, y, *args, band='vv', ax=None, **kwargs)` | Mark a single point on the current axes.  Useful for indicating locations on a map. |
| `labelProfilePlot(ax, band='vv', xLabel=None, yLabel=None, units='m', title=None, fontScale=1.0, plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a profile plot. |
| `labelPointPlot(ax, band='vv', xLabel=None, yLabel=None, title=None, plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a point-vs-time plot. |

### Low-level display helpers *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `displayVar(band, date=None, ax=None, ...)` | Generic colour-map display for any band. |
| `colorSetup(scale, cmap, vmin, vmax, backgroundColor)` | Build a matplotlib `(norm, cmap)` pair. |
| `autoScaleRange(band, date, vmin, vmax, percentile, quantize)` | Compute colour limits clipped to a percentile. |
| `hsvSpeedRender(speed, vmin, vmax)` | Render speed as an HSV image (hue = flow direction). |

---

## Utility

| Method | Description |
|--------|-------------|
| `copy()` | Deep copy. *(inherited from nisarBase2D)* |
| `timeSliceData(date1, date2)` | Return a copy restricted to a date range. *(inherited from nisarBase2D)* |
| `inspect(band='vv', date=None, imgOpts={}, plotOpts={})` | Interactive Panel/HoloViews widget — map + point-click time series. |
