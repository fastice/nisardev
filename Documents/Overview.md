# nisardev — Package Overview

nisardev provides Python classes for reading, subsetting, interpolating, and
visualising glacier velocity and SAR/optical image products produced by the
Greenland Ice Mapping Project (GrIMP) and NISAR.

## Class hierarchy

```
nisarBase2D  (abstract base — shared geometry, I/O, interpolation, plotting)
├── nisarVel          single velocity map  (vx, vy, vv, ex, ey, ev, dT)
├── nisarVelSeries    velocity time-series stack
├── nisarImage        single SAR/optical image  (image, sigma0, gamma0)
└── nisarImageSeries  image time-series stack

cvPoints              cal/val GPS ground-truth points (standalone)
```

## Import convention

```python
import nisardev as nisar          # conventional alias used throughout GrIMPNotebooks
```

---

## Typical remote-access workflow

All GrIMP data are served as Cloud-Optimised GeoTIFFs (COGs) from NSIDC.
Data are opened *lazily* — nothing is downloaded until needed.

```python
import nisardev as nisar
import grimpfunc as grimp

# 1 — Authenticate with NSIDC EarthData
myUrls = grimp.cmrUrls(mode='nisar')
myUrls.initialSearch()

# 2 — Build the file-name list (wildcards replace the band component)
myCogs = myUrls.getCogs(replace='vv', removeTiff=True)

# 3 — Open (lazy, no download yet)
myVelSeries = nisar.nisarVelSeries(numWorkers=4)
myVelSeries.readSeriesFromTiff(myCogs, url=True, readSpeed=False, useStack=True)

# 4 — Subset to region of interest
bbox = {'minx': 200e3, 'miny': -1600e3, 'maxx': 280e3, 'maxy': -1500e3}
myVelSeries.subsetVel(bbox)

# 5 — Force download of the subsetted region
myVelSeries.loadRemote()

# 6 — Work with the data
fig, ax = plt.subplots()
myVelSeries.displayVelForDate('2020-01-01', ax=ax, units='km', scale='log')
```

---

## Key parameters and concepts

### `useStack` — how data are chunked

| Value | Behaviour |
|-------|-----------|
| `True` (default) | Each band is loaded as one contiguous array read — fastest for downloading and repeating operations on in-memory data. `chunks` keyword is ignored. |
| `False` | Data are tiled by dask/rioxarray.  Useful when you only need a few scattered points from a large subset *without* calling `loadRemote`. |

Prefer `useStack=True` for almost every workflow.

### `numWorkers` — parallel download threads

`numWorkers=4` is the recommended default; it is robust and performant.
Values above 8 give diminishing returns and may trigger connection limits at NSIDC.
Reduce to 2 if downloads fail with apparent file-not-found errors.

### `url=True` — remote vs local access

Pass `url=True` when loading from an `https://` link.  Omit (or pass `url=False`)
when the files are on a local or network filesystem.

### `readSpeed=False` — velocity-only products

When `readSpeed=False`, the speed band `vv` is computed on-the-fly from
`vx`/`vy` rather than read from a separate file — much faster than reading
the full product.

### `loadRemote()` — materialise lazy data

After `subsetVel` / `subsetImage` the data are still on the server.
Call `loadRemote()` to download and hold the subset in memory.
Subsequent operations (plots, interpolations, statistics) run much faster
because everything is cached locally.

### Bounding-box format

All methods that accept a spatial clip expect a dict with keys in **metres**
(default) or kilometres when `units='km'` is indicated:

```python
bbox = {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
```

Use `boundingBox(units='m')` to query the current extent of a loaded object.

---

## Quick-start examples

### Single velocity map — remote access

```python
import nisardev as nisar

myVel = nisar.nisarVel(numWorkers=4)
myVel.readDataFromTiff(myCogs[3], url=True, readSpeed=False, useStack=True)
myVel.subsetVel(bbox)
myVel.loadRemote()

# Interpolate at GPS points (coordinates in metres)
vx, vy, vv = myVel.interp(xGPS, yGPS, units='m')

# Interpolate passing lat/lon directly
vx, vy, vv = myVel.interp(latGPS, lonGPS, sourceEPSG=4326)

# Display
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
myVel.displayVel(ax=ax, units='km', midDate=True)
```

### Velocity time series — statistics and display

```python
import nisardev as nisar

myVelSeries = nisar.nisarVelSeries(numWorkers=4)
myVelSeries.readSeriesFromTiff(myCogs, url=True, readSpeed=False, useStack=True)
myVelSeries.subsetVel(bbox)
myVelSeries.loadRemote()

# Temporal statistics
velMean   = myVelSeries.mean()
velSigma  = myVelSeries.stdev()
velCount  = myVelSeries.numberValid()
velAnomaly = myVelSeries.anomaly()

# Display all dates
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for date, ax in zip(myVelSeries.time, axes.flatten()):
    myVelSeries.displayVelForDate(date=date, band='vv', ax=ax, units='km')
fig.tight_layout()

# Save and reload
myVelSeries.toNetCDF('subset.nc')
myVelReload = nisar.nisarVelSeries()
myVelReload.readSeriesFromNetCDF('subset.nc')
```

### Velocity time series — profile and point plots

```python
import numpy as np

# Profile (fixed y, varying x)
xprof = np.arange(200, 280, 0.25)          # km
yprof = np.full(xprof.shape, -1530.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for date in myVelSeries.time:
    myVelSeries.plotProfile(xprof, yprof, ax=axes[0], units='km', date=date)
myVelSeries.labelProfilePlot(axes[0], title='Speed Profile', fontScale=1.3)
axes[0].legend()

# Single-point time series
myVelSeries.plotPoint(250, -1530, 'r-*', ax=axes[1], units='km')
myVelSeries.labelPointPlot(axes[1], title='Speed at Point')
```

### SAR image series — load and display

```python
import nisardev as nisar

myImageSeries = nisar.nisarImageSeries(numWorkers=4)
myImageSeries.readSeriesFromTiff(myImageUrls.getCogs(), useStack=True)
myImageSeries.subsetImage(bbox)
myImageSeries.loadRemote()

# Interactive exploration
myImageSeries.inspect()

# Display statistics
mean    = myImageSeries.mean()
anomaly = myImageSeries.anomaly()
sigma   = myImageSeries.stdev()

fig, axes = plt.subplots(1, 3, figsize=(24, 11))
for image, ax, title in zip([mean, sigma, anomaly], axes, ['Mean', 'Sigma', 'Anomaly']):
    image.displayImageForDate(date='2020-02-28', ax=ax, cmap='gray', midDate=False)

# Save and reload
myImageSeries.toNetCDF('imageSeries.nc')
myImageSeriesReload = nisar.nisarImageSeries()
myImageSeriesReload.readSeriesFromNetCDF('imageSeries.nc')
```

### Overview image at reduced resolution

```python
myOverviewImage = nisar.nisarImage(numWorkers=4)
# overviewLevel=4 → 2^(4+1) × 25 m ≈ 800 m resolution
myOverviewImage.readDataFromTiff(myImageUrls.getCogs()[0], overviewLevel=4)
myOverviewImage.loadRemote()

# Use aspect ratio from sx/sy attributes
height = 3
width  = height * myOverviewImage.sx / myOverviewImage.sy
fig, ax = plt.subplots(figsize=(width, height))
myOverviewImage.displayImage(ax=ax, percentile=99, units='km',
                              cmap=plt.cm.gray, backgroundColor=(0.9, 0.9, 0.9))
```

### Compare against GPS cal/val points

```python
from nisardev import nisarVel, cvPoints

v  = nisarVel()
v.readDataFromTiff('Vel-20200101.20201231')
cv = cvPoints('gps_points.dat', epsg=3413)
muX, muY, sigX, sigY, rmsX, rmsY, n = cv.vRangeStats(v, 10, 5000)
```

---

## Coordinate systems

All spatial coordinates use the polar-stereographic projection of the loaded
product (EPSG:3413 for Greenland, EPSG:3031 for Antarctica).
Pass `units='m'` (default) or `units='km'` to most coordinate-accepting methods.

For interpolation from geographic coordinates, pass `sourceEPSG=4326` instead
of converting manually:

```python
vx, vy, vv = myVel.interp(latArray, lonArray, sourceEPSG=4326)
```

---

## Detailed class documentation

| File | Class |
|------|-------|
| [nisarVel.md](nisarVel.md) | Single velocity map |
| [nisarVelSeries.md](nisarVelSeries.md) | Velocity time series |
| [nisarImage.md](nisarImage.md) | Single SAR/optical image |
| [nisarImageSeries.md](nisarImageSeries.md) | Image time series |
| [cvPoints.md](cvPoints.md) | Cal/val GPS ground-truth points |
