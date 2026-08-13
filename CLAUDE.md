# CLAUDE.md — nisardev

Active NISAR/GrIMP cal/val toolkit — xarray-based classes for reading, subsetting,
interpolating, and plotting velocity/image Cloud-Optimised GeoTIFFs (COGs) from
NSIDC, plus GPS cal/val comparison. Successor to `nisarfunc` (deprecated). See the
[packages CLAUDE.md](../CLAUDE.md) for pipeline context and
[nisarhdf/CLAUDE.md](../nisarhdf/CLAUDE.md) for the upstream HDF5 reader.

## Relationship to nisarhdf

**`nisardev` does not import or extend `nisarhdf` classes** — there is no class
inheritance between the two packages. They sit at different ends of the pipeline:

- `nisarhdf` reads **NISAR L1/L2 HDF5** products and writes **GrIMP big-endian
  binary flat files** consumed by the GIT64 C mosaic pipeline.
- `mosaicworkflow`/NSIDC processing turns those flat files into **Cloud-Optimised
  GeoTIFFs (COGs)**, which is the *input format* `nisardev` reads (`readDataFromTiff`,
  `readSeriesFromTiff`) via `rioxarray`/`xarray`, plus a NetCDF cache format
  (`toNetCDF`/`readDataFromNetCDF`).

Conceptually `nisardev` is the "downstream/cal-val" counterpart to `nisarhdf`'s
"upstream/ingest" role — both serve the same overall GrIMP velocity pipeline but
operate on different file formats and have independent class hierarchies.

## Class hierarchy

```
nisarBase2D  (abstract base — geometry, I/O, interpolation, stats, plotting)
├── nisarVel          single velocity map  (vx, vy, vv, ex, ey, ev, dT)
├── nisarVelSeries    velocity time-series stack
├── nisarImage        single SAR/optical image  (image, sigma0, gamma0)
└── nisarImageSeries  image time-series stack

cvPoints              cal/val GPS ground-truth points (standalone)
```

All concrete classes hold their data in a single `xarray.DataArray` (`self.xr`,
with the active view in `self.subset`). Series classes add a `time`/`time1`/`time2`
dimension; single-product classes have one implicit layer.

Import convention used throughout `GrIMPNotebooks`:
```python
import nisardev as nisar
```

## nisarBase2D — shared functionality

Constructor: `nisarBase2D(sx=None, sy=None, x0=None, y0=None, dx=None, dy=None, verbose=True, epsg=None, numWorkers=2, template=None)`

| Category | Methods |
|---|---|
| Geometry | `boundingBox`, `size`, `pixSize`, `origin`, `bounds`, `extent`, `outline`, `xyGrid`, `sizeInPixels`, `getDomain(epsg)` |
| Interpolation | `interp(x, y, units='m', sourceEPSG=None, returnXR=False, **kwargs)`, `interpGeo(x, y, myVars, date=None, ...)` |
| Stats | `mean`, `stdev`, `meanXY`, `stdevXY`, `anomaly`, `numberValid` |
| Dates | `parseDate`, `datetime64ToDatetime`, `timeSliceData(date1, date2)` |
| I/O | `toNetCDF(cdfFile)`, `writeCloudOptGeo(tiffRoot, full=False, myVars=None)`, `readGeodatFromTiff` |
| Display | `displayVar`, `colorSetup`, `autoScaleRange`, `hsvSpeedRender` |
| Misc | `copy()` (deep copy) |

`numWorkers` (default 2, recommended 4) sets dask thread count for COG downloads.
`epsg` is auto-detected from the source file if omitted (3413 = Greenland, 3031 = Antarctica).

## nisarVel — single velocity map

Bands: `vx`, `vy`, `vv`, `ex`, `ey`, `ev`, `dT` (component velocities, errors, time-interval in days).

| Method | Notes |
|---|---|
| `readDataFromTiff(fileNameBase, url=False, useStack=True, readSpeed=False, useVelocity=True, useErrors=True, useDT=True, bbox=None, overviewLevel=-1, chunkSize=2048, masked=True, suffix='')` | `fileNameBase` excludes the `.{band}.tif` suffix. `readSpeed=False` (recommended) computes `vv` from `vx`/`vy`. |
| `readDataFromNetCDF(cdfFile)` | reload a saved subset |
| `loadRemote()` | materialise lazy/dask data into memory |
| `subsetVel(bbox, useVelocity=True)` / `subsetData` | clip to `{'minx','miny','maxx','maxy'}` (metres); always re-clips from full extent |
| `interp(x, y, units='m', sourceEPSG=None, returnXR=False, **kwargs)` | returns `[vx, vy, vv, ex, ey, ev, dT]` (loaded bands only) |
| `displayVel(band='vv', ax=None, units='m', midDate=True, scale='linear', percentile=100, autoScale=True, cmap='RdYlBu_r', ...)` | colour-map plot |
| `plotProfile`, `plotPoint`, `labelProfilePlot`, `labelPointPlot` | transect/point plots |
| `inspect(band='vv', date=None, imgOpts={}, plotOpts={})` | interactive Panel/HoloViews widget |

## nisarVelSeries — velocity time series

Stack of `nisarVel`-shaped layers indexed by `time`/`time1`/`time2`.

| Method | Notes |
|---|---|
| `readSeriesFromTiff(fileNames, url=False, useStack=True, readSpeed=False, index1=3, index2=4, dateFormat='%d%b%y', overviewLevel=-1, chunkSize=2048, bbox=None)` | `fileNames` typically from `grimp.cmrUrls(...).getCogs(replace='vv', removeTiff=True)` |
| `readSeriesFromNetCDF(cdfFile)` / `loadRemote()` / `toNetCDF(cdfFile)` | NetCDF caches *current subset* (full dataset can be hundreds of GB if unsubsetted) |
| `writeSeriesToCOG(baseName, dateFormat='%d%b%y', suffix='V')` | write each layer as a COG |
| `subsetVel(bbox)` / `timeSliceVel(date1, date2)` | spatial / temporal clipping |
| `getMap(date, band='vv', returnXR=False)` | extract nearest-date layer |
| `interp(x, y, date=None, units='m', sourceEPSG=None, grid=False, returnXR=False, **kwargs)` | `date=None` interpolates all layers, result indexed by time |
| `mean(skipna=True, squaredErrors=True)`, `stdev`, `meanXY`, `stdevXY`, `anomaly`, `numberValid` | temporal/spatial stats; `mean`/`anomaly` return a `nisarVelSeries` |
| `displayVelForDate(date=None, band='vv', ...)` | same kwargs as `nisarVel.displayVel` |
| `plotProfile`, `plotPoint`, `inspect` | as above, looped over `time` |

## nisarImage / nisarImageSeries

Single/series SAR or optical image. Active band is one of `image` (DN, uint8),
`sigma0` (dB, float32), `gamma0` (terrain-corrected dB, float32) — auto-detected
from filename via `detectImageType`/`myVariables`. A series cannot mix image types.

| Method | Notes |
|---|---|
| `readDataFromTiff(fileNameBase, url=False, useStack=True, imageType=None, bbox=None, overviewLevel=-1, chunkSize=2048, masked=True)` | `overviewLevel=4` → ~800 m from 25 m base data |
| `readSeriesFromTiff(fileNames, url=False, useStack=True, index1=3, index2=4, dateFormat='%d%b%y', overviewLevel=-1, chunkSize=2048, bbox=None)` | image type inferred from first file |
| `subsetImage(bbox)` / `subsetData` | spatial clip |
| `timeSliceImage(date1, date2)` (series only) | temporal clip |
| `interp(x, y, units='m', returnXR=False, **kwargs)` | returns `[band_values]` |
| `displayImage(date=None, ax=None, cmap='gray', masked=True, percentile=100, scale='linear', ...)` / `displayImageForDate` (series) | `masked=True` treats 0 as no-data, `None` auto-detects |
| `mean`, `stdev`, `meanXY`, `stdevXY`, `anomaly`, `numberValid` (series) | temporal stats |
| `inspect()` (series) | interactive Panel widget |

`sx`/`sy` (pixel width/height) are useful for setting figure aspect ratio when
using reduced-resolution overviews.

## cvPoints — GPS cal/val comparison

Reads GPS-derived reference velocities, reprojects to polar stereographic, and
computes residual statistics against a `nisarVel`/`nisarVelSeries`.

```python
from nisardev import cvPoints

cv = cvPoints('gps_points.dat', epsg=3413)              # static file
cv = cvPoints(['pt1.dat', 'pt2.dat', 'pt3.dat'], epsg=3413)  # time-varying, one file per point
```

- Static file format (space-delimited): `lat lon elevation vx vy vz [weight]`
- Time-varying file format (comma-delimited): `date, site, lat, lon, vx, vx_sigma, vy, vy_sigma, vv, vv_sigma`

| Category | Methods |
|---|---|
| Point selection | `allCVs`, `zeroCVs`, `vRangeCVs(minv, maxv)`, `NallCVs`, `NzeroCVs`, `NVRangeCVs` |
| Coordinates | `lltoxy`, `xyAll`, `xyNoCull`, `xyZero`, `xyVRange`, `boundingBox(units='m', pad=10000.)` |
| Differences | `cvDifferences(x, y, iPts, vel, units='m', date=None)` → `(dvx, dvy)` = map − GPS |
| Stats | `vRangeStats(vel, minv, maxv, date=None, table=False)`, `noCullStats`, `timeSeriesStats(myVelSeries, minv, maxv)` |
| Plotting | `plotAllCVLocs`, `plotVRangeCVLocs`, `plotOutlierLocs`, `plotVRangeCVDiffs`, `plotVRangeHistDiffs`, `plotTimesSeriesData` |
| Time-varying | `velocityForDateRange(date1, date2)`, `timeSeriesDifferences(myVelSeries, minv, maxv)` |
| I/O | `readCVs`, `writeCVs`, `applyCullFile`, `setNoCull` |

## Typical remote-access workflow

```python
import nisardev as nisar
import grimpfunc as grimp

myUrls = grimp.cmrUrls(mode='nisar')
myUrls.initialSearch()
myCogs = myUrls.getCogs(replace='vv', removeTiff=True)

myVelSeries = nisar.nisarVelSeries(numWorkers=4)
myVelSeries.readSeriesFromTiff(myCogs, url=True, readSpeed=False, useStack=True)

bbox = {'minx': 200e3, 'miny': -1600e3, 'maxx': 280e3, 'maxy': -1500e3}
myVelSeries.subsetVel(bbox)
myVelSeries.loadRemote()       # download the subset
myVelSeries.toNetCDF('subset.nc')

# Compare to GPS
cv = cvPoints('gps_points.dat', epsg=3413)
stats = cv.timeSeriesStats(myVelSeries, 10, 5000)
```

## Notes

- **`useStack`**: `True` (default) loads each band as one contiguous array — fastest
  for downloads and repeated in-memory ops; `chunks` ignored. `False` tiles via
  dask/rioxarray — useful for sparse point access without `loadRemote()`.
- **`numWorkers=4`** is the recommended default; >8 gives diminishing returns and can
  hit NSIDC connection limits; drop to 2 if downloads fail with apparent
  file-not-found errors.
- **`url=True`** for `https://` NSIDC links; omit for local/network paths.
- **`readSpeed=False`** is recommended — `vv` is derived from `vx`/`vy` on the fly
  instead of fetching a separate file.
- **`loadRemote()`** must be called after `subsetVel`/`subsetImage` to actually
  download and cache the subset; before that, data are lazy/remote.
- All spatial coordinates are in the product's polar-stereographic projection
  (EPSG:3413 Greenland, EPSG:3031 Antarctica). Pass `sourceEPSG=4326` to `interp`
  to supply lat/lon directly.
- `nisardev/nisardev/nisarScaler.py` defines a second, older `nisarVel` class but is
  **not imported anywhere** (not in `__init__.py`, no internal references) — dead
  code left over from an earlier version; don't confuse it with `nisarVel.py`.
- Pre-built reference docs for each class live in `Documents/` (`Overview.md`,
  `nisarVel.md`, `nisarVelSeries.md`, `nisarImage.md`, `nisarImageSeries.md`,
  `cvPoints.md`) — these are more thorough usage guides than this file.
- For end-to-end examples, see `https://github.com/fastice/GrIMPNotebooks`.
