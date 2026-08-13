# nisarImage — Single SAR/Optical Image

Stores one georeferenced image.  The active band is one of `image` (DN,
uint8), `sigma0` (backscatter in dB, float32), or `gamma0` (terrain-corrected
backscatter in dB, float32).  The image type is detected automatically from
the filename or set explicitly.

---

## Construction

```python
import nisardev as nisar

myImage = nisar.nisarImage(numWorkers=4)
```

**Parameters:**
- `numWorkers` — parallel dask download threads (default: 2; recommended: 4)
- `imageType` — `'image'`, `'sigma0'`, or `'gamma0'` (auto-detected from filename if omitted)
- `verbose` — print progress messages

---

## Key attributes

| Attribute | Description |
|-----------|-------------|
| `xr` | Full xarray DataArray |
| `subset` | Currently active view after subsetting |
| `image` | Image pixel values (dask or numpy after `loadRemote()`) |
| `time` | Midpoint date (`numpy.datetime64`) |
| `time1`, `time2` | Start and end dates of the measurement window |
| `sx`, `sy` | Spatial width and height in pixels — useful for computing aspect ratios |

---

## I/O

| Method | Description |
|--------|-------------|
| `readDataFromTiff(fileNameBase, url=False, useStack=True, imageType=None, bbox=None, index1=3, index2=4, dateFormat='%d%b%y', overviewLevel=-1, chunkSize=2048, masked=True)` | Open a COG image product.  `imageType` is auto-detected from the filename if `None`.  `overviewLevel` selects a pyramid level for reduced resolution (`-1` = full; `4` → 2^5 × native ≈ 800 m for 25 m products). |
| `readDataFromNetCDF(cdfFile)` | Read a NetCDF file previously saved by `toNetCDF()`. |
| `loadRemote()` | Download and materialise the lazy data into memory.  Call after subsetting. |
| `toNetCDF(cdfFile)` | Save to NetCDF. *(inherited from nisarBase2D)* |
| `writeCloudOptGeo(tiffRoot, full=False, myVars=None)` | Write as a Cloud-Optimised GeoTIFF. *(inherited from nisarBase2D)* |

**Example — overview image at reduced resolution:**
```python
# overviewLevel=4 → ~800 m resolution for 25 m base data
myOverview = nisar.nisarImage(numWorkers=4)
myOverview.readDataFromTiff(myImageUrls.getCogs()[0], overviewLevel=4)
myOverview.loadRemote()

# Use sx/sy for a correctly proportioned figure
height = 3
width  = height * myOverview.sx / myOverview.sy
fig, ax = plt.subplots(figsize=(width, height))
myOverview.displayImage(ax=ax, percentile=99, units='km',
                         cmap=plt.cm.gray.with_extremes(bad=(0.4, 0.4, 0.4)),
                         backgroundColor=(0.9, 0.9, 0.9))
```

---

## Subsetting

| Method | Description |
|--------|-------------|
| `subsetImage(bbox)` | Clip to `{'minx', 'miny', 'maxx', 'maxy'}` in metres. |
| `subsetData(bbox)` | Alias for `subsetImage`. |

---

## Image type detection

| Method | Description |
|--------|-------------|
| `detectImageType(fileNameBase)` | Infer `'image'`, `'sigma0'`, or `'gamma0'` from the filename stem. |
| `myVariables(imageType)` | Set internal band names and dtypes for the given type. |

---

## Interpolation

| Method | Description |
|--------|-------------|
| `interp(x, y, units='m', returnXR=False, **kwargs)` | Bilinear interpolation at `x`, `y`. Returns `[band_values]`. |
| `interpGeo(x, y, myVars, ...)` | Lower-level interpolation for a named list of bands. *(inherited from nisarBase2D)* |

---

## Date helpers

| Method | Description |
|--------|-------------|
| `parseImageDatesFromFileName(fileNameBase, index1=3, index2=4, dateFormat='%d%b%y')` | Parse `time1` and `time2` from the filename. |
| `parseDate(date, ...)` | Convert `'YYYY-MM-DD'` or `datetime`. *(inherited from nisarBase2D)* |
| `datetime64ToDatetime(date64)` | Convert `numpy.datetime64` to Python `datetime`. *(inherited from nisarBase2D)* |

---

## Geometry helpers *(inherited from nisarBase2D)*

`boundingBox`, `size`, `pixSize`, `origin`, `bounds`, `extent`, `outline`,
`xyGrid`, `sizeInPixels`, `getDomain` — see [nisarVel.md](nisarVel.md).

---

## Statistics *(inherited from nisarBase2D)*

`mean`, `stdev`, `meanXY`, `stdevXY`, `anomaly`, `numberValid`.

---

## Display

### `displayImage`

```python
myImage.displayImage(date=None, ax=None,
                      vmin=None, vmax=None, percentile=100,
                      autoScale=True, scale='linear',
                      cmap='gray', backgroundColor=None,
                      colorBar=True, colorBarLabel=None,
                      colorBarPosition='right', colorBarSize='5%', colorBarPad=0.05,
                      title=None, axisOff=False, midDate=True,
                      masked=True, units='m', extend='both', wrap=None)
```

Display the image using a colour map.

| Key parameter | Notes |
|---------------|-------|
| `percentile` | Clip colour limits (e.g. `99` ignores the top 1%) |
| `autoScale` | `True` auto-scales from data; `False` uses `vmin`/`vmax` |
| `scale` | `'linear'` or `'log'` |
| `cmap` | Any matplotlib colourmap.  Use `plt.cm.gray.with_extremes(bad=(0.4, 0.4, 0.4))` to shade masked pixels differently. |
| `backgroundColor` | Colour for masked / no-data pixels (RGB tuple or colour name) |
| `midDate` | `True` → title shows midpoint; `False` → date range |
| `axisOff` | Remove x/y axis ticks |
| `masked` | `True` treats zeros as no-data; `False` renders raw values; `None` uses auto-detection |
| `colorBarPosition` | `'right'`, `'left'`, `'top'`, `'bottom'` |
| `extend` | `'both'`, `'min'`, `'max'`, `'neither'` |
| `colorBarSize` | Fraction of axes size for the colorbar (e.g. `'2%'`) |
| `colorBarPad` | Padding between image and colorbar |

**Examples:**
```python
import matplotlib.pyplot as plt

# Basic display
myImage.displayImage(ax=ax, percentile=99, units='km')

# With custom grey colourmap and background for no-data
myImage.displayImage(ax=ax, percentile=99, units='km',
                      cmap=plt.cm.gray.with_extremes(bad=(0.4, 0.4, 0.4)),
                      backgroundColor=(0.9, 0.9, 0.9),
                      colorBarPosition='bottom', colorBarPad=0.25,
                      colorBarSize='2%', midDate=False)

# No colorbar, axes off (e.g. for an inset)
myImage.displayImage(ax=axInset, colorBar=False, axisOff=True,
                      units='km', title='', extend='both')
```

### Profile and point plots

| Method | Description |
|--------|-------------|
| `plotProfile(x, y, *args, band=None, ax=None, date=None, units='m', midDate=True, **kwargs)` | Plot values along a transect. Extra args go to `ax.plot`. |
| `plotPoint(x, y, *args, band=None, ax=None, **kwargs)` | Mark a single point on the axes. |
| `labelProfilePlot(ax, band=None, xLabel=None, yLabel=None, units='m', title=None, fontScale=1.0, plotFontSize=10, titleFontSize=12)` | Label a profile plot. |
| `labelPointPlot(ax, band=None, xLabel=None, yLabel=None, title=None, plotFontSize=10, titleFontSize=12)` | Label a point-vs-time plot. |

---

## Utility

| Method | Description |
|--------|-------------|
| `copy()` | Deep copy. *(inherited from nisarBase2D)* |
| `timeSliceData(date1, date2)` | Return a copy restricted to a date range. *(inherited from nisarBase2D)* |
