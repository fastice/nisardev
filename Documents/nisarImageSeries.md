# nisarImageSeries — Image Time Series

Stores a stack of SAR/optical images as a single xarray DataArray indexed by
`time` (midpoint), `time1` (start), and `time2` (end).  All layers in one
series must share the same image type (`image`, `sigma0`, or `gamma0`);
calibrated and uncalibrated products cannot be mixed.

---

## Construction

```python
import nisardev as nisar

myImageSeries = nisar.nisarImageSeries(numWorkers=4)
```

**Parameters:**
- `numWorkers` — parallel dask download threads (default: 2; recommended: 4)
- `imageType` — `'image'`, `'sigma0'`, or `'gamma0'` (auto-inferred from the first file if omitted)
- `verbose` — print progress messages

---

## Key attributes

| Attribute | Description |
|-----------|-------------|
| `xr` | Full xarray DataArray (band × time × y × x).  Inspect with `myImageSeries.xr`. |
| `subset` | Currently active spatial view.  Inspect with `myImageSeries.subset`. |
| `time` | Array of midpoint dates (`numpy.datetime64`).  Iterate to loop over layers. |
| `time1`, `time2` | Arrays of start / end dates for each layer |

---

## I/O

| Method | Description |
|--------|-------------|
| `readSeriesFromTiff(fileNames, url=False, useStack=True, bbox=None, index1=3, index2=4, dateFormat='%d%b%y', overviewLevel=-1, chunkSize=2048)` | Build a series from a list of COG file paths or NSIDC https URLs.  The image type is inferred from the first file.  `fileNames` typically comes from `myImageUrls.getCogs()`. |
| `readSeriesFromNetCDF(cdfFile)` | Load from a NetCDF file previously saved by `toNetCDF()`. |
| `loadRemote()` | Download and materialise the lazy / dask-backed subset into memory.  Call after `subsetImage` to cache the clipped region locally. |
| `toNetCDF(cdfFile)` | Save the *current subset* to NetCDF.  If no subset has been applied, the full dataset is written (potentially terabytes). *(inherited from nisarBase2D)* |

**Typical remote-access pattern:**
```python
# Image series
myImageSeries = nisar.nisarImageSeries(numWorkers=4)
myImageSeries.readSeriesFromTiff(myImageUrls.getCogs(), useStack=True)
myImageSeries.subsetImage(bbox)    # clip to region of interest
myImageSeries.loadRemote()         # download the subset

# sigma0 / gamma0 series (same pattern)
myGamma0Series = nisar.nisarImageSeries(numWorkers=4)
myGamma0Series.readSeriesFromTiff(myGamma0Urls.getCogs(), useStack=True)
myGamma0Series.subsetImage(bbox)
myGamma0Series.loadRemote()

# Save and reload
myImageSeries.toNetCDF('imageSeries.nc')
myImageSeriesReload = nisar.nisarImageSeries()
myImageSeriesReload.readSeriesFromNetCDF('imageSeries.nc')
myImageSeriesReload.loadRemote()
```

---

## Subsetting

| Method | Description |
|--------|-------------|
| `subsetImage(bbox)` | Clip all layers to `{'minx', 'miny', 'maxx', 'maxy'}` in metres.  Can be called repeatedly to change the region. |
| `subsetData(bbox)` | Alias for `subsetImage`. |
| `timeSliceImage(date1, date2)` | Return a new `nisarImageSeries` containing only layers between `date1` and `date2`. |
| `timeSliceData(date1, date2)` | Lower-level time-slice returning the base type. *(inherited from nisarBase2D)* |

---

## Accessing layers

| Method | Description |
|--------|-------------|
| `getMap(date, returnXR=False)` | Extract the image array from the layer nearest `date`. |
| `inspect(band='image', date=None, imgOpts={}, plotOpts={})` | Interactive Panel widget — map view + point-click time series.  No arguments needed for basic use. |

---

## Interpolation

| Method | Description |
|--------|-------------|
| `interp(x, y, date=None, units='m', returnXR=False, **kwargs)` | Bilinear interpolation at `x`, `y`.  If `date=None` all layers are interpolated; otherwise the nearest layer is used. |
| `interpGeo(x, y, myVars, ...)` | Lower-level interpolation. *(inherited from nisarBase2D)* |

---

## Geometry helpers *(inherited from nisarBase2D)*

`boundingBox`, `size`, `pixSize`, `origin`, `bounds`, `extent`, `outline`,
`xyGrid`, `sizeInPixels`, `getDomain` — see [nisarVel.md](nisarVel.md).

---

## Statistics *(inherited from nisarBase2D)*

| Method | Description |
|--------|-------------|
| `mean(skipna=True)` | Temporal mean.  Returns a `nisarImageSeries` with one layer. |
| `stdev(skipna=True)` | Temporal standard deviation. |
| `meanXY(returnXR=False)` | Spatial mean across x and y for each time. |
| `stdevXY(returnXR=False)` | Spatial standard deviation for each time. |
| `anomaly()` | Each layer minus the temporal mean. |
| `numberValid()` | Count non-NaN pixels per band per layer. |

**Example:**
```python
mean    = myImageSeries.mean()
anomaly = myImageSeries.anomaly()
sigma   = myImageSeries.stdev()

fig, axes = plt.subplots(1, 3, figsize=(24, 11))
for image, ax, title in zip([mean, sigma, anomaly], axes,
                              ['Mean', 'Sigma', 'Anomaly']):
    image.displayImageForDate(date='2020-02-28', ax=ax, cmap='gray',
                               midDate=False, masked=None,
                               vmin=-10, vmax=10, extend='both')
    ax.set_title(f'{title} — {ax.get_title()}', fontsize=18)
```

---

## Date helpers *(inherited from nisarBase2D)*

`parseDate`, `datetime64ToDatetime`.

---

## Display

### `displayImageForDate`

```python
myImageSeries.displayImageForDate(date=None, ax=None,
                                   vmin=None, vmax=None, percentile=100,
                                   autoScale=True, scale='linear',
                                   cmap='gray', backgroundColor=None,
                                   colorBar=True, colorBarLabel=None,
                                   colorBarPosition='right', colorBarSize='5%', colorBarPad=0.05,
                                   title=None, axisOff=False, midDate=True,
                                   masked=True, units='m', extend='both', wrap=None)
```

Display the image layer nearest to `date`.  If `date=None`, the first layer is shown.

| Key parameter | Notes |
|---------------|-------|
| `date` | `'YYYY-MM-DD'` string, `datetime`, or `None` |
| `percentile` | Clip colour limits (e.g. `99` clips the top and bottom 1%) |
| `autoScale` | `True` auto-scales from data; `False` uses `vmin`/`vmax` |
| `cmap` | Any matplotlib colourmap |
| `masked` | `True` treats zeros as no-data; `False` shows all values; `None` auto |
| `midDate` | `True` → title shows midpoint date; `False` → date range |
| `axisOff` | Remove x/y axis ticks |
| `colorBarPosition` | `'right'`, `'left'`, `'top'`, `'bottom'` |
| `extend` | `'both'`, `'min'`, `'max'`, `'neither'` |
| `colorBarSize` | Fraction of axes for colorbar width (e.g. `'3%'`) |

**Examples:**
```python
# Display one date
myImageSeries.displayImageForDate(date='2020-02-01', ax=ax, percentile=99)

# With explicit colour range, no axes, smaller colorbar
myImageSeries.displayImageForDate(date='2020-02-01', ax=ax, percentile=99,
                                   units='km', title='', colorBarSize='3%',
                                   axisOff=False)

# Statistics image (single time layer): grey colourmap, forced range
sigma.displayImageForDate(date='2020-02-28', ax=ax,
                           vmin=0, vmax=20, autoScale=False,
                           cmap='gray', midDate=False, extend='max')
```

### Profile and point plots

| Method | Description |
|--------|-------------|
| `plotProfile(x, y, *args, band=None, date=None, ax=None, units='m', **kwargs)` | Plot values along a transect for one date.  Typically called in a loop over `myImageSeries.time`.  Extra `*args`/`**kwargs` are passed to `ax.plot`. |
| `plotPoint(x, y, *args, band=None, ax=None, units='m', label=None, **kwargs)` | Plot the time series at a single map location. |
| `labelProfilePlot(ax, band=None, xLabel=None, yLabel=None, units='m', title=None, fontScale=1.0, plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a profile plot. |
| `labelPointPlot(ax, band=None, xLabel=None, yLabel=None, title=None, xLabel='Date', plotFontSize=10, titleFontSize=12)` | Apply axis labels and title to a point-vs-time plot. |

**Example — sigma0 and gamma0 time series at a point:**
```python
fig, axes = plt.subplots(1, 2, figsize=(19, 9))

# Map — mean image
mean.displayImageForDate(date='2020-02-28', ax=axes[0], percentile=99,
                          colorBarPosition='top', units='km')
axes[0].plot(xc, yc, 'r*', markersize=20)

# Point time series — two products on same axes
mySigma0Series.plotPoint(xc, yc, 'r*-', band='sigma0', label=r'$\sigma_o$',
                           units='km', ax=axes[1])
myGamma0Series.plotPoint(xc, yc, 'k*-', label=r'$\gamma_o$',
                           units='km', ax=axes[1])
myGamma0Series.labelPointPlot(axes[1],
                               title=r'$\sigma_o$ and $\gamma_o$ vs time',
                               xLabel='Radar Cross Section (dB)',
                               plotFontSize=14)
axes[1].legend(fontsize=16)
```

**Example — profile through several dates:**
```python
x = np.arange(450, 490, 0.2)   # km
y = np.full(x.shape, -1110.0)
bwr = plt.get_cmap('bwr', len(myGamma0Series.time))

fig, ax = plt.subplots(figsize=(15, 8))
for time, color in zip(myGamma0Series.time, range(len(myGamma0Series.time))):
    myGamma0Series.plotProfile(x, y, '.', date=time, ax=ax, units='km',
                                color=bwr(color), label=time.strftime('%Y-%m-%d'))
myGamma0Series.labelProfilePlot(ax, fontScale=1.3,
                                  title=r'$\gamma_o$ profile')
ax.legend(ncol=3, loc='lower left')
```

---

## Utility

| Method | Description |
|--------|-------------|
| `copy()` | Deep copy. *(inherited from nisarBase2D)* |
| `myVariables(imageType)` | Re-configure band names and dtypes after a type change. |
