# nisardev
Tools for nisar cal/val activities and cryo notebooks

For examples of how this library is used refer to notebooks in https://github.com/fastice/GrIMPNotebooks

## Documentation

- [Overview](Documents/Overview.md) — package overview and class summary
- [nisarImage](Documents/nisarImage.md) — single-scene NISAR image class
- [nisarImageSeries](Documents/nisarImageSeries.md) — time-series of NISAR images
- [nisarVel](Documents/nisarVel.md) — NISAR velocity product class
- [nisarVelSeries](Documents/nisarVelSeries.md) — time-series of NISAR velocity products
- [cvPoints](Documents/cvPoints.md) — calibration/validation point tools

## Installation

`nisardev` relies on conda-forge packages (gdal, rasterio, rioxarray, holoviews, panel, qgis, ...)
that pip cannot install cleanly, so build the conda environment first and then pip install the
package into it:

```
conda env create -f environment.yml      # creates env "nisardev"; tested on python 3.10, 3.12, 3.14
conda activate nisardev
```

`environment.yml` (a copy of the one in
[GrIMPNotebooks/binder](https://github.com/fastice/GrIMPNotebooks/blob/master/binder/environment.yml))
already pip-installs `nisardev`, `grimpfunc`, and `grimpqgis` from GitHub. To install or update
`nisardev` by hand in an existing environment:

```
pip install git+https://github.com/fastice/nisardev.git@main
```

Notes:
- The QGIS python bindings are only importable inside QGIS or with
  `$CONDA_PREFIX/share/qgis/python` on `PYTHONPATH`; `import qgis` failing in a plain python
  session is expected.
- A stale `~/.local/lib/pythonX.Y/site-packages` can shadow the conda environment and produce
  odd import errors; test with `PYTHONNOUSERSITE=1 python ...` if that happens.
- Remote NSIDC reads need an Earthdata login; see the `NSIDCLoginNotebook` in GrIMPNotebooks.

## Release Notes

**0.0.11  2026-09-10**  Fixed `displayVel` on matplotlib >= 3.11 (`cm.get_cmap` removed) and made
                    `inspect()` self-contained (`hvplot.xarray` is now imported by nisardev).
                    Added `environment.yml` (adds `rio-stac`, required by `grimpfunc`).


**0.0.10  2025-09-03**  Updated to remove stackstac and riostack dependencies and replace with non-chunked
                    alternative to gain performance improvements.
                    
**0.0.9  2025-08-20**  Updated to fix issues with NSIDC migration of data sets to cloud 


