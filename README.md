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

In an existing python virtual environment: `pip install https://github.com/fastice/nisardev.git@main`

## Release Notes

**0.0.10  2025-09-03**  Updated to remove stackstac and riostack dependencies and replace with non-chunked
                    alternative to gain performance improvements.
                    
**0.0.9  2025-08-20**  Updated to fix issues with NSIDC migration of data sets to cloud 


