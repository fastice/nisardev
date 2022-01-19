#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from nisarfunc import readGeoTiff
from nisarfunc import nisarBase2D, parseDatesFromMeta, parseDatesFromDirName
import os
from datetime import datetime
import xarray as xr
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from osgeo import gdal


class nisarVel(nisarBase2D):
    ''' This class creates objects to contain nisar velocity and/or error maps.
    The data can be pass in on init, or read from a geotiff.

    The variables that are used are specified with useVelocity, useErrrors,
    and, noSpeedand (see nisarVel.readDatafromTiff).

    The variables used are returned as a list (e.g., ["vx","vy"])) by
    nisarVel.myVariables(useVelocity=True, useErrors=False, noSpeed=True). '''

    labelFontSize = 16  # Font size for plot labels
    plotFontSize = 15  # Font size for plots
    legendFontSize = 15  # Font size for legends

    def __init__(self, vx=None, vy=None, v=None, ex=None, ey=None, e=None,
                 sx=None, sy=None, x0=None, y0=None, dx=None, dy=None,
                 epsg=None, useXR=False, verbose=True):
        '''
        Instantiate nisarVel object. Possible bands are 'vx', 'vy','v', 'ex',
        'ey', 'e'

        Parameters
        ----------
        vx : 2d array, optional
            Initialize vx values. The default is None.
        vy : 2d array, optional
            Initialize vy values. The default is None.
        v : 2d array, optional
            Initialize v values. The default is None.
        ex : 2d array, optional
            Initialize ex values. The default is None.
        ey : 2d array, optional
            Initialize ey values. The default is None.
        e : 2d array, optional
            Initialize e values. The default is None.
        sx : int, optional
            Set x size in pixels. The default is None.
        sy : int, optional
            Set y size in pixels. The default is None.
        x0 : float, optional
            x coordinate of origin (ll) in m. The default is None.
        y0 : float, optional
            y coordinate of origin (ll) in m. The default is None.
        dx : float, optional
            pixel size for x coordinate in m. The default is None.
        dy : float, optional
            pixel size for y coordinate in m. The default is None.
        epsg : int, optional
            epsg code. The default is None.
        useXR : bool, optional
            Use xarray rather than nparray (not tested). The default is False.
        verbose : bool, optional
            Increase level of informational messages. The default is True.
        Returns
        -------
        None.
        '''
        nisarBase2D.__init__(self, sx=sx, sy=sy, x0=x0, y0=y0, dx=dx, dy=dy,
                             epsg=epsg, useXR=useXR)
        self.vx, self.vy, self.vv, self.ex, self.ey = vx, vy, v, ex, ey
        self.vxInterp, self.vyInterp, self.vvInterp = None, None, None
        self.exInterp, self.eyInterp = None, None
        self.verbose = verbose
        self.noDataDict = {'vx': -2.0e9, 'vy': -2.0e9, 'vv': -1.0,
                           'ex': -1.0, 'ey': -1.0}
        self.gdalType = gdal.GDT_Float32  # data type for velocity products

    def myVariables(self, useVelocity, useErrors, noSpeed=False):
        '''
        Based on the input flags, this routine determines which velocity/error
        fields that an instance will contain.
        Parameters
        ----------
        useVelocity : bool
            Include 'vx', 'vy', and 'vv'.
        useErrors : bool
            Include 'ex', and 'ey'.
        noSpeed : bool, optional
            If false, don't include vv. The default is False.
        Returns
        -------
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        '''
        myVars = []
        if useVelocity:
            myVars += ['vx', 'vy']
        if not noSpeed:
            myVars += ['vv']
        if useErrors:
            myVars += ['ex', 'ey']
        return myVars

    # ------------------------------------------------------------------------
    # Interpolation routines - to populate abstract methods from nisarBase2D
    # ------------------------------------------------------------------------
    def _setupInterp(self, useVelocity=True, useErrors=False):
        '''
        Setup interpolaters for velocity (vx,vy for useVelocity) and
        error (ex,ey for useErrors).
        Parameters
        ----------
        useVelocity : bool, optional
            Setup velocity interpolation if True. The default is True.
        useErrors : TYPE, optional
            Setup error interpolation if True. The default is False.
        Returns
        -------
        None.
        '''
        # select variables
        myVars = self.myVariables(useVelocity, useErrors)
        self._setupInterpolator(myVars)

    def interp(self, x, y, useVelocity=True, useErrors=False, noSpeed=False):
        '''
        Call appropriate interpolation method to interpolate myVars at x, y
        points.

        Parameters
        ----------
        x : nparray
            DESCRIPTION.
        y : nparray
            DESCRIPTION.
        useVelocity : bool, optional
            Interpolate velocity if True. The default is True.
        useErrors : bool, optional
            Interpolate errors if True. The default is False.
        noSpeed : bool, optional
            Skip interpolate of speed if True. The default is False.
        Returns
        -------
        npArray
            interpolate results for [nbands, npts].
        '''
        # determine velocity variables to interp
        myVars = self.myVariables(useVelocity, useErrors, noSpeed=noSpeed)
        # Set up interpolator if not already done
        for myVar in myVars:
            if getattr(self, f'{myVar}Interp') is None:
                self._setupInterp(useVelocity=useVelocity, useErrors=useErrors)
                break  # One call will setup all myVars
        return self.interpGeo(x, y, myVars)

    # ------------------------------------------------------------------------
    # I/O Routines
    # ------------------------------------------------------------------------

    def dataFileNames(self, fileNameBase, myVars):
        '''
        Compute the file names that need to be read. Appends tif.
        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
            pattern.*.abc or pattern*.
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        Returns
        -------
        fileNames : ['str'...]
            List of filenames.
        '''
        #
        fileNames = []
        if '*' not in fileNameBase:  # in this case append myVar as suffix
            fileNameBase = f'{fileNameBase}.*'
        # Form names
        for myVar in myVars:
            fileNames.append(f'{fileNameBase.replace("*", myVar)}.tif')
        return fileNames

    def readDataFromTiff(self, fileNameBase, useVelocity=True, useErrors=False,
                         noSpeed=True, useXR=False):
        '''
        read in a tiff product fileNameBase.*.tif. If
        useVelocity=True read velocity (e.g, fileNameBase.vx(vy).tif)
        useErrors=True read errors (e.g, fileNameBase.ex(ey).tif)
        useSpeed=True read speed (e.g, fileNameBase.vv.tif) otherwise
        compute from vx,vy.

        Files can be read as np arrays of xarrays (useXR=True, not well tested)

        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
            pattern.*.abc or pattern*.
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        useVelocity : bool, optional
            Interpolate velocity if True. The default is True.
        useErrors : bool, optional
            Interpolate errors if True. The default is False.
        noSpeed : bool, optional
            Skip interpolate of speed if True. The default is False.
        useXR : bool, optional
            read data as xarrays. The default is False.
        Returns
        -------
        None.
        '''
        #  get the values that match the type
        self.useXR = useXR
        myVars = self.myVariables(useVelocity, useErrors, noSpeed=noSpeed)
        fileNames = self.dataFileNames(fileNameBase, myVars)
        # loop over arrays and file names to read data
        for fileName, myVar in zip(fileNames, myVars):
            start = datetime.now()
            if useXR:
                myArray = xr.open_rasterio(fileName)
                myArray = myArray.where(myArray > self.noDataDict[myVar])
            else:
                myArray = readGeoTiff(fileName, noData=self.noDataDict[myVar])
            setattr(self, myVar, myArray)
            setattr(self, f'{myVar}Interp', None)  # flush any prior interp f
            if self.verbose:
                print(f'read time {datetime.now()-start}')
        #
        self.readGeodatFromTiff(fileNames[0])
        # compute mag for velocity and errors
        if useVelocity:
            self.vv = np.sqrt(np.square(self.vx) + np.square(self.vy))
        self.fileNameBase = fileNameBase  # save filenameBase

    def writeDataToTiff(self, fileNameBase, useVelocity=True, useErrors=False,
                        noSpeed=True, overviews=None, predictor=1,
                        noDataDefault=None):
        '''
        Write a velocity product with useVelocity/useErrors/noSpeed to
        determine which bands are written.
        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
            pattern.*.abc or pattern*.
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        useVelocity : bool, optional
            Use the velocity variables. The default is True.
        useErrors : bool, optional
            Use the error variables. The default is False.
        noSpeed : bool, optional
            Don't use speed. The default is True.
        overviews : list, optional
            List of overvew sizes (e.g., [2, 4, 8, 16]). The default is None.
        predictor : int, the default is 1
            Predictor used by gdal for compression. The default is 1.
        noDataDefault : optional - same as data, optional
            No data value. The default is None.
        Returns
        -------
        None.
        '''
        # Get var names and create files names from template
        myVars = self.myVariables(useVelocity, useErrors, noSpeed=noSpeed)
        fileNames = self.dataFileNames(fileNameBase, myVars)
        # loop and write each band in myVars
        for fileName, myVar in zip(fileNames, myVars):
            self.writeCloudOptGeo(fileName, myVar, gdalType=self.gdalType,
                                  overviews=None, predictor=1,
                                  noData=self.noDataDict[myVar])
    # ------------------------------------------------------------------------
    # Dates routines.
    # ------------------------------------------------------------------------

    def parseVelDatesFromMeta(self, metaFile=None):
        '''
        Parse dates from a GIMP meta file.
        Parameters
        ----------
        metaFile : str, optional
            metaFile name, if not specified use basename. The default is None.
        Returns
        -------
        dates : [datetime, datetime]
            First and last dates from meta file.
        '''
        if metaFile is None:
            metaFile = self.fileName + '.meta'
        #
        return parseDatesFromMeta(metaFile)

    def parseVelDatesFromDirName(self, dirName=None, divider='.',
                                 dateTemplate='Vel-%Y-%m-%d.%Y-%m-%d'):
        '''
        Parse the dates from the directory name the velocity products are
        stored in.
        Parameters
        ----------
        dirName : str, optional
            Name of the directory. The default is None.
        divider : str, optional
            character that divides up name xyz.date1.date2. The default is '.'.
        dateTemplate : str, optional
            Template for dir name. The default is 'Vel-%Y-%m-%d.%Y-%m-%d'.
        Returns
        -------
        dates : [datetime, datetime]
            First and last dates from meta file.
        '''
        if dirName is None:
            # Special case to temove release subdir if it exists.
            prodDir = self.fileNameBase.replace('/release', '')
            dirName = os.path.dirname(prodDir).split('/')[-1]
        return parseDatesFromDirName(dirName, dateTemplate, divider)

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------
    def maxPlotV(self, maxv=7000):
        '''
        Uses 99 percentile range for maximum value, with maxv as an upper
        limit. Default in most cases will be much greater than actual max from
        the percentiles.
        Parameters
        ----------
        maxv : float or int, optional
            max velocity. The default is 7000.
        Returns
        -------
        float
            Upper limit on velocity for plotting.
        '''
        maxVel = min(np.percentile(self.vv[np.isfinite(self.vv)], 99), maxv)
        return math.ceil(maxVel/100.)*100.  # round up to nearest 100

    def displayVel(self, axImage=None, fig=None, maxv=7000):
        '''
        Use matplotlib to show velocity in a single subplot with a color
        bar. Clip to absolute max set by maxv, though in practives percentile
        will clip at a signficantly lower value.
        Parameters
        ----------
        fig : matplot lib fig, optional
            Pass in an existing figure. The default is None.
        maxv : float or int, optional
            max velocity. The default is 7000.
        Returns
        -------
        fig : matplot lib fig
            Figure used for plot.
        axImage : matplot lib ax
            Axis used for plot.
        '''
        if fig is None:
            sx, sy = self.sizeInPixels()
            fig = plt.figure(constrained_layout=True,
                             figsize=(10.*sx/sy, 10.))
        #
        # b = self.boundsInKm()  # bounds for plot window
        if axImage is None:
            axImage = fig.add_subplot(111)
        divider = make_axes_locatable(axImage)  # Create space for colorbar
        cbAx = divider.append_axes('right', size='5%', pad=0.05)
        pos = axImage.imshow(self.vv, origin='lower', vmin=0,
                             vmax=self.maxPlotV(maxv=maxv),
                             extent=self.extentInKm())
        cb = fig.colorbar(pos, cax=cbAx, orientation='vertical', extend='max')
        cb.set_label('Speed (m/yr)', size=self.labelFontSize)
        cb.ax.tick_params(labelsize=self.plotFontSize)
        axImage.set_xlabel('X (km)', size=self.labelFontSize)
        axImage.set_ylabel('Y (km)', size=self.labelFontSize)
        axImage.tick_params(axis='x', labelsize=self.plotFontSize)
        axImage.tick_params(axis='y', labelsize=self.plotFontSize)
        return fig, axImage
