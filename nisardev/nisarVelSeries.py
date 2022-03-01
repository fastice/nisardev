#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from nisardev import nisarBase2D, nisarVel
import os
from datetime import datetime
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from osgeo import gdal
import xarray as xr
from dask.diagnostics import ProgressBar


class nisarVelSeries(nisarBase2D):
    ''' This class creates objects to contain nisar velocity and/or error maps.
    The data can be pass in on init, or read from a geotiff.

    The variables that are used are specified with useVelocity, useErrrors,
    and, readSpeed and (see nisarVel.readDatafromTiff).

    The variables used are returned as a list (e.g., ["vx","vy"])) by
    nisarVel.myVariables(useVelocity=True, useErrors=False, readSpeed=False).
    '''

    labelFontSize = 14  # Font size for plot labels
    plotFontSize = 12  # Font size for plots
    legendFontSize = 12  # Font size for legends
    titleFontSize = 15  # Font size for legends

    def __init__(self, verbose=True):
        '''
        Instantiate nisarVel object. Possible bands are 'vx', 'vy','v', 'ex',
        'ey', 'e'
        Parameters
        ----------
        verbose : bool, optional
            Increase level of informational messages. The default is True.
        Returns
        -------
        None.
        '''
        nisarBase2D.__init__(self)
        self.vx, self.vy, self.vv, self.ex, self.ey = [None] * 5
        self.variables = None
        self.verbose = verbose
        self.noDataDict = {'vx': -2.0e9, 'vy': -2.0e9, 'vv': -1.0,
                           'ex': -1.0, 'ey': -1.0}
        self.gdalType = gdal.GDT_Float32  # data type for velocity products
        self.nLayers = 0  # Number of time layers

    def myVariables(self, useVelocity, useErrors, readSpeed=False):
        '''
        Based on the input flags, this routine determines which velocity/error
        fields that an instance will contain.
        Parameters
        ----------
        useVelocity : bool
            Include 'vx', 'vy', and 'vv'.
        useErrors : bool
            Include 'ex', and 'ey'.
        readSpeed : bool, optional
            If false, don't include vv. The default is False.
        Returns
        -------
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        '''
        myVars = []
        if useVelocity:
            myVars += ['vx', 'vy', 'vv']
        if useErrors:
            myVars += ['ex', 'ey']
        self.variables = myVars
        return myVars

    # ------------------------------------------------------------------------
    # Interpolation routines - to populate abstract methods from nisarBase2D
    # ------------------------------------------------------------------------

    def interp(self, x, y, date=None, units='m', returnXR=False, **kwargs):
        '''
        Call appropriate interpolation method to interpolate myVars at x, y
        points.

        Parameters
        ----------
        x : nparray
            DESCRIPTION.
        y : nparray
            DESCRIPTION.
        Returns
        -------
        npArray
            interpolate results for [nbands, npts].
        '''
        if not self._checkUnits(units):
            return
        return self.interpGeo(x, y, self.variables, date=date, units=units,
                              returnXR=returnXR, **kwargs)

    def getMap(self, date, returnXR=False):
        '''
        Extract map closest in time to date

        Parameters
        ----------
        date : datetime, or str "YYYY-MM-DD"
            date closest to desired layer
        Returns
        -------
        vx, vy
        '''
        date = self.parseDate(date)  # Convert str to datetime if needed
        result = self.subset.sel(time=date, method='nearest')
        # return either xr or np data
        if returnXR:
            return result
        else:
            return [x.data for x in result] + [self.datetime64ToDatetime(
                result[0].time['time'].data)]
    # ------------------------------------------------------------------------
    # I/O Routines
    # ------------------------------------------------------------------------

    def readSeriesFromTiff(self, fileNames, useVelocity=True, useErrors=False,
                           readSpeed=False, url=False, stackVar=None,
                           index1=4, index2=5, dateFormat='%d%b%y'):
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
        readSpeed : bool, optional
            Read speed (.vv) if True. The default is False.
        url : bool, optional
            Read data from url
        stacVar : diction, optional
            for stackstac {'bounds': [], 'resolution': res, 'epsg': epsg}
        index1, index2 : location of dates in filename with seperated by _
        dateFormat : format code to strptime
        Returns
        -------
        None.
        '''
        self.variables = self.myVariables(useVelocity, useErrors, readSpeed)
        self.velMaps = []
        with ProgressBar():
            for fileName in fileNames:
                myVel = nisarVel()
                myVel.readDataFromTiff(fileName, useVelocity=useVelocity,
                                       useErrors=useErrors,
                                       readSpeed=readSpeed,
                                       url=url, stackVar=stackVar,
                                       index1=index1, index2=index2,
                                       dateFormat=dateFormat)
                self.velMaps.append(myVel)
        bBox = myVel.boundingBox(units='m')
        # Combine individual bands
        self.nLayers = len(fileNames)
        self.xr = xr.concat([x.xr for x in self.velMaps], dim='time',
                            join='override', combine_attrs='drop')
        # ensure that properly sorted in time
        self.xr = self.xr.sortby(self.xr.time)
        # This forces a subset=entire image, which will trigger initialization
        # Spatial parameters derived from the first velMap
        self.subSetVel(bBox)
        self.xr = self.xr.rename('VelocitySeries')
        self.time = [self.datetime64ToDatetime(x)
                     for x in self.xr.time.data]
        self.time1 = [self.datetime64ToDatetime(x)
                      for x in self.xr.time1.data]
        self.time2 = [self.datetime64ToDatetime(x)
                      for x in self.xr.time2.data]

    def _addSpeedSeries(self):
        ''' Add speed if only have vx and vy '''
        dv = xr.DataArray(np.sqrt(np.square(self.vx) + np.square(self.vy)),
                          coords=[self.xr.time, self.xr.y, self.xr.x],
                          dims=['time', 'y', 'x'])
        # add band for vv
        dv = dv.expand_dims(dim=['band'])
        dv['band'] = ['vv']
        dv['time'] = self.xr['time']
        dv['name'] = self.xr['name']
        # Add to vx, vy xr
        self.xr = xr.concat([self.xr, dv], dim='band', join='override',
                            combine_attrs='drop')
        #
        if 'vv' not in self.variables:
            self.variables.append('vv')
        self._mapVariables()

    def readSeriesFromNetCDF(self, cdfFile):
        '''
        Read a cdf file previously saved by a velSeries instance.
        Parameters
        ----------
        cdfFile : str
            netcdf file name.
        Returns
        -------
        None.

        '''
        # Read the date
        self.readFromNetCDF(cdfFile)
        # Initialize various variables.
        self.nLayers = len(self.xr.time.data)
        self.variables = list(self.xr.band.data)
        if 'vv' not in self.variables and 'vx' in self.variables and \
                'vy' in self.variables:

            self._addSpeedSeries()
            self.subset = self.xr

        # get times
        self.time = [self.datetime64ToDatetime(x) for x in self.xr.time.data]
        self.time1 = [self.datetime64ToDatetime(x) for x in self.xr.time1.data]
        self.time2 = [self.datetime64ToDatetime(x) for x in self.xr.time2.data]

    def subSetVel(self, bbox, useVelocity=True):
        ''' Subset dataArray to a bounding box
        Parameters
        ----------
        bbox : dict
            {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        useVelocity: bool, optional
            compute speed from vx, vy
        Returns
        -------
        None.
        '''
        self.subSetData(bbox)

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayVelForDate(self, date=None, ax=None, band='vv',
                          plotFontSize=plotFontSize,
                          titleFontSize=titleFontSize,
                          labelFontSize=labelFontSize,
                          autoScale=True, axisOff=False,
                          vmin=0, vmax=7000, percentile=100,
                          colorBarLabel='Speed (m/yr)', **kwargs):
        '''
         Use matplotlib to show a velocity layer selected by date.
         Clip to absolute max set by maxv, though in practives percentile
         will clip at a signficantly lower value.

        Parameters
        ----------
        date : str or datetime
            Approximate date to plot (nearest selected).
        ax : matplotlib axis, optional
            axes for plot. The default is None.
        band : str, optional
            component to plot (any of loaded variables). The default is 'vv'.
        plotFontSize : int, optional
            Font size for plot. The default is plotFontSize.
        titleFontSize : TYPE, optional
            Font size for title. The default is titleFontSize.
        labelFontSize : TYPE, optional
            Font size for labels. The default is labelFontSize.
        autoScale : bool, optional
            Autoscale plot range,but not exceed vmin,vmax. The default is True.
        axisOff : TYPE, optional
            Turn axes off. The default is False.
        vmax : number, optional
            max velocity to display. The default is 7000.
        vmin : number, optional
            min velocity to display. The default is 0.
        percentile : number, optional
            percentile to clip display at. The default is 100
        **kwargs : dict
            kwargs passed to imshow.
        Returns
        -------
        None.

        '''
        # Compute auto scale params
        if autoScale:
            vmin, vmax = self.autoScaleRange(band, date, vmin, vmax,
                                             percentile)

        # Create plot
        self.displayVar(band, date=date, ax=ax, plotFontSize=plotFontSize,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        axisOff=axisOff,
                        colorBarLabel=colorBarLabel, vmax=vmax, vmin=vmin,
                        **kwargs)

    @classmethod
    def reproduce(cls):
        ''' Create and return a new instance of velocitySeries '''
        return cls()
