#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from nisardev import nisarBase2D
import os
from datetime import datetime
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from osgeo import gdal
import xarray as xr


class nisarVel(nisarBase2D):
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
        # if readSpeed:
        #    myVars += ['vv']
        if useErrors:
            myVars += ['ex', 'ey']
        self.variables = myVars
        return myVars

    # ------------------------------------------------------------------------
    # Interpolation routines - to populate abstract methods from nisarBase2D
    # ------------------------------------------------------------------------

    def interp(self, x, y, **kwargs):
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
        return self.interpGeo(x, y, self.variables, **kwargs)

    # ------------------------------------------------------------------------
    # I/O Routines
    # ------------------------------------------------------------------------

    def readDataFromTiff(self, fileNameBase, useVelocity=True, useErrors=False,
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
        self.parseVelDatesFromFileName(fileNameBase)
        self.variables = self.myVariables(useVelocity, useErrors, readSpeed)
        if readSpeed:
            skip = []
        else:
            skip = ['vv']  # Force skip
        self.readXR(fileNameBase, url=url, masked=True, stackVar=stackVar,
                    time=self.midDate, skip=skip, time1=self.date1,
                    time2=self.date2)
        # compute speed rather than download
        if not readSpeed and useVelocity:
            dv = xr.DataArray(np.sqrt(np.square(self.vx) + np.square(self.vy)),
                              coords=[self.xr.y, self.xr.x], dims=['y', 'x'])
            dv = dv.expand_dims(dim=['time', 'band'])
            dv['band'] = ['vv']
            dv['time'] = self.xr['time']
            self.xr = xr.concat([self.xr, dv], dim='band', join='override',
                                combine_attrs='drop')
        self.fileNameBase = fileNameBase  # save filenameBase

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
        if useVelocity:
            self.vv = np.sqrt(np.square(self.vx) + np.square(self.vy))

    # ------------------------------------------------------------------------
    # Dates routines.
    # ------------------------------------------------------------------------

    def parseVelDatesFromFileName(self, fileNameBase, index1=4, index2=5,
                                  dateFormat='%d%b%y'):
        '''
        Parse the dates from the directory name the velocity products are
        stored in.
        Parameters
        ----------
           fileNameBase : str
            FileNameBase should be of the form
            pattern.*.abc or pattern*.
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        index1, index2 : ints, optional
            location of dates in filename with seperated by _
        dateFormat : str, optional
            format code to strptime
        Returns
        -------
        date: mid date.
            First and last dates from meta file.
        '''
        baseNamePieces = os.path.basename(fileNameBase).split('_')
        self.date1 = datetime.strptime(baseNamePieces[index1], dateFormat)
        self.date2 = datetime.strptime(baseNamePieces[index2], dateFormat)
        self.midDate = self.date1 + (self.date2 - self.date1) * 0.5
        #
        return self.midDate

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayVel(self, ax=None, component='vv',
                   plotFontSize=plotFontSize,
                   titleFontSize=titleFontSize,
                   labelFontSize=labelFontSize,
                   autoScale=True, axisOff=False,
                   vmin=0, vmax=7000, percentile=100, **kwargs):
        '''
        Use matplotlib to show velocity in a single subplot with a color
        bar. Clip to absolute max set by maxv, though in practives percentile
        will clip at a signficantly lower value.
        Parameters
        ----------
        ax : matplotlib axis, optional
            axes for plot. The default is None.
        component : str, optional
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
        # Compute display bounds
        if autoScale:
            maxVel = min(np.percentile(
                getattr(self, component)[np.isfinite(self.vv)], percentile),
                vmax)
            vmax = math.ceil(maxVel/100.) * 100.
            minVel = max(np.percentile(
                getattr(self, component)[np.isfinite(self.vv)],
                100-percentile), vmin)
            vmin = math.floor(minVel/100.) * 100.
        # Display data
        self.displayVar(component, ax=ax, plotFontSize=self.plotFontSize,
                        labelFontSize=self.labelFontSize,
                        colorBarLabel='Speed (m/yr)', vmax=vmax, vmin=vmin,
                        **kwargs)
