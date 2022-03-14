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
from datetime import datetime, timedelta
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        self.dtype = 'float32'

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

    def _addSpeed(self):
        ''' Add speed if only have vx and vy '''
        # Compute speed
        dv = xr.DataArray(np.sqrt(np.square(self.vx) + np.square(self.vy)),
                          coords=[self.xr.y, self.xr.x], dims=['y', 'x'])
        # setup band as vv
        dv = dv.expand_dims(dim=['time', 'band'])
        dv['band'] = ['vv']
        dv['time'] = self.xr['time']
        dv['name'] = self.xr['name']
        # Add to existing xr with vx and vy
        self.xr = xr.concat([self.xr, dv], dim='band', join='override',
                            combine_attrs='drop')
        #
        if 'vv' not in self.variables:
            self.variables.append('vv')
        self._mapVariables()

    def readDataFromTiff(self, fileNameBase, useVelocity=True, useErrors=False,
                         readSpeed=False, url=False, stackVar=None,
                         index1=4, index2=5, dateFormat='%d%b%y',
                         overviewLevel=None, masked=True):
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
        overviewLevel: int
            Overview (pyramid) level to read: None->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level
        Returns
        -------
        None.
        '''
        self.parseVelDatesFromFileName(fileNameBase, index1=index1,
                                       index2=index2, dateFormat=dateFormat)
        self.variables = self.myVariables(useVelocity, useErrors, readSpeed)
        if readSpeed:
            skip = []
        else:
            skip = ['vv']  # Force skip
        self.readXR(fileNameBase, url=url, masked=True, stackVar=stackVar,
                    time=self.midDate, skip=skip, time1=self.date1,
                    time2=self.date2, overviewLevel=overviewLevel)
        # compute speed rather than download
        if not readSpeed and useVelocity:
            self._addSpeed()
        #
        self.xr = self.xr.rename('VelocityMap')
        self.fileNameBase = fileNameBase  # save filenameBase
        # force intial subset to entire image
        self.subSetData(self.boundingBox(units='m'))

    def readDataFromNetCDF(self, cdfFile):
        '''
        Read a cdf file previously saved by a nisarVel instance.
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
            self._addSpeed()
            self.subset = self.xr
        # set times
        self.time = [np.datetime64(self.xr.time.item(), 'ns')]
        self.time1 = [np.datetime64(self.xr.time1.item(), 'ns')]
        self.time2 = [np.datetime64(self.xr.time2.item(), 'ns')]

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
        if index2 is not None:
            self.date2 = datetime.strptime(baseNamePieces[index2], dateFormat)
        else:
           # assume monthly
           tmp = self.date1 + timedelta(days=32)
           self.date2 = tmp - timedelta(days=tmp.day)    
        self.midDate = self.date1 + (self.date2 - self.date1) * 0.5
        #
        return self.midDate

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayVel(self, ax=None, band='vv',
                   plotFontSize=plotFontSize,
                   titleFontSize=titleFontSize,
                   labelFontSize=labelFontSize,  colorBar=True,
                   scale='linear', axisOff=False, midDate=True,
                   vmin=0, vmax=7000, percentile=100, **kwargs):
        '''
        Use matplotlib to show velocity in a single subplot with a color
        bar. Clip to absolute max set by maxv, though in practives percentile
        will clip at a signficantly lower value.
        Parameters
        ----------
        ax : matplotlib axis, optional
            axes for plot. The default is None.
        band : str, optional
            band to plot (any of loaded variables). The default is 'vv'.
        plotFontSize : int, optional
            Font size for plot. The default is plotFontSize.
        titleFontSize : TYPE, optional
            Font size for title. The default is titleFontSize.
        labelFontSize : TYPE, optional
            Font size for labels. The default is labelFontSize.
        scale : str, optional
            Options are "linear" and "log" The default is linear.
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
        if scale == 'linear':
            # clip to percentile value
            vmin, vmax = self.autoScaleRange(band, None, vmin, vmax,
                                             percentile)
        elif scale == 'log':
            vmin = max(.1, vmin)  # Don't allow to small a value
        else:
            print('Invalid scale option, use linear or log')
            return
        # Display data
        self.displayVar(band, ax=ax, plotFontSize=self.plotFontSize,
                        labelFontSize=self.labelFontSize, midDate=midDate,
                        colorBarLabel='Speed (m/yr)', vmax=vmax, vmin=vmin,
                        axisOff=axisOff, colorBar=colorBar,
                        scale=scale, **kwargs)
