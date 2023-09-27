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
        self.vx, self.vy, self.vv, self.ex, self.ey, self.dT = [None] * 6
        self.variables = None
        self.verbose = verbose
        self.noDataDict = {'vx': -2.0e9, 'vy': -2.0e9, 'vv': -1.0,
                           'ex': -1.0, 'ey': -1.0, 'ev': -1.0, 'dT': -2.e9}
        self.gdalType = gdal.GDT_Float32  # data type for velocity products
        self.dtype = 'float32'

    def myVariables(self, useVelocity, useErrors, useDT, readSpeed=False):
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
        if useDT:
            myVars += ['dT']
        self.variables = myVars
        return myVars

    @classmethod
    def reproduce(cls):
        ''' Create and return a new instance of velocitySeries '''
        return cls()
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

    def _addSpeed(self, bandType='v'):
        ''' Add speed if only have vx and vy '''
        # Compute speed
        band = f'{bandType}v'
        if bandType == 'v':
            wx, wy = 1., 1.,
        elif bandType == 'e':
            wx = self.vx / self.vv
            wy = self.vy / self.vv
        else:
            print('_addSpeed: Invalid band type')
        dv = xr.DataArray(
            np.sqrt(np.square(wx * getattr(self, f'{bandType}x')) +
                    np.square(wy * getattr(self, f'{bandType}y'))),
            coords=[self.xr.y, self.xr.x], dims=['y', 'x'])
        # setup band as vv
        dv = dv.expand_dims(dim=['time', 'band'])
        dv['band'] = [band]
        dv['time'] = self.xr['time']
        dv['name'] = self.xr['name']
        dv['_FillValue'] = self.noDataDict[band]
        # Add to existing xr with vx and vy
        self.xr = xr.concat([self.xr, dv], dim='band', join='override',
                            combine_attrs='drop')
        # Fix order of coordinates - force vx, vy, vv, ex...
        self.xr = self._setBandOrder({'vx': 0, 'vy': 1, 'vv': 2,
                                      'ex': 3, 'ey': 4, 'ev': 5, 'dT': 6})
        #
        if band not in self.variables:
            self.variables.append(band)
        self._mapVariables()

    def readDataFromTiff(self, fileNameBase, useVelocity=True, useErrors=False,
                         useDT=False,
                         readSpeed=False, url=False, useStack=True,
                         index1=4, index2=5, dateFormat='%d%b%y',
                         overviewLevel=-1, masked=True, suffix='',
                         date1=None, date2=None, chunkSize=1024):
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
            Include velocity if True. The default is True.
        useErrors : bool, optional
            Include errors if True. The default is False.
        useDT : bool, optional
            Include dT (see GrIMP documentation). The default is False.
        readSpeed : bool, optional
            Read speed (.vv) if True. The default is False.
        url : bool, optional
            Read data from url
        useStack : boolean, optional
            Uses stackstac for full resolution data. The default is True.
        index1, index2 : location of dates in filename with seperated by _
            dateFormat : format code to strptime
        overviewLevel: int, optional
            Overview (pyramid) level to read: None->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level.
            The default is -1 (full res).
        date1 : datetime
            First date. The defaults is None (extract from filename)
        date2 : datetime
            Second date. The defaults is None (extract from filename)
        chunkSize : int, optional
            Chunksize for xarray. Default is 1024.
        Returns
        -------
        None.
        '''
        self.parseVelDatesFromFileName(fileNameBase, index1=index1,
                                       index2=index2, dateFormat=dateFormat,
                                       date1=date1, date2=date2)
        self.variables = self.myVariables(useVelocity, useErrors, useDT,
                                          readSpeed=readSpeed)
        if readSpeed:
            skip = []
        else:
            skip = ['vv']  # Force skip
        self.readXR(fileNameBase, url=url, masked=True, useStack=useStack,
                    time=self.midDate, skip=skip, time1=self.date1,
                    time2=self.date2, overviewLevel=overviewLevel,
                    suffix=suffix, chunkSize=chunkSize)
        # compute speed rather than download
        if not readSpeed and useVelocity:
            self._addSpeed(bandType='v')
            self.subSetData(self.boundingBox(units='m'))
        if useErrors:
            self._addSpeed(bandType='e')
        #
        self.xr = self.xr.rename('VelocityMap')
        self.fileNameBase = fileNameBase  # save filenameBase
        # force intial subset to entire image
        # print(self.boundingBox(units='m'))
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
        self.xr = self._setBandOrder(
             {'vx': 0, 'vy': 1, 'vv': 2, 'ex': 3, 'ey': 4, 'ev': 5, 'dT': 6})
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
        # Should be done on read now
        # if useVelocity:
        #    self.vv = np.sqrt(np.square(self.vx) + np.square(self.vy))

    # ------------------------------------------------------------------------
    # Dates routines.
    # ------------------------------------------------------------------------

    def _decodeProductType(self, fileNameBase):
        '''
        Get product type from beginning of filename ProductType_....
        Parameters
        ----------
        fileNameBase : str
            name of file.
        Returns
        -------
        productType.

        '''
        validTypes = ['GL', 'TSX', 'OPT', 'CSK']
        productType = fileNameBase.split('_')[0]
        if productType in validTypes:
            return productType
        return None

    def _getDateFormat(self, productType, index1=None, index2=None,
                       dateFormat=None):
        '''
        Return parameters to decode name based on product type
        Parameters
        ----------
        productType : str
            Identifier for product type (e.g,. S1, TSX...).

        Returns
        -------
        index1, index2, dateFormat
        '''
        indices = {'GL': {'index1': 4, 'index2': 5, 'dateFormat': '%d%b%y'},
                   'TSX': {'index1': 2, 'index2': 3, 'dateFormat': '%d%b%y'},
                   'CSK': {'index1': 2, 'index2': 3, 'dateFormat': '%d%b%y'},
                   'OPT': {'index1': 2, 'index2': None, 'dateFormat': '%Y-%m'}}
        # get defaults
        if productType is not None:
            try:
                myIndices = indices[productType]
            except Exception:
                myIndices = indices['GL']
                print('_getDateFormat: '
                      'Could not parse product type {productType}')
                print('Defaulting to GL')
        else:
            # default if no defined product type
            myIndices = {'index1': 4, 'index2': 5, 'dateFormat': '%d%b%y'}
        # allow overrides
        for key in myIndices:
            if locals()[key] is not None:
                myIndices[key] = locals()[key]
        #
        return list(myIndices.values())

    def parseVelDatesFromFileName(self, fileNameBase, index1=None, index2=None,
                                  dateFormat=None, date1=None, date2=None):
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
        # formats for decoding dates from names
        if date1 is None:
            productType = \
                self._decodeProductType(os.path.basename(fileNameBase))
            index1, index2, dateFormat = \
                self._getDateFormat(productType,
                                    index1=index1,
                                    index2=index2,
                                    dateFormat=dateFormat)
            baseNamePieces = os.path.basename(fileNameBase).split('_')
            self.date1 = datetime.strptime(baseNamePieces[index1], dateFormat)
        else:
            self.date1 = date1
        if date2 is None and index2 is not None:
            if index2 is not None:
                self.date2 = datetime.strptime(baseNamePieces[index2],
                                               dateFormat)
        elif date2 is not None:
            self.date2 = date2
        else:
            tmp = self.date1 + timedelta(days=32)
            self.date2 = tmp - timedelta(days=tmp.day)
        self.midDate = self.date1 + (self.date2 - self.date1) * 0.5
        #
        self.time1 = self.date1
        self.time2 = self.date2
        return self.midDate

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayVel(self,
                   ax=None,
                   band='vv',
                   vmin=0,
                   vmax=7000,
                   percentile=100,
                   autoScale=True,
                   units='m',
                   title=None,
                   plotFontSize=13,
                   titleFontSize=16,
                   labelFontSize=15,
                   fontScale=1.,
                   scale='linear',
                   axisOff=False,
                   midDate=True,
                   colorBar=True,
                   colorBarPosition='right',
                   colorBarLabel='Speed (m/yr)',
                   colorBarSize='5%',
                   colorBarPad=0.05,
                   wrap=None,
                   masked=None,
                   backgroundColor=(1, 1, 1),
                   extend=None,
                   **kwargs):
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
            component to plot (any of loaded variables). The default is 'vv'.\
        vmin : number, optional
            min velocity to display. The default is 0.
        vmax : number, optional
            max velocity to display. The default is 7000.
        percentile : number, optional
            percentile to clip display at. The default is 100
        autoScale : bool, optional
            Autoscale plot range,but not exceed vmin,vmax. The default is True.
        units : str, optional
            units of coordiinates (m or km). The default is 'm'.
        title : str, optional
            Plot title, use for '' for no title. A value of None defaults to
            the image date.
        plotFontSize : int, optional
            Font size for plot. The default is 13.
        titleFontSize : int, optional
            Font size for title. The default is 16.
        labelFontSize : int, optional
            Font size for labels. The default is 15.
        fontScale : float, optional
            Scale factor to apply to label, title, and plot fontsizes (e.g.,
            1.2 would increase by 20%). The default is 1 .
        scale : str, optional
            Scale type ('linear' or 'log') The default is 'linear'.
        axisOff : TYPE, optional
            Turn axes off. The default is False.
        midDate : Boolean, optional
            Use middle date for titel. The default is True.
        colorBarLabel : str, optional
            Label for colorbar. The default is 'Speed (m/yr)'.
        colorBarPosition : TYPE, optional
            Color bar position (e.g., left, top...). The default is 'right'.
        colorBarSize : str, optional
            Color bar size specfied as 'n%'. The default is '5%'.
        colorBarPad : float, optional
            Color bar pad. The default is 0.05.
        wrap : float, optional
            Display data modulo wrap. The default is None.
        extend : str, optional
            Colorbar extend ('both','min', 'max', 'neither').
            The default is None.
        backgroundColor : color, optional
            Background color. The default is (1, 1, 1).
        wrap :  number, optional
             Display velocity modululo wrap value
        masked : Boolean, optional
            Masked for imshow. The default is None.
        **kwargs : dict
            kwargs passed to imshow.
        Returns
        -------
        pos : matplotlib.image.AxesImage
            return value from imshow.

        '''
        # Compute display bounds
        if scale == 'linear':
            # clip to percentile value
            if autoScale:
                vmin, vmax = self.autoScaleRange(band, None, vmin, vmax,
                                                 percentile)
        elif scale == 'log':
            vmin = max(.1, vmin)  # Don't allow too small a value
        else:
            print('Invalid scale option, use linear or log')
            return
        # Display data
        return self.displayVar(band, ax=ax,
                               vmin=vmin,
                               vmax=vmax,
                               units=units,
                               title=title,
                               plotFontSize=plotFontSize,
                               labelFontSize=labelFontSize,
                               titleFontSize=titleFontSize,
                               fontScale=fontScale,
                               scale=scale,
                               midDate=midDate,
                               axisOff=axisOff,
                               colorBar=colorBar,
                               colorBarLabel=colorBarLabel,
                               colorBarPad=colorBarPad,
                               colorBarSize=colorBarSize,
                               colorBarPosition=colorBarPosition,
                               extend=extend,
                               backgroundColor=backgroundColor,
                               wrap=wrap,
                               masked=masked,
                               **kwargs)

    def plotPoint(self, x, y, *args, band='vv', ax=None, **kwargs):
        '''
        Interpolate data set at point x, y, and plot result vs time

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.

        *argv : list
            Additional args to pass to plt.plot (e.g. 'r*').
        band : str, optional
            band name (vx, vy, vv). The default is 'vv'.
        ax : axis, optional
            matplotlib axes. The default is None.
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax.

        '''
        ax = self._plotPoint(x, y, band, *args, ax=ax, **kwargs)
        return ax

    def plotProfile(self, x, y, *argv, band='vv', ax=None,
                    midDate=True, distance=None, units='m', **kwargs):
        '''
        Interpolate data for profile x, y and plot as a function of distance

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.

        *argv : list
            Additional args to pass to plt.plot (e.g. 'r*').
        band : str, optional
            band name (vx, vy, vv). The default is 'vv'.
        ax : axis, optional
            matplotlib axes. The default is None.
        distance : nparray, optional
            distance variable for plot.
            The default is None, which causes it to be calculated.
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax.
        '''
        ax = self._plotProfile(x, y, band, *argv, date=None, ax=ax,
                               midDate=midDate, distance=distance, units=units,
                               **kwargs)
        return ax

    def labelProfilePlot(self, ax, band='vv',
                         xLabel=None, yLabel=None,
                         units='m',
                         title=None,
                         labelFontSize=15, titleFontSize=16, plotFontSize=13,
                         fontScale=1, axisOff=False):
        '''
        Label a profile plot

        Parameters
        ----------
        ax : axis
            matplotlib axes. The default is None.
        band : str, optional
            band name (vx, vy, vv). The default is 'vv'.
        xLabel : tr, optional
            x-axis label. The default is 'Distance', use '' to disable.
        yLabel : tr, optional
            x-axis label. The default is band appropriate (e.g, Speed),
            use '' to disable.
        units : str, optional
            Units (m or km) for the x, y coordinates. The default is 'm'
        title : str, optional
            Plot titel. The default is None.
        labelFontSize : int, optional
            Font size for x&y labels. The default is 15.
        titleFontSize : int, optional
            Fontsize for plot title The default is 16.
        plotFontSize : int, optional
            Font size for tick labels. The default is 13.
        fontScale : float, optional
            Scale factor to apply to label, title, and plot fontsizes.
            The default is 1.
        axisOff : Boolean, optional
            Set to True to turn axis off. The default is False.

        Returns
        -------
        None.

        '''
        speedLabels = {'vv': 'Speed', 'vx': '$v_x$', 'vy': '$v_y$'}
        if xLabel is None:
            xLabel = f'Distance ({units})'
        if yLabel is None:
            yLabel = speedLabels[band] + ' (m/yr)'
        #
        self._labelAxes(ax, xLabel, yLabel,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        plotFontSize=plotFontSize,
                        fontScale=fontScale,
                        axisOff=axisOff, title=title)

    def labelPointPlot(self, ax, band='vv',
                       xLabel=None, yLabel=None,
                       units='m',
                       title=None,
                       labelFontSize=15, titleFontSize=16, plotFontSize=13,
                       fontScale=1, axisOff=False):
        '''
        Label a profile plot

        Parameters
        ----------
        ax : axis
            matplotlib axes. The default is None.
        band : str, optional
            band name (vx, vy, vv). The default is 'vv'.
        xLabel : tr, optional
            x-axis label. The default is 'Distance', use '' to disable.
        yLabel : tr, optional
            x-axis label. The default is band appropriate (e.g, Speed),
            use '' to disable.
        units : str, optional
            Units (m or km) for the x, y coordinates. The default is 'm'
        title : str, optional
            Plot title. The default is None.
        labelFontSize : int, optional
            Font size for x&y labels. The default is 15.
        titleFontSize : int, optional
            Fontsize for plot title The default is 16.
        plotFontSize : int, optional
            Font size for tick labels. The default is 13.
        fontScale : float, optional
            Scale factor to apply to label, title, and plot fontsizes.
            The default is 1.
        axisOff : Boolean, optional
            Set to True to turn axis off. The default is False.

        Returns
        -------
        None.

        '''
        speedLabels = {'vv': 'Speed', 'vx': '$v_x$', 'vy': '$v_y$'}
        if xLabel is None:
            xLabel = 'Date'
        if yLabel is None:
            yLabel = speedLabels[band] + ' (m/yr)'
        #
        self._labelAxes(ax, xLabel, yLabel,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        plotFontSize=plotFontSize,
                        fontScale=fontScale,
                        axisOff=axisOff, title=title)
