#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from nisardev import nisarBase2D, nisarVel
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
import xarray as xr
from dask.diagnostics import ProgressBar
# from datetime import datetime


class nisarVelSeries(nisarBase2D):
    ''' This class creates objects to contain nisar velocity and/or error maps.
    The data can be pass in on init, or read from a geotiff.

    The variables that are used are specified with useVelocity, useErrrors,
    and, readSpeed and (see nisarVel.readDatafromTiff).

    The variables used are returned as a list (e.g., ["vx","vy"])) by
    nisarVel.myVariables(useVelocity=True, useErrors=False, readSpeed=False).
    '''

#    labelFontSize = 15  # Font size for plot labels
#    plotFontSize = 13  # Font size for plots
    legendFontSize = 12  # Font size for legends
#    titleFontSize = 16  # Font size for legends

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
                           'ex': -1.0, 'ey': -1.0, 'ev': -1.0, 'dT': -2.0e9}
        self.gdalType = gdal.GDT_Float32  # data type for velocity products
        self.dtype = 'float32'
        self.nLayers = 0  # Number of time layers

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
        if useErrors:
            myVars += ['ex', 'ey', 'ev']
        if useDT:
            myVars += ['dT']
        self.variables = myVars
        return myVars

    # ------------------------------------------------------------------------
    # Interpolation routines - to populate abstract methods from nisarBase2D
    # ------------------------------------------------------------------------

    def interp(self, x, y, date=None, units='m', returnXR=False, grid=False,
               **kwargs):
        '''
        Call appropriate interpolation method to interpolate myVars at x, y
        points.
        Parameters
        ----------
        x : nparray
            x coordinates.
        y : nparray
            y coordinates.
        date : datestr, optional
            Interpolate layer nearest this date. The default is None.
        units : str ('m' or 'km'), optional
            Units. The default is 'm'.
        returnXR : boolean, optional
            Return result as xarray instead of nparray. The default is False.
        grid : boolean, optional
            If false, interpolate at x, y values. If true create grid with
            x and y 1-d arrays for each dimension. The default is False.
        **kwargs : TYPE
            DESCRIPTION.
        Returns
        -------
        npaarray or xarray
            Interpolated result as nparray or xarray depending on returnXR.
        '''

        if not self._checkUnits(units):
            return
        return self.interpGeo(x, y, self.variables, date=date, units=units,
                              grid=grid, returnXR=returnXR, **kwargs)

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
                           useDT=False,
                           readSpeed=False, url=False, useStack=True,
                           index1=None, index2=None, dateFormat=None,
                           overviewLevel=-1, suffix='', chunkSize=1024):
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
        useStack : Boolean, optional
            Use stackstac for full res data, overviews will xr.
            The default is True.
        index1, index2 : location of dates in filename with seperated by _
        dateFormat : format code to strptime
        overviewLevel: int, optional
            Overview (pyramid) level to read: -1->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level.
            The default is -1 (full res).
        suffix : str, optional
            Any suffix that needs to be appended (e.g., for dropbox links)
        chunkSize : int, optional
            Chunksize for xarray. Default is 1024.
        Returns
        -------
        None.
        '''
        self.variables = self.myVariables(useVelocity, useErrors, useDT,
                                          readSpeed=readSpeed)
        self.velMaps = []
        #
        stackTemplate = None
        with ProgressBar():
            for fileName in fileNames:
                myVel = nisarVel()
                myVel.stackTemplate = stackTemplate
                myVel.readDataFromTiff(fileName, useVelocity=useVelocity,
                                       useErrors=useErrors,
                                       readSpeed=readSpeed,
                                       url=url, useStack=useStack,
                                       index1=index1, index2=index2,
                                       dateFormat=dateFormat,
                                       overviewLevel=overviewLevel,
                                       suffix=suffix, chunkSize=chunkSize)
                stackTemplate = myVel.stackTemplate
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
        self.time = [self.datetime64ToDatetime(x) for x in self.xr.time.data]
        # Update times
        self._getTimes()

    def _addSpeedSeries(self, bandType='v'):
        ''' Add speed if only have vx and vy '''
        band = f'{bandType}v'
        #
        if bandType == 'v':
            wx, wy = 1., 1.,
        elif bandType == 'e':
            wx = self.vx / self.vv
            wy = self.vy / self.vv
        else:
            print('_addSpeed: Invalid band type')
        #
        dv = xr.DataArray(np.sqrt(
            np.square(wx * getattr(self, f'{bandType}x')) +
            np.square(wy * getattr(self, f'{bandType}u'))),
                          coords=[self.xr.time, self.xr.y, self.xr.x],
                          dims=['time', 'y', 'x'])
        # add band for vv
        dv = dv.expand_dims(dim=['band'])
        dv['band'] = [band]
        dv['time'] = self.xr['time']
        dv['name'] = self.xr['name']
        # Add to vx, vy xr
        self.xr = xr.concat([self.xr, dv], dim='band', join='override',
                            combine_attrs='drop')
        #
        if band not in self.variables:
            self.variables.append(band)
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

        # fix band order
        self.xr = self._setBandOrder(
            {'vx': 0, 'vy': 1, 'vv': 2, 'ex': 3, 'ey': 4, 'ev': '5', 'dT': 5})
        self.subset = self.xr
        # get times
        self._getTimes()

    def timeSliceVel(self, date1, date2):
        ''' Create a new velSeries for the range date1 to date2
        Parameters
        ----------
        date1 : Tdatetime or "YYYY-MM-DD"
            First date in range.
        date2 : TYPE
            Second date in range.

        Returns
        -------
        series new velSeries
        '''
        newSeries = self.timeSliceData(date1, date2)
        # self.time = [self.datetime64ToDatetime(x) for x in self.xr.time.data]
        # get times
        newSeries._getTimes()
        return newSeries

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

    def displayVelForDate(self, date=None,
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
                          extend=None, **kwargs):
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
        # Compute auto scale params
        if scale == 'linear':
            # clip to percentile value
            if autoScale:
                vmin, vmax = self.autoScaleRange(band, date, vmin, vmax,
                                                 percentile)
        elif scale == 'log':
            vmin = max(.1, vmin)  # Don't allow too small a value
        else:
            print('Invalid scale option, use linear or log')
            return
        # Create plot
        return self.displayVar(band, date=date, ax=ax,
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

    @classmethod
    def reproduce(cls):
        ''' Create and return a new instance of velocitySeries '''
        return cls()

    def plotPoint(self, x, y, *args, band='vv', ax=None, sourceEPSG=None,
                  units='m', **kwargs):
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
        band : str
            band name (vx, vy, vv). The default is 'vv'.
        ax : axis, optional
            matplotlib axes. The default is None.
        units : str, optional
            Units (m or km) for the x, y coordinates. The default is 'm'
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax.

        '''
        ax = self._plotPoint(x, y, band, *args, ax=ax, units=units,
                             sourceEPSG=sourceEPSG, **kwargs)
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

    def plotProfile(self, x, y, *argv, band='vv', date=None, ax=None,
                    label=None, distance=None, units='m', **kwargs):
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
        date : 'YYYY-MM-DD' or datetime, optional
            The date in the series to plot. The default is the first date.
        ax : axis, optional
            matplotlib axes. The default is None.
        label : str, optional
            label for plot (same as plt.plot label). The defaults is the date.
        distance : nparray, optional
            distance variable for plot.
            The default is None, which causes it to be calculated.
        units : str, optional
            Units (m or km) for the x, y coordinates. The default is 'm'
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax. Either the value passed on or the one created if none given.
        '''
        ax = self._plotProfile(x, y, band, date, *argv,  ax=ax, label=label,
                               distance=distance, units=units,
                               **kwargs)
        return ax

    def inspect(self, band='vv', date=None, imgOpts={}, plotOpts={}):
        '''
        Display one layer of stack with interactive map to plot time series
        at a point.

        Parameters
        ----------
        band : str, optional
            Band id. The default is 'vv'.
        date : 'YYYY-MM-DD' str or datetime, optional
            The date for the map image. The default is None.
        imgOpts : dict, optional
            Image display options. The default None for vv defaults to
            {'clim': (0, 3000), 'logz': True, 'cmap': 'viridis'}.
        plotOpts : dict, optional
            Plot display options. The default is None, which defaults for vv to
             {'ylabel': 'Speed (m/yr)', 'xlabel': 'Date'}.

        Returns
        -------
        panel
            Returns the panel with the interactive plots.

        '''
        defaultImgOpts = {
            'vv': {'clim': (0, 3000), 'logz': True, 'cmap': 'viridis'},
            'vx': {'clim': (-1500, 1500), 'logz': False, 'cmap': 'bwr'},
            'vy': {'clim': (-1500, 1500), 'logz': False, 'cmap': 'bwr'},
            'ex': {'clim': (0, 20), 'logz': False, 'cmap': 'Magma'},
            'ey': {'clim': (0, 20), 'logz': False, 'cmap': 'Magma'},
            'dT': {'clim': (-30, 30), 'logz': False, 'cmap': 'bwy'}
        }

        defaultPlotOpts = {
            'vv': {'ylabel': 'Speed (m/yr)', 'xlabel': 'Date'},
            'vx': {'ylabel': 'vx (m/yr)', 'xlabel': 'Date'},
            'vy': {'ylabel': 'vy (m/yr)', 'xlabel': 'Date'},
            'ex': {'ylabel': '$ex (m/yr)$', 'xlabel': 'Date'},
            'ey': {'ylabel': 'ey (m/yr)', 'xlabel': 'Date'},
            'dT': {'ylabel': 'dT (days)', 'xlabel': 'Date'},
        }
        # Customize other common options
        for key in defaultImgOpts[band]:
            if key not in imgOpts:
                imgOpts[key] = defaultImgOpts[band][key]
        # extra img opts
        if 'xlabel' not in imgOpts:
            imgOpts['xlabel'] = 'X (m)'
        if 'ylabel' not in imgOpts:
            imgOpts['ylabel'] = 'Y (m)'
        # extra plot opts
        for key in defaultPlotOpts[band]:
            if key not in plotOpts:
                plotOpts[key] = defaultPlotOpts[band][key]
        if 'title' not in plotOpts:
            plotOpts['title'] = f'{band} time series'

        return self._view(band, imgOpts, plotOpts, date=date, markerColor='w')

    def mean(self, skipna=True, squaredErrors=True):
        '''
        Compute mean along time axis and return new instance of same class
        Note that the original xr and subset correspond to the subset
        of the calling instance.
        Parameters
        ----------
        skipna : bool, optional
            Skips nans in computation. The default is True.
        skipna : bool, optional
            Computes mean sq errors to represet errors of other averaged
            variables. The default is True.
        Returns
        -------
        same as class method called from
            Object with mean and time axis reduced to dimension of 1.

        '''
        if squaredErrors:
            errors = [x for x in self.variables if x in ['ex', 'ey', 'ev']]
        else:
            errors = []
        #
        return nisarBase2D.mean(self, skipna=skipna, errors=errors)
    #     return result