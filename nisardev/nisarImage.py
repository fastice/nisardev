#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:32:18 2022

@author: ian
"""

import numpy as np
from nisardev import nisarBase2D
import os
from datetime import datetime
from osgeo import gdal_array

imageTypes = ['image', 'sigma0', 'gamma0']


class nisarImage(nisarBase2D):
    ''' This class creates objects to contain nisar amplitude/power imagery.
    '''

    labelFontSize = 14  # Font size for plot labels
    plotFontSize = 12  # Font size for plots
    legendFontSize = 12  # Font size for legends
    titleFontSize = 15  # Font size for legends

    def __init__(self, imageType=None, verbose=True,
                 noData=None, dtype=None, numWorkers=2):
        '''
        Instantiate nisarVel object. Possible bands are 'image', 'sigma0',
        'gamma0', or user defined.
        Parameters
        ----------
        imageType: str
            imageType custom name, or image, sigma0, gamma0. If not specified
            determined from grimp product name.
        verbose : bool, optional
            Increase level of informational messages. The default is True.
        noData : scalar
            no data value. Defaults to np.nan if not image/sigma/gamma
        Returns
        -------
        None.
        '''
        nisarBase2D.__init__(self, numWorkers=numWorkers)
        # self.image, self.sigma0, self.gamma0 = [None] * 3
        if type(imageType) is not str and imageType is not None:
            print('Warning invalid image type')
            return None

        #
        if dtype is not None:
            try:
                np.dtype(dtype)
            except Exception:
                print(f'Invalid dtype {dtype}')
                return None
        #
        self.myVariables(imageType)
        #
        self.imageType = imageType
        self.verbose = verbose
        #
        if imageType not in imageTypes:
            self.noDataDict = dict(zip(imageTypes, [0, -30., -30.]))
        else:
            if noData is None:
                noData = np.nan
            self.noDataDict = dict(zip([imageType], [noData]))

    def myVariables(self, imageType):
        '''
        Set myVariables based on imageType (image, sigma0, gamma0).
        Unlike velocity, images are single band.
        Parameters
        ----------
        imageType str:
            imageType either 'image', 'sigma0', or 'gamma0'
        Returns
        -------
        myVars : list of str
            list with variable name, e.g. ['gamma0'].
        '''
        if imageType is None:
            return
        # if imageType not in imageTypes:
        #    print(f'Invalid Image Type: {imageType} must be {imageTypes}')
        myVars = [imageType]
        self.variables = myVars
        #
        # Resolve dtype
        if imageType in imageTypes and self.dtype is None:
            self.dtype = dict(zip(imageTypes,
                                  ['uint8', 'float32', 'float32']))[imageType]
        else:
            if self.dtype is None:
                self.dtype = 'float32'  # Custom default
        #
        self.gdalType = gdal_array.NumericTypeCodeToGDALTypeCode(
            np.dtype(self.dtype))
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

    def detectImageType(self, fileNameBase):
        '''
        Detect image type from file name, defaults to 'image' if not found
        Parameters
        ----------
        fileNameBase : str
            Filename base.
        Returns
        -------
        None.
        '''
        if 'image' in fileNameBase:
            self.imageType = 'image'
        elif 'gamma0' in fileNameBase:
            self.imageType = 'gamma0'
        elif 'sigma0' in fileNameBase:
            self.imageType = 'sigma0'
        else:
            print('Could not detect image defaulting to "image"')
            self.imageType = 'image'

        self.myVariables(self.imageType)

    def readDataFromTiff(self, fileNameBase, url=False, useStack=True,
                         dateFormat='%d%b%y', index1=3, index2=4,
                         overviewLevel=-1, suffix='', chunkSize=1024,
                         date1=None, date2=None):
        '''
        read in a tiff product fileNameBase.*[,tif], tif ext optional.
        Files can be read as np arrays of xarrays (useXR=True, not well tested)
        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
        url : bool, optional
            Read data from url
        useStack : boolean, optional
            Uses stackstac for full resolution data. The default is True.
        dateFormat : str, optional
            Format code to strptime from file name. Default is %d%b%y',
        index1, index2 : location of dates in filename with seperated by _
            dateFormat : format code to strptime
        overviewLevel: int
            Overview (pyramid) level to read: None->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level
        suffix : str, optional
            Any suffix that needs to be appended (e.g., for dropbox links)
        chunkSize : int, optional
            Chunksize for xarray. Default is 1024.
        date1 : "YYYY-MM-DD" or datetime
            First date. Default is None, which parses date from filename.
        date2 : "YYYY-MM-DD" or datetime
            Second date. Default is None, which parses date from filename.
        Returns
        -------
        None.
        '''
        # reader will add tiff, so strip here if present
        fileNameBase = fileNameBase.replace('.tif', '')

        self.detectImageType(fileNameBase)
        #
        self.parseImageDatesFromFileName(fileNameBase,
                                         index1=index1, index2=index2,
                                         date1=date1, date2=date2)
        self.variables = self.myVariables(self.imageType)
        #
        if 'image' in self.variables:
            fill_value = 0
        else:
            fill_value = np.nan
        self.readXR(fileNameBase, url=url, masked=False, useStack=useStack,
                    time=self.midDate, time1=self.date1,
                    time2=self.date2, overviewLevel=overviewLevel,
                    suffix=suffix, fill_value=fill_value, chunkSize=chunkSize)
        # Rename with to image type
        self.xr = self.xr.rename(self.imageType)
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
        # set times
        self.time = [np.datetime64(self.xr.time.item(), 'ns')]
        self.time1 = [np.datetime64(self.xr.time1.item(), 'ns')]
        self.time2 = [np.datetime64(self.xr.time2.item(), 'ns')]

    def subSetImage(self, bbox):
        ''' Subset dataArray to a bounding box
        Parameters
        ----------
        bbox : dict
            {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        Returns
        -------
        None.
        '''
        self.subSetData(bbox)

    #
    # ---- Dates routines.
    #

    def parseImageDatesFromFileName(self, fileNameBase, index1=3, index2=4,
                                    dateFormat='%d%b%y',
                                    date1=None, date2=None):
        '''
        Parse the dates from the directory name the image products are
        stored in.
        Parameters
        ----------
           fileNameBase : str
            FileNameBase should be of the form
        index1, index2 : ints, optional
            location of dates in filename with seperated by _
        dateFormat : str, optional
            format code to strptime
        date1 : "YYYY-MM-DD" or datetime
            First date. Default is None, which parses date from filename.
        date2 : "YYYY-MM-DD" or datetime
            Second date. Default is None, which parses date from filename.
        Returns
        -------
        date: mid date.
            First and last dates from meta file.
        '''
        baseNamePieces = os.path.basename(fileNameBase).split('_')
        # No dates, so parse from file
        if date2 is None and date1 is None:
            self.date1 = datetime.strptime(baseNamePieces[index1], dateFormat)
            self.date2 = datetime.strptime(baseNamePieces[index2], dateFormat)
        else:
            # Use provided dates
            if date1 is not None:
                self.date1 = self.parseDate(date1)
            if date2 is not None:
                self.date2 = self.parseDate(date2)
            else:  # No date range
                date2 = date1
        # Dateless case
        if self.date1 is None:
            self.midDate = None
        else:
            self.midDate = self.date1 + (self.date2 - self.date1) * 0.5
        #
        return self.midDate

    #
    # ---- Ploting routines.
    #

    def displayImage(self, date=None,
                     ax=None,
                     vmin=None,
                     vmax=None,
                     percentile=100,
                     autoScale=True,
                     units='m',
                     title=None,
                     cmap='gray',
                     plotFontSize=13,
                     titleFontSize=16,
                     labelFontSize=15,
                     fontScale=1,
                     axisOff=False,
                     midDate=True,
                     colorBar=True,
                     colorBarPosition='right',
                     colorBarLabel=None,
                     colorBarSize='5%',
                     colorBarPad=0.05,
                     backgroundColor=(1, 1, 1),
                     wrap=None,
                     masked=None,
                     extend=None,
                     **kwargs):
        '''
         Use matplotlib to show a image layer selected by date.
         Clip to absolute max set by maxv, though in practives percentile
         will clip at a signficantly lower value.

        Parameters
        ----------
        date : str or datetime
            Approximate date to plot (nearest selected).
        ax : matplotlib axis, optional
            axes for plot. The default is None.
        vmin : number, optional
            min value to display. None -> default to type specific (dB/DN).
        vmax : number, optional
            min value to display. None -> default to type specific (dB/DN).
        percentile : number, optional
            percentile to clip display at. The default is 100
        units : str, optional
            units of coordiinates (m or km). The default is 'm'.
        autoScale : bool, optional
            Autoscale plot range,but not exceed vmin,vmax. The default is True.
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
            1.2 would increase by 20%). The default is 1.
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
        band = self.variables[0]
        if vmax is None:
            vmax = {'image': 255, 'sigma0': 10, 'gamma0': 10}[band]
        if vmin is None:
            vmin = {'image': 0, 'sigma0': -30, 'gamma0': -30}[band]
        # clip to percentile value
        vmin, vmax = self.autoScaleRange(band, None, vmin, vmax, percentile,
                                         quantize=1.)
        # Display data
        colorBarLabel = {'image': 'DN', 'gamma0': '$\\gamma_o$ (dB)',
                         'sigma0': '$\\sigma_o$ (dB)'}[band]
        self.displayVar(band,
                        date=date,
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                        units=units,
                        title=title,
                        cmap=cmap,
                        plotFontSize=plotFontSize,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        fontScale=fontScale,
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

    def plotPoint(self, x, y, *args, band=None, ax=None, **kwargs):
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
            band name (image, sigma0, gamma0). The default is 1st band loaded.
        ax : axis, optional
            matplotlib axes. The default is None.
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax.

        '''
        band = self.variables[0]
        ax = self._plotPoint(x, y, band, *args, ax=ax, **kwargs)
        return ax

    def plotProfile(self, x, y, *argv, band=None, ax=None,
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
            band name (image, sigma0, gamma0). The default is 1st band loaded.
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
        band = self.variables[0]
        ax = self._plotProfile(x, y, band, *argv, date=None, ax=ax,
                               midDate=midDate, distance=distance, units=units,
                               **kwargs)
        return ax

    def labelProfilePlot(self, ax, band=None,
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
        if band not in self.variables:
            band = self.variables[0]
        imageLabels = {'image': 'DN value', 'gamma0': '$\\gammma_o$ (dB)',
                       'sigma0': '$\\sigma_o$ (dB)'}
        if xLabel is None:
            xLabel = f'Distance ({units})'
        if yLabel is None:
            yLabel = imageLabels[band]
        #
        self._labelAxes(ax, xLabel, yLabel,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        plotFontSize=plotFontSize,
                        fontScale=fontScale,
                        axisOff=axisOff, title=title)

    def labelPointPlot(self, ax, band=None,
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
        band = self.variables[0]
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
