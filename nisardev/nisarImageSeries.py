#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

from nisardev import nisarBase2D, nisarImage
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
import xarray as xr
import numpy as np
import warnings
# from dask.diagnostics import ProgressBar

imageTypes = ['image', 'sigma0', 'gamma0']


class nisarImageSeries(nisarBase2D):
    ''' This class creates objects to contain nisar/gimp maps.
    The data can be pass in on init, or read from a geotiff.

    '''

    labelFontSize = 14  # Font size for plot labels
    plotFontSize = 12  # Font size for plots
    legendFontSize = 12  # Font size for legends
    titleFontSize = 15  # Font size for legends

    def __init__(self, verbose=True,  imageType=None, numWorkers=2, **kwds):
        '''
        Instantiate nisarImageSeries object.
        Parameters
        ----------
        verbose : bool, optional
            Increase level of informational messages. The default is True.
        imageType : str, optional
            Type of image data: 'image' (DN), 'sigma0' (dB), or 'gamma0' (dB).
            The default is None (must be set before reading data).
        numWorkers : int, optional
            Number of dask workers for parallel I/O. The default is 2.
        **kwds : dict
            Additional keyword arguments passed to nisarBase2D.__init__.
        Returns
        -------
        None.
        '''
        super().__init__(numWorkers=numWorkers, **kwds)
        self.image, self.sigma0, self.gamma0 = [None] * 3
        #
        self.myVariables(imageType)
        self.imageType = imageType
        self.verbose = verbose
        self.noDataDict = dict(zip(imageTypes, [0, -30., -30.]))
        self.nLayers = 0  # Number of time layers
        self.numWorkers = numWorkers

    def myVariables(self, imageType):
        '''
        Set myVariables based on imageType (image, sigma0, gamma0).
        Unlike velocity, images are single band.
        Parameters
        ----------
        imageType : str
            Image type: 'image', 'sigma0', or 'gamma0'.
        Returns
        -------
        list of str
            List containing the single variable name, e.g. ['gamma0'].
        '''
        if imageType is None:
            return
        if imageType not in imageTypes:
            print(f'Invalid Image Type: {imageType} must be {imageTypes}')
        myVars = [imageType]
        self.variables = myVars
        self.dtype = dict(zip(imageTypes,
                              ['uint8', 'float32', 'float32']))[imageType]
        self.gdalType = dict(zip(imageTypes,
                             [gdal.GDT_Byte, gdal.GDT_Float32,
                              gdal.GDT_Float32]))[imageType]
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
        x : np.ndarray
            x coordinates (polar stereographic).
        y : np.ndarray
            y coordinates (polar stereographic).
        date : str or datetime, optional
            Interpolate layer nearest to this date. None returns all layers.
            The default is None.
        units : str, optional
            Coordinate units ('m' or 'km'). The default is 'm'.
        returnXR : bool, optional
            Return an xarray DataArray instead of a numpy array.
            The default is False.
        Returns
        -------
        np.ndarray or xarray.DataArray
            Interpolated values with shape [nbands, npts].
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
        date : datetime or str 'YYYY-MM-DD'
            Date closest to the desired layer.
        returnXR : bool, optional
            If True, return an xarray DataArray instead of numpy arrays.
            The default is False.
        Returns
        -------
        list of np.ndarray + datetime, or xarray.DataArray
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

    def readSeriesFromTiff(self, fileNames,  bbox=None, url=False,
                           useStack=False,
                           index1=3, index2=4, dateFormat='%d%b%y',
                           overviewLevel=-1, suffix='', chunkSize=2048,
                           subsetMode=False):
        '''
        read in a tiff product fileNameBase.*.tif. If
        Files can be read as np arrays of xarrays (useXR=True, not well tested)

        Parameters
        ----------
        fileNames : list
            List of files should be of the form
            [pattern.*.abc ...] or [pattern*....]
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        bbox : dict, optional
            Bounding box to clip on load: {'minx': ..., 'miny': ...,
            'maxx': ..., 'maxy': ...}. The default is None (full extent).
        url : bool, optional
            Set True if filenames are URLs. The default is False.
        useStack : bool, optional
            Use cached template for faster repeated opens. The default is False.
        index1 : int, optional
            0-based position of the first date token in the '_'-split filename.
            The default is 3.
        index2 : int, optional
            0-based position of the second date token. The default is 4.
        dateFormat : str, optional
            strptime format for date tokens. The default is '%d%b%y'.
        overviewLevel : int, optional
            Overview (pyramid) level to read: -1 → full resolution,
            0 → 1/2 res, 1 → 1/4 res, etc. The default is -1.
        suffix : str, optional
            Suffix appended to the URL/filename (e.g. for Dropbox links).
            The default is ''.
        chunkSize : int, optional
            Chunk size for dask-backed xarray. The default is 2048.
        subsetMode : bool, optional
            If True, store the result as a subset instead of the full xr.
            The default is False.
        Returns
        -------
        None.
        '''
        self.imageMaps = []
        self.template = None
        self.inputParamsSeries = locals()
        self.fileNames = fileNames
        for x in ['self', 'subsetMode', 'bbox', 'fileNames']:
            self.inputParamsSeries.pop(x, None)
        #
        for fileName in fileNames:
            fileName = fileName.replace('.tif', '')
            myImage = nisarImage(numWorkers=self.numWorkers)
            myImage.template = self.template
            myImage.readDataFromTiff(fileName,
                                     bbox=bbox,
                                     url=url, useStack=useStack,
                                     index1=index1, index2=index2,
                                     dateFormat=dateFormat,
                                     overviewLevel=overviewLevel,
                                     suffix=suffix, chunkSize=chunkSize)
            self.template = myImage.template
            self.imageMaps.append(myImage)
            if bbox is None:
                bbox = myImage.boundingBox(units='m')
        bBox = myImage.boundingBox(units='m')
        self.imageType = myImage.imageType
        # Combine individual bands
        self.nLayers = len(fileNames)
        time1 = [np.datetime64(x.xr.time1.data.item(), 'ns')
                 for x in self.imageMaps]
        time2 = [np.datetime64(x.xr.time2.data.item(), 'ns')
                 for x in self.imageMaps]
        myXR = xr.concat([x.xr for x in self.imageMaps],
                         dim='time',
                         join='override',
                         combine_attrs='drop',
                         coords='minimal',
                         compat='override')
        myXR = myXR.assign_coords(time1=("time", np.array(time1)),
                                  time2=("time", np.array(time2)))
        # ensure that properly sorted in time
        myXR = myXR.sortby(myXR.time)
        # This forces a subset=entire image, which will trigger initialization
        # Spatial parameters derived from the first imageMap
        self.variables = self.myVariables(self.imageType)
        myXR = myXR.rename('ImageSeries')
        # save as xr, subset, or as subset if bbox
        if not subsetMode:
            self.xr = myXR
            self.subset = self.xr.copy(deep=True)
        else:
            self.subset = myXR
        # Save times
        self._getTimes()

    def readSeriesFromNetCDF(self, cdfFile):
        '''
        Read a cdf file previously saved by a imageSeries instance.
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
        self.subset = self.xr
        # get times
        self._getTimes()

    def timeSliceImage(self, date1, date2):
        ''' Create a new imageSeries for the range date1 to date2
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
        # get times
        newSeries._getTimes()
        return newSeries

    def subSetData(self, bbox):
        warnings.warn('\nsubSetData deprecated. Use subsetData',
                      category=DeprecationWarning,
                      stacklevel=2)
        self.subsetImage(bbox)

    def subSetImage(self, bbox):
        warnings.warn('\nsubSetImage deprecated. Use subsetImage',
                      category=DeprecationWarning,
                      stacklevel=2)
        self.subsetImage(bbox)

    def subsetData(self, bbox):
        ''' Subset dataArray to a bounding box
        Parameters
        ----------
        bbox : dict
            {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        Returns
        -------
        None.
        '''
        self.subsetImage(bbox)

    def subsetImage(self, bbox):
        ''' Subset dataArray to a bounding box
        Parameters
        ----------
        bbox : dict
            {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        Returns
        -------
        None.
        '''
        if self.template is not None:
            self.readSeriesFromTiff(self.fileNames,
                                    bbox=bbox,
                                    subsetMode=True,
                                    **self.inputParamsSeries)
            #
            self._mapVariables()
            self._parseGeoInfo()
        else:
            self._subsetData(bbox)

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayImageForDate(self,
                            date=None,
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
        cmap : str or colormap, optional
            Colormap. The default is 'gray'.
        axisOff : bool, optional
            Turn axes off. The default is False.
        midDate : bool, optional
            Use middle date for title. The default is True.
        colorBar : bool, optional
            Show a colour bar. The default is True.
        colorBarLabel : str, optional
            Label for colorbar. The default is band-appropriate
            ('DN', '$\\sigma_o$ (dB)', or '$\\gamma_o$ (dB)').
        colorBarPosition : str, optional
            Color bar position (e.g., 'left', 'top'). The default is 'right'.
        colorBarSize : str, optional
            Color bar size specified as 'n%'. The default is '5%'.
        colorBarPad : float, optional
            Color bar padding. The default is 0.05.
        wrap : float, optional
            Display data modulo wrap. The default is None.
        extend : str, optional
            Colorbar extend ('both', 'min', 'max', 'neither').
            The default is None.
        backgroundColor : color, optional
            Background color. The default is (1, 1, 1).
        masked : Boolean, optional
            Mask array for imshow. The default is None.
        **kwargs : dict
            Kwargs passed to imshow.
        Returns
        -------
        None.

        '''
        band = self.variables[0]
        if vmax is None:
            vmax = {'image': 255, 'sigma0': 10, 'gamma0': 10}[band]
        if vmin is None:
            vmin = {'image': 0, 'sigma0': -30, 'gamma0': -30}[band]
        # clip to percentile value
        vmin, vmax = self.autoScaleRange(band, None, vmin, vmax, percentile,
                                         quantize=1.)
        if colorBarLabel is None:
            colorBarLabel = {'image': 'DN', 'gamma0': '$\\gamma_o$ (dB)',
                             'sigma0': '$\\sigma_o$ (dB)'}[band]
        # Create plot
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

    @classmethod
    def reproduce(cls):
        ''' Create and return a new instance of imageSeries '''
        return cls()

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
        if band not in self.variables:
            band = self.variables[0]

        ax = self._plotPoint(x, y, band, *args, ax=ax, **kwargs)
        return ax

    def plotProfile(self, x, y, *argv, band=None, ax=None, date=None,
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
            Band name (image, sigma0, gamma0). The default is the 1st loaded band.
        ax : axis, optional
            Matplotlib axes. The default is None.
        date : str or datetime, optional
            Layer to profile. The default is the first time step.
        midDate : bool, optional
            Use the mid-date (True) or date range (False) for the plot title.
            The default is True.
        distance : nparray, optional
            Pre-computed distance array for the x-axis.
            The default is None (calculated from x, y).
        units : str, optional
            Coordinate units ('m' or 'km'). The default is 'm'.
        **kwargs : dict
            Kwargs passed through to plt.plot.

        Returns
        -------
        ax.
        '''
        if band not in self.variables:
            band = self.variables[0]
        if date is None:
            date = self.subset.time[0]
        ax = self._plotProfile(x, y, band, date, *argv, ax=ax,
                               distance=distance, units=units,
                               **kwargs)
        return ax

    def labelProfilePlot(self, ax,
                         band=None,
                         xLabel=None,
                         yLabel=None,
                         units='m',
                         title=None,
                         labelFontSize=15,
                         titleFontSize=16,
                         plotFontSize=13,
                         fontScale=1,
                         axisOff=False):
        '''
        Label a profile plot.

        Parameters
        ----------
        ax : axis
            matplotlib axes. The default is None.
        band : str, optional
            Band name (image, sigma0, gamma0). The default is 1st band loaded.
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
        imageLabels = {'image': 'DN value', 'gamma0': '$\\gamma_o$ (dB)',
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

    def labelPointPlot(self, ax,
                       band=None,
                       xLabel=None,
                       yLabel=None,
                       units='m',
                       title=None,
                       labelFontSize=15,
                       titleFontSize=16,
                       plotFontSize=13,
                       fontScale=1,
                       axisOff=False):
        '''
        Label a point-vs-time plot.

        Parameters
        ----------
        ax : axis
            matplotlib axes. The default is None.
        band : str, optional
            Band name (image, sigma0, gamma0). The default is 1st band loaded.
        xLabel : str, optional
            x-axis label. The default is 'Date', use '' to disable.
        yLabel : str, optional
            y-axis label. The default is band appropriate (e.g., DN value,
            gamma0 dB), use '' to disable.
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
        if band not in self.variables:
            band = self.variables[0]
        imageLabels = {'image': 'DN value', 'gamma0': '$\\gamma_o$ (dB)',
                       'sigma0': '$\\sigma_o$ (dB)'}
        if xLabel is None:
            xLabel = 'Date'
        if yLabel is None:
            yLabel = imageLabels[band]
        #
        #
        self._labelAxes(ax, xLabel, yLabel,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        plotFontSize=plotFontSize,
                        fontScale=fontScale,
                        axisOff=axisOff, title=title)

    def inspect(self,
                band='image',
                date=None,
                imgOpts={},
                plotOpts={}):
        '''
        Display one layer of stack with interactive map to plot time series
        at a point.

        Parameters
        ----------
        band : str, optional
            Band id. The default is 'image'.
        date : 'YYYY-MM-DD' str or datetime, optional
            The date for the map image. The default is None.
        imgOpts : dict, optional
            Image display options. The default None for vv defaults to
            {'clim': (0, 255), 'logz': False, 'cmap': 'gray'} for 'image'
            or
            {'clim': (-30, 20), 'logz': False, 'cmap': 'gray'} for sigma/gamma.
        plotOpts : dict, optional
            Plot display options. The default is None, which defaults for to
             {'ylabel': 'DN', 'xlabel': 'Date'} and for sigma/gamma to
             {'ylabel': r'$\\sigma/gamma_o$ (dB)', 'xlabel': 'Date'}.
        Returns
        -------
        panel
            Returns the panel with the interactive plots.

        '''
        defaultImgOpts = {
            'image': {'clim': (0, 255), 'logz': False, 'cmap': 'gray'},
            'sigma0': {'clim': (-30, 20), 'logz': False, 'cmap': 'gray'},
            'gamma0': {'clim': (-30, 20), 'logz': False, 'cmap': 'gray'}
        }

        defaultPlotOpts = {
            'image': {'ylabel': 'DN', 'xlabel': 'Date'},
            'sigma0': {'ylabel': r'$\\sigma_o$ (dB)', 'xlabel': 'Date'},
            'gamma0': {'ylabel': '$\\gamma_o$ (dB)', 'xlabel': 'Date'}
        }
        band = str(self.subset.band.data[0])
        # Customize other common options
        for key in defaultImgOpts[band]:
            if key not in imgOpts:
                imgOpts[key] = defaultImgOpts[band][key]
        if 'xlabel' not in imgOpts:
            imgOpts['xlabel'] = 'X (m)'
        if 'ylabel' not in imgOpts:
            imgOpts['ylabel'] = 'Y (m)'
        for key in defaultPlotOpts[band]:
            if key not in plotOpts:
                plotOpts[key] = defaultPlotOpts[band][key]
        if 'title' not in plotOpts:
            plotOpts['title'] = f'{band} time series'
        return self._view(band, imgOpts, plotOpts, date=date)
