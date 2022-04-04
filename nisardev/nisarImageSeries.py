#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from nisardev import nisarBase2D, nisarImage
# import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
import xarray as xr
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

    def __init__(self, verbose=True,  imageType=None, numWorkers=2):
        '''
        Instantiate nisarImageSeries object.
        Parameters
        ----------
        verbose : bool, optional
            Increase level of informational messages. The default is True.
        Returns
        -------
        None.
        '''
        nisarBase2D.__init__(self, numWorkers=numWorkers)
        self.image, self.sigma0, self.gamma0 = [None] * 3
        #
        self.myVariables(imageType)
        self.imageType = imageType
        self.verbose = verbose
        self.noDataDict = dict(zip(imageTypes, [0, -30., -30.]))
        self.nLayers = 0  # Number of time layers

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
        
    def readSeriesFromTiff(self, fileNames, url=False, stackVar=None,
                           index1=3, index2=4, dateFormat='%d%b%y',
                           overviewLevel=None, suffix=''):
        '''
        read in a tiff product fileNameBase.*.tif. If
        Files can be read as np arrays of xarrays (useXR=True, not well tested)

        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
            pattern.*.abc or pattern*.
            The wildcard (*) will be filled with the values in myVars
            e.g.,pattern.vx.abc.tif, pattern.vy.abc.tif.
        url : bool, optional
            Read data from url
        stacVar : diction, optional
            for stackstac {'bounds': [], 'resolution': res, 'epsg': epsg}
        index1, index2 : location of dates in filename with seperated by _
        dateFormat : format code to strptime
        overviewLevel: int
            Overview (pyramid) level to read: None->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level
        suffix : str, optional
            Any suffix that needs to be appended (e.g., for dropbox links)
        Returns
        -------
        None.
        '''
        self.imageMaps = []
        for fileName in fileNames:
            fileName = fileName.replace('.tif', '')
            myImage = nisarImage()
            myImage.readDataFromTiff(fileName,
                                     url=url, stackVar=stackVar,
                                     index1=index1, index2=index2,
                                     dateFormat=dateFormat,
                                     overviewLevel=overviewLevel,
                                     suffix=suffix)
            self.imageMaps.append(myImage)
        bBox = myImage.boundingBox(units='m')
        self.imageType = myImage.imageType
        # Combine individual bands
        self.nLayers = len(fileNames)
        self.xr = xr.concat([x.xr for x in self.imageMaps], dim='time',
                            join='override', combine_attrs='drop')
        # ensure that properly sorted in time
        self.xr = self.xr.sortby(self.xr.time)
        # This forces a subset=entire image, which will trigger initialization
        # Spatial parameters derived from the first imageMap
        self.variables = self.myVariables(self.imageType)
        self.subSetImage(bBox)
        self.xr = self.xr.rename('ImageSeries')
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

    # ------------------------------------------------------------------------
    # Ploting routines.
    # ------------------------------------------------------------------------

    def displayImageForDate(self, date=None, ax=None,
                            plotFontSize=plotFontSize,
                            titleFontSize=titleFontSize,
                            labelFontSize=labelFontSize,
                            autoScale=True, axisOff=False,  colorBar=True,
                            vmin=None, vmax=None, percentile=100, cmap='gray',
                            colorBarPosition='right', colorBarSize='5%',
                            colorBarPad=0.05,
                            colorBarLabel=None, **kwargs):
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
            max value to display. The default is 7000.
        vmin : number, optional
            min value to display. The default is 0.
        percentile : number, optional
            percentile to clip display at. The default is 100
        **kwargs : dict
            kwargs passed to imshow.
        Returns
        -------
        None.

        '''
        band = self.variables[0]
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
        self.displayVar(band, date=date, ax=ax, plotFontSize=plotFontSize,
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize,
                        axisOff=axisOff, colorBar=colorBar,
                        colorBarPosition=colorBarPosition,
                        colorBarSize=colorBarSize, colorBarPad=colorBarPad,
                        colorBarLabel=colorBarLabel, vmax=vmax, vmin=vmin,
                        cmap=cmap, **kwargs)

    @classmethod
    def reproduce(cls):
        ''' Create and return a new instance of imageSeries '''
        return cls()
