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
from osgeo import gdal

imageTypes = ['image', 'sigma0', 'gamma0']


class nisarImage(nisarBase2D):
    ''' This class creates objects to contain nisar amplitude/power imagery.
    '''

    labelFontSize = 14  # Font size for plot labels
    plotFontSize = 12  # Font size for plots
    legendFontSize = 12  # Font size for legends
    titleFontSize = 15  # Font size for legends

    def __init__(self, verbose=True, imageType=None, numWorkers=2):
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
        nisarBase2D.__init__(self, numWorkers=numWorkers)
        self.image, self.sigma0, self.gamma0 = [None] * 3
        #
        self.myVariables(imageType)
        self.imageType = imageType
        self.verbose = verbose
        self.noDataDict = dict(zip(imageTypes, [0, -30., -30.]))

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

    def readDataFromTiff(self, fileNameBase, url=False, stackVar=None,
                         dateFormat='%d%b%y', index1=3, index2=4,
                         overviewLevel=None, suffix=''):
        '''
        read in a tiff product fileNameBase.*[,tif], tif ext optional.
        Files can be read as np arrays of xarrays (useXR=True, not well tested)
        Parameters
        ----------
        fileNameBase : str
            FileNameBase should be of the form
        url : bool, optional
            Read data from url
        stackVar : diction, optional (under development)
            for stackstac {'bounds': [], 'resolution': res, 'epsg': epsg}
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
        # reader will add tiff, so strip here if present
        fileNameBase = fileNameBase.replace('.tif', '')
        if self.imageType is None:
            self.detectImageType(fileNameBase)
        self.parseImageDatesFromFileName(fileNameBase,
                                         index1=index1, index2=index2)
        self.variables = self.myVariables(self.imageType)
        #
        self.readXR(fileNameBase, url=url, masked=False, stackVar=stackVar,
                    time=self.midDate, time1=self.date1,
                    time2=self.date2, overviewLevel=overviewLevel, suffix=suffix)
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
                                    dateFormat='%d%b%y'):
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

    #
    # ---- Ploting routines.
    #

    def displayImage(self, ax=None,
                     plotFontSize=plotFontSize,
                     titleFontSize=titleFontSize,
                     labelFontSize=labelFontSize,
                     axisOff=False, midDate=True,  colorBar=True,
                     vmin=None, vmax=None, percentile=100, cmap='gray',
                     colorBarPosition='right', colorBarSize='5%',
                     colorBarPad=0.05,
                     **kwargs):
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
        band = self.variables[0]
        if vmax is None:
            vmax = {'image': 255, 'sigma0': 10, 'gamma0': 10}[band]
        if vmin is None:
            vmin = {'image': 0, 'sigma0': -30, 'gamma0': -30}[band]
        # clip to percentile value
        vmin, vmax = self.autoScaleRange(band, None, vmin, vmax, percentile,
                                         quantize=1.)
        # Display data
        cLabel = {'image': 'DN', 'gamma0': '$\\gamma_o$ (dB)',
                  'sigma0': '$\\sigma_o$ (dB)'}[band]
        self.displayVar(band, ax=ax, plotFontSize=self.plotFontSize,
                        labelFontSize=self.labelFontSize, midDate=midDate,
                        colorBarLabel=cLabel, vmax=vmax, vmin=vmin,
                        scale='linear', cmap=cmap,
                        axisOff=axisOff, colorBar=colorBar,
                        colorBarPosition=colorBarPosition,
                        colorBarSize=colorBarSize, colorBarPad=colorBarPad,
                        **kwargs)
