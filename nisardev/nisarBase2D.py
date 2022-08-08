#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:37:54 2020

@author: ian
"""
from abc import ABCMeta, abstractmethod
from nisardev import myError
import numpy as np
import functools
import xarray as xr
import pyproj
import dask
import rioxarray
import stackstac
import os
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from datetime import datetime
# from dask.diagnostics import ProgressBar
import math

CHUNKSIZE = 512


class nisarBase2D():
    ''' Abstract class to define xy polar stereo (3031, 3413) image objects
    such as nisarVel.

    Contains basic functionality to track xy coordinates and interpolate
    images give by myVariables. For example nisarVel as vx,vy,ex,ey, To
    interpolate just vx and vy pass myVars=['vx','vy'] to child. '''

    __metaclass__ = ABCMeta

    def __init__(self,  sx=None, sy=None, x0=None, y0=None, dx=None, dy=None,
                 verbose=True, epsg=None, numWorkers=4):
        ''' initialize a nisar velocity object'''
        self.sx, self.sy = sx, sy  # Image size in pixels
        self.x0, self.y0 = x0, y0  # Origin (center of lower left pixel) in m
        self.dx, self.dy = dx, dy  # Pixel size
        self.xx, self.yy = [], []  # Coordinate of image axis ( F(y,x))
        self.xGrid, self.yGrid = None, None  # Grid of coordinate for each pix
        self.verbose = verbose  # Print progress messages
        self.epsg = epsg  # EPSG (sussm 3031 and 3413; should work for others)
        self.xr = None
        self.subset = None
        self.variables = None
        self.flipY = True
        self.dtype = 'float32'
        self.useStackstac = False
        self.xforms = {}  # Cached pyproj transforms
        dask.config.set(num_workers=numWorkers)

    #
    # ---- Setup abtract methods for child classes
    #

    @abstractmethod
    def interp():
        ''' Abstract interpolation method - define in child class. '''
        pass

    @abstractmethod
    def readDataFromTiff():
        ''' Abstract read geotiff(s) - define in child class.'''
        pass

    @abstractmethod
    def writeDataToTiff():
        ''' Abstract write geotiff(s) - define in child class.'''
        pass

    @abstractmethod
    def reproduce():
        ''' Abstract reproduce to create a new version of this class.'''
        pass

    #
    # ---- Input and array creation
    #

    def readXR(self, fileNameBase, url=False, masked=False, stackVar=None,
               time=None, time1=None, time2=None, xrName='None', skip=[],
               overviewLevel=None, suffix=''):
        ''' Use read data into rioxarray variable
        Parameters
        ----------
        fileNameBase = str
            template with filename firstpart_*_secondpart (no .tif)
            the * will be replaced with the different compnents (e.g., vx,vy)
        url bool, optional
            Set true if fileNameBase is a link
        time : datetime, optional
            nominal center. The default is None.
        time1 : TYPE, optional
            Nominal start time. The default is None.
        time2 : TYPE, optional
            Nominal endtie. The default is None.
        xrName : str, optional
            Name for xarray. The default is None
        overviewLevel: int
            Overview (pyramid) level to read: None->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level
        suffix : str, optional
            Any suffix that needs to be appended (e.g., for dropbox links)
        '''
        # Do a lazy open on the tiffs
        if stackVar is None:
            myXR = self._lazy_openTiff(fileNameBase, url=url, masked=masked,
                                       time=time, xrName=xrName, skip=skip,
                                       overviewLevel=overviewLevel,
                                       suffix=suffix)
        else:  # Not debugged
            items = self._construct_stac_items(fileNameBase, stackVar,
                                               xrName=xrName, url=url,
                                               skip=skip)
            myXR = self._lazy_open_stack(items, stackVar,
                                         overviewLevel=overviewLevel)
        # Initialize array
        self.initXR(myXR, time=time, time1=time1, time2=time2)

    def initXR(self, XR, time=None, time1=None, time2=None, xrName='None'):
        '''
        Setup class using passed xarray either input from file or passed
        directly.
        Parameters
        ----------
        XR : xarray
            xarray with desired variables
        time : datetime, optional
            nominal center. The default is None.
        time1 : TYPE, optional
            Nominal start time. The default is None.
        time2 : TYPE, optional
            Nominal endtie. The default is None.
        Returns
        -------
        None
        '''
        if self.variables is None:
            self.variables = list(XR.band.data)
        # initially subset=full
        self.xr = XR
        self.subset = XR
        # get the geoinfo
        self._parseGeoInfo()
        #
        if time1 is not None and 'time1' not in list(self.xr.coords):
            self.xr['time1'] = time1
        if time2 is not None and 'time2' not in list(self.xr.coords):
            self.xr['time2'] = time2
        #
        self.xr.rename(xrName)
        # Save variables to (self.vx, self.vy)
        self._mapVariables()

    def readFromNetCDF(self, cdfFile):
        '''
        Load data from netcdf file
        Parameters
        ----------
        cdfFile : str
            NetCDF file name.
        Returns
        -------
        None.
        '''
        if '.nc' not in cdfFile:
            cdfFile = f'{cdfFile}.nc'
        xDS = xr.open_dataset(cdfFile)
        xDS['time1'] = xDS.time1.compute()
        xDS['time2'] = xDS.time2.compute()
        # Pull the first variable that is not spatial_ref
        for var in list(xDS.data_vars.keys()):
            if var != 'spatial_ref':
                self.xr = xDS[var]
                break
        try:
            self.xr['spatial_ref'] = xDS['spatial_ref']
        except Exception:
            print('warning missing spatial_ref')
        #
        self.subset = self.xr  # subset is whole array at this point.
        self._parseGeoInfo()
        self._mapVariables()

    def subSetData(self, bbox):
        ''' Subset dataArray using a box to crop limits
        Parameters
        ----------
        bbox, dict
            crop area {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        '''
        # trap potential errors
        xrBox = self._xrBoundingBox(self.xr)
        if bbox['minx'] > xrBox['maxx'] or bbox['maxx'] < xrBox['minx'] or \
           bbox['miny'] > xrBox['maxy'] or bbox['maxy'] < xrBox['miny']:
            print('Crop failed: Subset region does not overlap data')
            return
        # Crop the data
        self.subset = self.xr.rio.clip_box(**bbox)
        # Save variables
        self._mapVariables()
        self._parseGeoInfo()

    def loadRemote(self):
        ''' Load the current XR, either full array if not subset,
        or the subsetted version '''
        self.subset.load()
        self._mapVariables()

    def _construct_stac_items(self, fileNameBase, stackVar, url=True, skip=[]):
        ''' construct STAC-style dictionaries of CMR urls for stackstac
        Currently not debugged
        '''
        bandTiff = fileNameBase
        option = '?list_dir=no'
        if url:
            bandTiff = f'/vsicurl/{option}&url={fileNameBase}'
        collection = bandTiff.split('/')[-3].replace('*', 'vv')
        myId = os.path.basename(fileNameBase).replace('*', 'vv')
        item = {'id': myId,
                'collection': collection,
                'properties': {'datetime': datetime.strptime(
                    fileNameBase.split('/')[-2], '%Y.%m.%d').isoformat()},
                'assets': {},
                'bbox': stackVar['bbox']
                }
        for band in self.variables:
            if band in skip:
                continue
            item['assets'][band] = {'href':
                                    bandTiff.replace('*', band) + '.tif',
                                    'type': 'application/x-geotiff'}
        return [item]

    def _lazy_open_stack(self, items, stackVar):
        ''' return stackstac xarray dataarray - not debugged '''
        print('this method should not be used until problems with '
              'stackstac resolved')
        return
        fill_value = np.nan
        self.useStackstac = True
        #
        self.epgs = stackVar['epsg']
        da = stackstac.stack(items,
                             assets=self.variables,
                             epsg=stackVar['epsg'],
                             resolution=stackVar['resolution'],
                             fill_value=fill_value,
                             dtype=self.dtype,
                             xy_coords='center',
                             chunksize=CHUNKSIZE,
                             bounds=stackVar['bounds']
                             )
        da = da.rio.write_crs(f'epsg:{stackVar["epsg"]}')
        return da

    # @dask.delayed
    def _lazy_openTiff(self, fileNameBase, masked=True, url=False, time=None,
                       xrName='None', skip=[], overviewLevel=None, suffix=''):
        '''
        Lazy open of a single velocity product
        Parameters
        ----------
        fileNameBase, str
            template with filename firstpart_*_secondpart (no .tif)

        Parameters
        ----------
        fileNameBase, str
            template with filename firstpart_*_secondpart (no .tif)
        masked : bool, optional
            Set no data to nan if true. The default is True.
        url : bool, optional
            Avoids directory read when using urls. The default is False.
        time : datetime, optional
            The time for the product. The default is None.
        xrName : str, optional
            Name for xarray. The default is None
        Returns
        -------
        xarray
            xarray with the data from the tiff.
        '''
        das = []
        chunks = {'band': 1, 'y': CHUNKSIZE, 'x': CHUNKSIZE}
        option = '?list_dir=no'  # Saves time by not reading full directory
        # Read individual bands
        for band in self.variables:
            if band in skip:
                continue
            bandTiff = fileNameBase
            if url:
                bandTiff = f'/vsicurl/{option}&url={fileNameBase}'
            bandTiff = bandTiff.replace('*', band) + '.tif' + suffix
            # read file via rioxarry
            da = rioxarray.open_rasterio(bandTiff, lock=True,
                                         default_name=fileNameBase,
                                         chunks=chunks, masked=masked,
                                         overview_level=overviewLevel)
            # Process time dim
            if time is not None:
                da = da.expand_dims(dim='time')
                da['time'] = [time]
            da['band'] = [band]
            da['name'] = xrName
            da['_FillValue'] = self.noDataDict[band]
            das.append(da)
        # Concatenate bands (components)
        return xr.concat(das, dim='band', join='override',
                         combine_attrs='drop')

    def _mapVariables(self):
        ''' Map the xr variables to band variables (e.g., 'vx' -> self.vx) '''
        # Map variables
        for myVar in self.subset.band.data:
            myVar = str(myVar)
            bandData = np.squeeze(self.subset.sel(band=myVar).data)
            setattr(self, myVar, bandData)

    def _parseGeoInfoStack(self):
        ''' parse geoinfo for stackstac - do not use until stac fixed'''
        myXr = self.subset
        # get epsg
        self.epsg = int(myXr.crs.split(":")[1])
        gT = myXr.transform
        self.dx, self.dy = abs(gT[0]), abs(gT[4])
        #
        self._computeCoords(myXr)

    def _parseGeoInfo(self):
        '''Parse geo info out of xr spatial ref'''
        myXr = self.subset
        # Get wkt and convert to crs
        self.wkt = myXr.spatial_ref.attrs['spatial_ref']
        myCRS = pyproj.CRS.from_wkt(self.wkt)
        # Save epsg
        self.epsg = myCRS.to_epsg()
        # get lower left corner, sx, sy, dx, dy
        gT = list(myXr.rio.transform())
        self.dx, self.dy = abs(gT[0]), abs(gT[4])
        self._computeCoords(myXr)

    def timeSliceData(self, date1, date2):
        '''
        Extract a subSeries from an existing series and return a new series
        spanning date1 to date2.

        Parameters
        ----------
        date1 : Tdatetime or "YYYY-MM-DD"
            First date in range.
        date2 : TYPE
            Second date in range.

        Returns
        -------
        series of current class

        '''
        # Ensure date time
        date1, date2 = self.parseDate(date1), self.parseDate(date2)
        # Create new instance
        newSubset = self.reproduce()
        newXR = self.xr.sel(time=slice(date1, date2))
        # Add data
        newSubset.initXR(newXR)
        # Get current bounding box and subset
        bbox = self._xrBoundingBox(self.subset)
        newSubset.subSetData(bbox)
        # Return result
        return newSubset

    def _getTimes(self):
        ''' Load times from Xarray for instances that track start and times'''
        for time in ['time', 'time1', 'time2']:
            if len(getattr(self.xr, time).data.shape) == 0:
                setattr(self, time,
                        [self.datetime64ToDatetime(np.datetime64(
                            getattr(self.xr, time).item(), 'ns'))])
            else:
                setattr(self, time, [self.datetime64ToDatetime(x)
                                     for x in getattr(self.xr, time).data])

    def _setBandOrder(self, bandOrder):
        '''
        Ensure band order is correct
        Returns
        -------
        None.
        '''
        bandOrderList = []
        # Create list for sort
        for b in self.xr.band:
            bandOrderList.append(bandOrder[b.item()])
        # creete array for sort
        myOrderXR = xr.DataArray(bandOrderList, coords=[self.xr['band']],
                                 dims='band')
        # do sort
        self.xr = self.xr.sortby(myOrderXR)

    #
    # ---- def setup xy coordinates
    #

    def xyGrid(self):
        '''
        Computer grid version of coordinates.
        Save as 2-D arrays: self.xGrid, self.yGrid
        Returns
        -------
        None.
        '''
        # Computer 1-D (xx,yy) coordinates if not computed.
        if self.xx is None or self.sx is None:
            print('xyGrid: geolocation variables not defined')
            return
        # setup arrays
        self.xGrid, self.yGrid = np.meshgrid(self.xx, self.yy, indexing='xy')

    # ------------------------------------------------------------------------
    # Projection routines
    # ------------------------------------------------------------------------

    def computePixEdgeCornersXYM(self, units='m'):
        '''
        Return dictionary with corner locations. Note unlike pixel
        centered xx, x0 etc values, these corners are defined as the outer
        edges as in a geotiff
        Returns
        -------
        corners : dict
            corners in xy coordinates: {'ll': {'x': xll, 'y': yll}...}.
        '''
        # Make sure geometry defined
        params = ['sx', 'sy', 'x0', 'y0', 'dx', 'dy']
        if None in [getattr(self, x) for x in params]:
            myError(f'Geometry param not defined size {self.nx,self.ny}, '
                    'origin {self.x0,self.y0}), or pix size {self.dx,self.dy}')
        # compute pixel corners from pixel centers
        self.checkUnits(units)
        #
        xll, yll = self.x0 - self.dx/2, self.y0 - self.dx/2
        xur, yur = xll + self.sx * self.dx, yll + self.sy * self.dy
        if units == 'km':
            xll, yll = self._toKM(xll, yll)
            xur, yur = self._toKM(xur, yur)
        xul, yul = xll, yur
        xlr, ylr = xur, yll
        corners = {'ll': {'x': xll, 'y': yll}, 'lr': {'x': xlr, 'y': ylr},
                   'ur': {'x': xur, 'y': yur}, 'ul': {'x': xul, 'y': yul}}
        return corners

    def computePixEdgeCornersLL(self):
        '''
        Return dictionary with corner locations in lat/lon. Note unlike pixel
        centered values, these corners are defined as the outer
        edges as in a geotiff
        Returns
        -------
        llcorners : dict
            corners in lat/lon coordinates:
                {'ll': {'lat': latll, 'lon': lonll}...}.
        '''
        # compute xy corners
        corners = self.computePixEdgeCornersXYM()
        xytollXform = pyproj.Transformer.from_crs(f"EPSG:{self.epsg}",
                                                  "EPSG:4326")
        llcorners = {}
        # Loop to do coordinate transforms.
        for myKey in corners.keys():
            lat, lon = xytollXform.transform(np.array([corners[myKey]['x']]),
                                             np.array([corners[myKey]['y']]))
            llcorners[myKey] = {'lat': lat[0], 'lon': lon[0]}
        return llcorners

    def getDomain(self, epsg):
        '''
        Return names of domain name based on epsg (antartica or greenland).
        Here for historical reasons only.
        Parameters
        ----------
        epsg : int
            epsg code.
        Returns
        -------
        domain : str
            For now either 'greenland' or 'antarctica'.
        '''
        if epsg is None or epsg == 3413:
            domain = 'greenland'
        elif epsg == 3031:
            domain = 'antarctica'
        else:
            myError('Unexpected epsg code: '+str(epsg))
        return domain

    #
    # ---- Interpolate results
    #

    def _checkUnits(self, units):
        '''
        Check units return True for valid units. Print message for invalid.
        '''
        if units not in ['m', 'km']:
            print(f'Invalid units: must be m or km not {units}, defaulting '
                  'to m')
            return False
        return True

    def interpGeo(self, x, y, myVars, date=None, returnXR=False, units='m',
                  **kwargs):
        '''
        Call appropriate interpolation method for each variable specified by
        myVars
        Parameters
        ----------
        x : np float array
            x coordinates in m to interpolate to.
        y : np float array
            y coordinates in m to interpolate to.
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        **kwargs : TBD
            keywords passed through to interpolator.
        Returns
        -------
        np float array
            Interpolated valutes from x,y locations.
        '''
        if not self._checkUnits(units):
            return
        return self._interpNP(x, y, myVars, date=date, returnXR=returnXR,
                              units=units, **kwargs)

    def _convertCoordinates(func):
        '''
        Wrap interpolators orother functions so that the input coordinates
        (e.g. lat lon) will automatically be mapped to xy as specificed by
        Parameters
        ----------
        func : function
            function to be decorated
        Returns
        -------
        function result
        '''
        @functools.wraps(func)
        def convertCoordsInner(inst, xin, yin, *args, units='m',
                               sourceEPSG=None, **kwargs):
            '''
            Coordinate conversion
            Parameters
            ----------
            inst : self
                Passed in self.
            xin, yin : scalar or nparray
                Input coordinates as x, y or lat, lon coords.
            *args : optional args of wrapped function
            units : str, optional
                Input units. The default is 'm'.
            sourceEPSG : str or int, optional
                EPSG code for x, y. The default is None (no xform).
            **kwargs : dict
                Keywords to pass through to function.
            Returns
            -------
            TBD
                Results from wrapped function
            '''
            # No source EPSG so just pass original coords back
            if sourceEPSG is None:
                xout, yout = xin, yin
            else:
                if sourceEPSG not in inst.xforms:  # Calc. xform if not cached
                    inst.xforms[str(sourceEPSG)] = \
                        pyproj.Transformer.from_crs(f"EPSG:{sourceEPSG}",
                                                    f"EPSG:{inst.epsg}")
                # Transform coordates
                xout, yout = inst.xforms[str(sourceEPSG)].transform(xin, yin)
            # If km, convert to m for internal calcs
            if units == 'km':
                xout, yout = xout * 1000, yout * 1000.
            return func(inst, xout, yout, *args, **kwargs)
        return convertCoordsInner

    @_convertCoordinates
    def _interpNP(self, x, y, myVars, date=None, returnXR=False):
        '''
        Interpolate myVar(y,x) specified myVars (e.g., ['vx'..]) where myVars
        are nparrays. Linear interpolation in space and nearest neighbor in
        time.
        Parameters
        ----------
        x: np float array
            x coordinates in m to interpolate to.
        y: np float array
            y coordinates in m to interpolate to.
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        date: str (YYYY-mm-dd), or datetime optional
            Interpolate for data nearest date, None returns all times
        units : str, optional
                Input units. The default is 'm'.
        sourceEPSG : str or int, optional
                EPSG code for x, y. The default is None (no xform).
        Returns
        -------
        result : float np array with shape (band, time, pt) for data is None
            or with shape (band, pt) for specific date
            Interpolated valutes from x,y locations.
        '''
        if np.isscalar(x):
            x, y = [x], [y]
        xx1, yy1 = xr.DataArray(x), xr.DataArray(y)
        myXR = self.subset
        # Array to receive results
        date = self.parseDate(date, defaultDate=False)
        if date is not None:
            myXR = self.subset.sel(time=date, method='nearest')
        #
        result = myXR.interp(x=xx1, y=yy1)
        if date is None:
            result = result.transpose('band', 'time', 'dim_0')
        if returnXR:
            return result
        else:
            return np.squeeze(
                [result.sel(band=[band]).data for band in myVars])

    #
    # ---- Operations on data (e.g., mean, std)
    #

    def meanXY(self, returnXR=False):
        '''
        Return the mean for each spatial layer at each time
        Parameters
        ----------
        returnXR : bool, optional
            Return xarray if True else numpy. The default is False.
        Returns
        -------
        nparray, or xarray depending on
            mean for each time and band.
        '''
        result = self.subset.mean(dim=['x', 'y'])
        if returnXR:
            return result
        return result.data.transpose()

    def stdevXY(self, returnXR=False):
        '''
        Return sigma for each spatial layer at each time
        Parameters
        ----------
        returnXR : bool, optional
            Return xarray if True else numpy. The default is False.
        Returns
        -------
        nparray, or xarray depending on
            mean for each time and band.
        '''
        result = self.subset.std(dim=['x', 'y'])
        if returnXR:
            return result
        return result.data.transpose()

    def _applyInTime(func):
        '''
        Decorator to apply a function in time and return a new nisar object
        '''
        @functools.wraps(func)
        def _applyInTimeInner(inst, *args, **kwargs):
            myXR = func(inst, *args, **kwargs)
            if 'time' not in myXR.dims:
                myXR = myXR.expand_dims(dim='time')
                meanTime = np.datetime64(inst.xr.time.mean().item(), 'ns')
                myXR['time'] = [meanTime]
            # myXR['time1'] = inst.subset.time1.data[0]
            # myXR['time2'] = inst.subset.time2.data[-1]
            # Init mean instance
            result = inst.reproduce()
            result.initXR(myXR, time1=inst.subset.time1.data[0],
                          time2=inst.subset.time2.data[-1])
            if len(myXR.time) > 1:
                result.time1 = [x for x in inst.time1]
                result.time2 = [x for x in inst.time2]
            else:
                result.time1 = inst.time1[0]
                result.time2 = inst.time2[-1]
            result.time = [inst.datetime64ToDatetime(x) for x in
                           myXR.time.data]
            return result
        return _applyInTimeInner

    @_applyInTime
    def anomaly(self):
        #
        myMean = self.subset.mean(dim='time')
        myAnomalyXR = xr.concat(
            [self.subset.sel(time=t) - myMean for t in self.subset.time],
            dim='time', join='override', combine_attrs='drop')
        # fix coordinate order
        myAnomalyXR = myAnomalyXR.transpose('time', 'band', 'y', 'x')
        myAnomalyXR['time1'] = self.xr.time1.copy()
        myAnomalyXR['time2'] = self.xr.time2.copy()
        return myAnomalyXR

    @_applyInTime
    def mean(self, skipna=True):
        '''
        Compute mean along time axis and return new instance of same class
        Note that the original xr and subset correspond to the subset
        of the calling instance.
        Parameters
        ----------
        skipna : bool, optional
            Skips nans in computation. The default is True.

        Returns
        -------
        same as class method called from
            Object with mean and time axis reduced to dimension of 1.

        '''
        return self.subset.mean(dim='time', skipna=skipna)

    @_applyInTime
    def stdev(self, skipna=True):
        '''
        Compute standard deviation along time axis and return new instance of
        same class.
        Note that the original xr and subset correspond to the subset
        of the calling instance.
        Parameters
        ----------
        skipna : bool, optional
            Skips nans in computation. The default is True.

        Returns
        -------
        same as class method called from
            Object standard dev with time axis reduced to dimension of 1.

        '''
        return self.subset.std(dim='time', skipna=skipna)

    @_applyInTime
    def numberValid(self):
        '''
        Compute count of valid data along time axis and return new instance of
        same class.
        Note that the original xr and subset correspond to the subset
        of the calling instance.
        Parameters
        ----------
        skipna : bool, optional
            Skips nans in computation. The default is True.

        Returns
        -------
        same as class method called from
            Object valid-data count with time axis reduced to dimension of 1.
        return
        '''
        return self.subset.notnull().sum(dim='time')

    #
    # ---- time/date related routines
    #

    def parseDate(self, date, defaultDate=True):
        ''' Accept date as either datetime or YYYY-MM-DD
        Parameters
        ----------
        date, str or datetime
        Returns
        -------
        date in as datetime instance
        '''
        if date is None:
            if defaultDate:
                return np.datetime64(self.subset.time.item(0), 'ns')
            else:
                return None
        try:
            if type(date) == str:
                date = datetime.strptime(date, '%Y-%m-%d')
        except Exception:
            print('Error: Either invalid date (datetime or "YYYY-MM-DD")')
            print(f'Or date outside range: {min(self.subset.date)}, '
                  f'{max(self.subset.date)}')
        return date

    def _dates(self, date, asString=False):
        '''
        Extract dates for a layer based on current date
        Parameters
        ----------
        date : datetime
            extract the date info closest to this date.
        asString : bool, optional
            Return datetime if true else str. The default is False.

        Returns
        -------
        midDate : datetime or str
            The middle date for the data set.
        date1, date2 : datetime or str
            The first and last dates for the data set.
        '''
        date1 = self.datetime64ToDatetime(
            self.subset.time.sel(time=date, method='nearest').time1.data)
        date2 = self.datetime64ToDatetime(
            self.subset.time.sel(time=date, method='nearest').time2.data)
        midDate = self.datetime64ToDatetime(
            self.subset.time.sel(time=date, method='nearest').data)
        if asString:
            date1 = date1.strftime('%Y-%m-%d')
            date2 = date2.strftime('%Y-%m-%d')
            midDate = midDate.strftime('%Y-%m-%d')
        return midDate, date1, date2

    def datetime64ToDatetime(self, date64):
        return datetime.strptime(np.datetime_as_string(date64)[0:19],
                                 "%Y-%m-%dT%H:%M:%S")

    # ----------------------------------------------------------------------
    # Read geometry info
    # ----------------------------------------------------------------------

    def _computeCoords(self, myXr):
        self.sy, self.sx = myXr.shape[2:]
        self.x0 = np.min(myXr.x).item()
        self.y0 = np.min(myXr.y).item()
        self.xx = myXr.x.data
        self.yy = myXr.y.data

    #
    # ---- Plotting and display
    #

    def colorSetup(self, scale, cmap, vmin, vmax):
        '''
        Set up normalization and color table for imshow
        Parameters
        ----------
        scale : str
            scale
        cmap : color map
            color map.
        Returns
        -------
        norm, normalization
        cmap, color ma
        '''
        if scale == 'log':
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            # Default color map for log is truncated hsv
            if cmap is None:
                cmap = colors.LinearSegmentedColormap.from_list(
                    'myMap', cm.hsv(np.linspace(0.1, 1, 250)))
            return norm, cmap
        # Pass back colormap for linear case
        elif scale == 'linear':
            return colors.Normalize(vmin=vmin, vmax=vmax), cmap
        print('Invalid scale mode. Choices are "linear" and "log"')

    def autoScaleRange(self, band, date, vmin, vmax, percentile, quantize=100):
        '''
        If percentile less than 100, will select vmin as (100-percentile)
        and vmax as percentile of myVar, unless they fall out of the vmin and
        vmax bounds.
        Parameters
        ----------
        myVar : nparray
            Data being displayed.
        vmin : float
            Absolute minimum value.
        vmax : TYPE
            absolute maximum value.
        percentile : TYPE
            Clip data at (100-percentile) and percentile.

        Returns
        -------
        vmin, vmap - updated values based on percentiles.
        '''
        # compute max
        date = self.parseDate(date)
        if date is not None:
            myVar = self.subset.sel(time=date,
                                    method='nearest').sel(band=band).data
        else:
            myVar = self.subset.sel(band=band).data
        #
        maxVel = min(np.percentile(myVar[np.isfinite(myVar)], percentile),
                     vmax)
        #
        vmax = math.ceil(maxVel/quantize)*quantize
        minVel = max(np.percentile(myVar[np.isfinite(myVar)],
                                   100 - percentile), vmin)
        vmin = math.floor(minVel/quantize) * quantize
        return vmin, vmax

    def _createDivider(self, ax, colorBarPosition='right', colorBarSize='5%',
                       colorBarPad=0.05):
        '''
        Create divider for color bar
        '''
        divider = make_axes_locatable(ax)  # Create space for colorbar
        return divider.append_axes(colorBarPosition, size=colorBarSize,
                                   pad=colorBarPad)

    def _colorBar(self, pos, ax, colorBarLabel, colorBarPosition, colorBarSize,
                  colorBarPad, labelFontSize, plotFontSize):
        '''
        Color bar for image
        '''
        # Create an divided axis for cb
        cbAx = self._createDivider(ax, colorBarPosition=colorBarPosition,
                                   colorBarSize=colorBarSize,
                                   colorBarPad=colorBarPad)
        # Select orientation
        orientation = {'right': 'vertical', 'left': 'vertical',
                       'top': 'horizontal',
                       'bottom': 'horizontal'}[colorBarPosition]

        cb = plt.colorbar(pos, cax=cbAx, orientation=orientation, extend='max')
        cb.set_label(colorBarLabel, size=labelFontSize)
        cb.ax.tick_params(labelsize=plotFontSize)
        if colorBarPosition in ['right', 'left']:
            cbAx.yaxis.set_ticks_position(colorBarPosition)
            cbAx.yaxis.set_label_position(colorBarPosition)
        elif colorBarPosition in ['top', 'tottom']:
            cbAx.xaxis.set_ticks_position(colorBarPosition)
            cbAx.xaxis.set_label_position(colorBarPosition)

    def displayVar(self, var, date=None, ax=None, plotFontSize=14,
                   colorBar=True,
                   labelFontSize=12, titleFontSize=15, axisOff=False,
                   vmin=0, vmax=7000, units='m', scale='linear', cmap=None,
                   title=None, midDate=True, colorBarLabel='Speed (m/yr)',
                   masked=None, colorBarPosition='right', colorBarSize='5%',
                   colorBarPad=0.05, wrap=None,
                   **kwargs):
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
        wrap :  number, optional
            Display velocity modululo wrap value
        Returns
        -------
        fig : matplot lib fig
            Figure used for plot.
        ax : matplot lib ax
            Axis used for plot.
        '''
        if not self._checkUnits(units):
            return
        if ax is None:
            sx, sy = self.sizeInPixels()
            fig, ax = plt.subplots(1, 1, constrained_layout=True,
                                   figsize=(10.*float(sx)/float(sy), 10.))
        norm, cmap = self.colorSetup(scale, cmap, vmin, vmax)
        # Return if invalid var
        if var not in self.variables:
            print(f'{var} is not a valid, the choices are {self.variables}')
            return
        # Extract data for band
        displayVar = self.subset.sel(band=var)
        # Extract date for time
        date = self.parseDate(date)
        displayVar = displayVar.sel(time=date, method='nearest')
        # Display the data
        displayVar = np.squeeze(displayVar)
        if wrap is not None:
            displayVar = np.mod(displayVar, wrap)

        pos = ax.imshow(np.ma.masked_where(displayVar == masked, displayVar,
                                           copy=True), norm=norm, cmap=cmap,
                        extent=self.extent(units=units), **kwargs)
        # labels and such
        if axisOff:
            ax.axis('off')
        else:
            ax.set_xlabel(f'X ({units})', size=labelFontSize)
            ax.set_ylabel(f'Y ({units})', size=labelFontSize)
            ax.tick_params(axis='x', labelsize=plotFontSize)
            ax.tick_params(axis='y', labelsize=plotFontSize)
        # Create title from dates
        if title is None:
            middleDate, date1, date2 = self._dates(date, asString=True)
            if midDate:
                title = middleDate
            else:
                title = f'{date1} - {date2}'
        ax.set_title(title, fontsize=titleFontSize)
        # labels and such.
        if colorBar:
            self._colorBar(pos, ax, colorBarLabel, colorBarPosition,
                           colorBarSize, colorBarPad, labelFontSize,
                           plotFontSize)

        return ax

    #
    # ---- Return  geometry params in m and km
    #

    def sizeInPixels(self):
        '''
        Return size in pixels
        Returns
        -------
        sx, sx: int
            x, y size
        '''
        return self.sx, self.sy

    def _toKM(self, x, y):
        '''
        Conversion to kilometers
        Parameters
        ----------
        x, y : np.array or scaler
            x, y coordinates in m.
        Returns
        -------
        x,y : same as input
            x, y coordinates in km.
        '''
        return x/1000., y/1000.

    def size(self, units='m'):
        ''' Return size in meters '''
        self._checkUnits(units)
        if units == 'km':
            return self._toKM(self.sx*self.dx, self.sy*self.dy)
        return self.sx*self.dx, self.sy*self.dy

    def outline(self, units='m'):
        '''
        Return x, y coordinate of images exent to plot as box.

        Parameters
        ----------
        units : str, optional
            units. The default is 'm'.

        Returns
        -------
        x, y coordinates of outline
        '''
        self._checkUnits(units)
        bbox = self._xrBoundingBox(self.subset, units=units)
        xbox = np.array([bbox[x] for x in ['minx', 'minx', 'maxx',
                                           'maxx', 'minx']])
        ybox = np.array([bbox[y] for y in ['miny', 'maxy', 'maxy',
                                           'miny', 'miny']])
        return xbox, ybox

    def origin(self, units='m'):
        ''' Return origin in meters (default) or km '''
        self._checkUnits(units)
        if units == 'km':
            return self._toKM(self.x0, self.y0)
        return self.x0, self.y0

    def bounds(self, units='m'):
        '''
        Determine data bounds in meters (default) or km
        Returns
        -------
        xmin : float
            min x (lower left) coordinate.
        ymin : float
            min y (lower left) coordinate.
        xmax : float
            max x (upper right) coordinate.
        ymax : float
            max y (upper right) coordinate.
        '''
        self._checkUnits(units)
        xmax = (self.x0 + (self.sx - 1) * self.dx)
        ymax = (self.y0 + (self.sy-1) * self.dy)
        x0, y0 = self.x0, self.y0
        if units == 'km':
            x0, y0 = self._toKM(self.x0, self.y0)
            xmax, ymax = self._toKM(xmax, ymax)
        return x0, y0, xmax, ymax

    def extent(self, units='m'):
        '''
        Determine extent [xmin, xmax, ymin, ymax]
        Returns
        -------
        xmin : float
            min x (lower left) coordinate.
        xmax : float
            max x (upper right) coordinate.
        ymin : float
            min y (lower left) coordinate.
        ymax : float
            max y (upper right) coordinate.
        '''
        self._checkUnits(units)
        bounds = self.bounds(units=units)
        return bounds[0], bounds[2], bounds[1], bounds[3]

    def _xrBoundingBox(self, myXR, units='m'):
        ''' bounding box for an xr '''
        self._checkUnits(units)
        extremes = [np.min(myXR.x.data), np.max(myXR.x.data),
                    np.min(myXR.y.data), np.max(myXR.y.data)]
        if units == 'km':
            extremes = [x * 0.001 for x in extremes]
        return dict(zip(['minx', 'maxx', 'miny', 'maxy'], extremes))

    def boundingBox(self, units='m'):
        '''
        Return a bounding box used to crop
        Parameters
        ----------
        unit : TYPE, optional
            DESCRIPTION. The default is 'm'.

        Returns
        -------
        dict
           {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}.
        '''
        return dict(zip(['minx', 'maxx', 'miny', 'maxy'],
                        self.extent(units=units)))

    def pixSize(self, units='m'):
        '''
        Return pixel size in m
        Returns
        -------
        dx : float
            pixel size in x dimension.
        dy : float
            pixel size in y dimension.
        '''
        self._checkUnits(units)
        if units == 'km':
            return self._toKM(self.dx, self.dy)
        return self.dx, self.dy

    #
    # ---- Output results
    #

    def writeCloudOptGeo(self, tiffRoot, full=False, myVars=None, **kwargs):
        '''
        Write a cloud optimized geotiff(s) with overviews if requested.
        By default writes all bands to indiviual files.
        Write individual bands with
        Parameters
        ----------
        tiffFile : str
            Root name of tiff files
                myFile.*.X.tif or  myFile.*.X  -> myFile.myVar.X.tif
                myFile or myFile.tif -> myFile.myVar.tif
        myVar : str or [str,...], optional
            Name of a single or multiple variables to save  (e.g., ".vx").
        kwargs : optional
            pass through keywords to rio.to_raster
        Returns
        -------
        None.
        '''
        # variables to write
        if myVars is None:
            myVars is self.myVars
        else:
            if type(myVars) is str:  # Ensure its a str
                myVars = [myVars]
        # By default write subset unless none exists or full flagged
        if full or self.subset is None:
            myXR = self.xr
        else:
            myXR = self.subset
        for myVar, band in zip(self.myVars, myXR):
            if myVar in myVars:
                band.rio.to_raster(self.tiffFileName(tiffRoot, myVar),
                                   **kwargs)

    def toNetCDF(self, cdfFile):
        ''' Write existing working xarray to file. If a subset exists it will
        be the subset, ow it will be the entire data set.
        Parameters
        ----------
        cdfFile : str
            netcdf file name. Will append .nc. if not present
        Returns
        -------
        None.
        '''
        if self.subset is None:
            print('No subset present - set bbox={"minxx"...}')
            return
        if '.nc' not in cdfFile:
            cdfFile = f'{cdfFile}.nc'
        if os.path.exists(cdfFile):
            os.remove(cdfFile)
        #
        self.subset.to_netcdf(path=cdfFile)
        return cdfFile

    def tiffFileName(self, tiffRoot, myVar):
        ''' Create tiff Name from root and var type
        if "*" in name, replace with myVar, otherwise append ".myVar"
        append ".tif" if not already present
        Write a cloud optimized geotiff with overviews if requested.
        Parameters
        ----------
        tiffFile : str
            Root name for output tiff file.
        myVar : str
            Name of variable being written
        '''
        tiffRoot = tiffRoot.replace('.tif', '')
        if '*' in tiffRoot:
            tiffName = tiffRoot.replace('*', myVar)
        else:
            tiffName = f'{tiffRoot}.{myVar}'
        return f'{tiffName}.tif'
