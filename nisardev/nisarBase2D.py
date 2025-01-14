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
import xarray
import pyproj
import dask
import rioxarray
import stackstac
import os
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from datetime import datetime
import rio_stac
import pystac
import holoviews as hv
import panel as pn
from copy import deepcopy
# from dask.diagnostics import ProgressBar
import math
from bokeh.models.formatters import DatetimeTickFormatter

overviewLevels = dict(zip(range(-1, 10), 2**np.arange(0, 11)))


class nisarBase2D():
    ''' Abstract class to define xy polar stereo (3031, 3413) image objects
    such as nisarVel.

    Contains basic functionality to track xy coordinates and interpolate
    images give by myVariables. For example nisarVel as vx,vy,ex,ey, To
    interpolate just vx and vy pass myVars=['vx','vy'] to child. '''

    __metaclass__ = ABCMeta

    def __init__(self,  sx=None, sy=None, x0=None, y0=None, dx=None, dy=None,
                 verbose=True, epsg=None, numWorkers=4, stackTemplate=None):
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
        self.stackTemplate = stackTemplate
        dask.config.set(num_workers=numWorkers)

    #
    # ---- Setup abtract methods for child classes
    #

    @abstractmethod
    def plotPoint():
        ''' Abstract pointPlot method - define in child class. '''
        pass

    @abstractmethod
    def plotProfile():
        ''' Abstract plotProfile method - define in child class. '''
        pass

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

    @abstractmethod
    def inspectData():
        ''' Abstract inspect to create a new version of this class.'''
        pass

    #
    # ---- Input and array creation
    #

    def readXR(self, fileNameBase, url=False, masked=False, useStack=False,
               time=None, time1=None, time2=None, xrName='None', skip=[],
               overviewLevel=-1, suffix='', chunkSize=1024, fill_value=np.nan):
        ''' Use read data into rioxarray variable
        Parameters
        ----------
        fileNameBase = str
            template with filename firstpart_*_secondpart (no .tif)
            the * will be replaced with the different compnents (e.g., vx,vy)
        url bool, optional
            Set true if fileNameBase is a link
        masked : boolean, optional.
            Masked keyword to lazy_open_tiff. The default is False
        useStack : boolean, optional
            Uses stackstac for full resolution data. The default is True.
        time : datetime, optional
            nominal center. The default is None.
        time1 : TYPE, optional
            Nominal start time. The default is None.
        time2 : TYPE, optional
            Nominal endtie. The default is None.
        xrName : str, optional
            Name for xarray. The default is None
        overviewLevel: int
            Overview (pyramid) level to read: -1->full res, 0->1/2 res,
            1->1/4 res....to image dependent max downsampling level.
            The default is -1 (full res)
        suffix : str, optional
            Any suffix that needs to be appended (e.g., for dropbox links)
        '''
        # Do a lazy open on the tiffs
        # For now useStack turned off since it tries to read the full res
        # data and donwsample instead of the pyramid
        if not useStack or overviewLevel > -1:
            # print('noStack')
            myXR = self._lazy_openTiff(fileNameBase, url=url, masked=masked,
                                       time=time, xrName=xrName, skip=skip,
                                       overviewLevel=overviewLevel,
                                       suffix=suffix, chunkSize=chunkSize)
        else:  # Not debugged
            items = self._construct_stac_items(fileNameBase, time,
                                               url=url,
                                               skip=skip)
            myXR = self._lazy_open_stack(items, skip=skip,
                                         overviewLevel=overviewLevel,
                                         chunkSize=chunkSize,
                                         fill_value=fill_value)
        # Initialize array
        self.initXR(myXR, time=time, time1=time1, time2=time2,
                    useStack=useStack)

    def initXR(self, XR, time=None, time1=None, time2=None, xrName='None',
               useStack=False):
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
        xrName: str, optional
            Name for xr array. The default is 'None'

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

    def copy(self):
        '''
        Create a deep copy of current instance.
        Returns
        -------
        Copy (deep) of itself
        '''
        # Make an empty instance
        new = self.reproduce()
        # Deep copy the xr
        newXR = self.xr.copy(deep=True)
        new.initXR(newXR)
        # Get the bouding box and subset
        bbox = self._xrBoundingBox(self.subset)
        new.subSetData(bbox)
        # If the data have already been loaded, force a reload.
        if self.subset.chunks is None:
            new.subset.load()
            new._mapVariables()  # Forces remapping to non dask
        new._getTimes()
        return new

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
        self.epsg = np.int32(myXr.epsg)
        gT = list(myXr['proj:transform'].data.item())
        self.dx, self.dy = abs(gT[4]), abs(gT[5])
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
        # Open and close first to clear any old data
        xDS = xarray.open_dataset(cdfFile)
        xDS.close()
        xDS = xarray.open_dataset(cdfFile)
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

    def _get_stac_item_template(self, URL, skip=[], url=True):
        '''
        read first geotiff to get STAC Item template (returns pystac.Item)
        '''
        date = datetime(1999, 1, 1)  # dummy date
        # Create with first asset
        myVariables = [x for x in self.variables if x not in skip]
        band1 = myVariables[0]
        bandTiff = URL.replace('*', band1) + '.tif'
        if url:
            bandTiff = f'/vsicurl/?list_dir=no&url={bandTiff}'

        item = rio_stac.create_stac_item(bandTiff,
                                         input_datetime=date,
                                         asset_name=band1,
                                         asset_media_type=str(
                                             pystac.MediaType.COG),
                                         with_proj=True,
                                         with_raster=True,
                                         )
        for band in myVariables[1:]:
            newAsset = deepcopy(item.assets[band1].to_dict())
            newAsset['href'] = URL.replace('*', band) + '.tif'
            item.asset = item.add_asset(band,
                                        item.assets[band1].from_dict(newAsset))
        self.dtype = \
            item.assets[myVariables[0]].extra_fields[
                'raster:bands'][0]['data_type']
        return item

    def _construct_stac_items(self, fileNameBase, time, url=True, skip=[]):
        ''' construct STAC-style dictionaries of CMR urls for stackstac
        Currently not debugged
        '''
        if self.stackTemplate is None:
            self.stackTemplate = self._get_stac_item_template(fileNameBase,
                                                              skip=[], url=url)
        # make a copy
        item = deepcopy(self.stackTemplate)
        # update time
        item.set_datetime(time)
        # Conver to dict
        itemDict = item.to_dict()
        itemDict['id'] = fileNameBase
        #
        option = '?list_dir=no'
        # update variables
        for band in self.variables:
            if url:
                bandTiff = f'/vsicurl/{option}&url={fileNameBase}'
            else:
                bandTiff = fileNameBase
            itemDict['assets'][band]['href'] = \
                f'{bandTiff.replace("*",band)}.tif'
        #  Convert back to item type
        return item.from_dict(itemDict)

    def _lazy_open_stack(self, items, skip=[], chunkSize=512,
                         overviewLevel=-1, fill_value=np.nan):
        ''' return stackstac xarray dataarray - not debugged '''
        # fill_value = np.nan
        self.useStackstac = True
        myVariables = [x for x in self.variables if x not in skip]
        # Compute overview resolution consistent with xarray width/npix
        ny, nx = np.int64(items.properties['proj:shape'] /
                          overviewLevels[overviewLevel])
        box = items.properties['proj:bbox']
        resolution = (np.abs((box[2] - box[0]) / nx),
                      np.abs((box[3] - box[1]) / ny))
        # Create xarray using stackstac
        # fill_value = type(self.dtype)(fill_value)
        fill_value = getattr(np, self.dtype)(fill_value)
        da = stackstac.stack(items,
                             assets=myVariables,
                             fill_value=fill_value,
                             dtype=self.dtype,
                             chunksize=chunkSize,
                             snap_bounds=False,
                             xy_coords='center',
                             resolution=resolution,
                             rescale=False
                             )
        da.rio.write_crs(f'epsg:{int(da.epsg)}', inplace=True)
        da['name'] = 'temp'
        da.rio.write_crs(f'epsg:{int(da.epsg)}', inplace=True)
        return da

    # @dask.delayed
    def _lazy_openTiff(self, fileNameBase, masked=True, url=False, time=None,
                       xrName='None', skip=[], overviewLevel=None, suffix='',
                       chunkSize=1024):
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
        chunks = {'band': 1, 'y': chunkSize, 'x': chunkSize}
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
        return xarray.concat(das, dim='band', join='override',
                             combine_attrs='drop')

    def timeSliceData(self, date1, date2):
        '''
        Extract a subSeries from an existing series and return a new series
        spanning date1 to date2.

        Parameters
        ----------
        date1 : datetime or "YYYY-MM-DD"
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
        #
        # If existing subset has been loaded, load the new subset
        if self.subset.chunks is None:
            newSubset.subset.load()
            newSubset._mapVariables()
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

    def _setBandOrder(self, bandOrder, myXR=None):
        '''
        Ensure band order is correct
        Returns
        -------
        reordered xr.
        '''
        if myXR is None:
            myXR = self.xr
        bandOrderList = []
        # Create list for sort
        for b in myXR.band:
            bandOrderList.append(bandOrder[b.item()])
        # creete array for sort
        myOrderXR = xarray.DataArray(bandOrderList, coords=[myXR['band']],
                                     dims='band')
        # do sort
        return myXR.sortby(myOrderXR)

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

    def interpGeo(self, x, y, myVars, date=None, returnXR=False, grid=False,
                  units='m', **kwargs):
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
        grid : boolean, optional
            If false, interpolate at x, y values. If true create grid with
            x and y 1-d arrays for each dimension. The default is False.
        units : str ('m' or 'km'), optional
            Units. The default is 'm'.
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
                              grid=grid,
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
            result = func(inst, xout, yout, *args, **kwargs)
            if 'returnXR' in kwargs:
                if kwargs['returnXR'] and units == 'km':
                    result = result.assign_coords(x=result.x * 0.001)
                    result = result.assign_coords(y=result.y * 0.001)
            return result
        return convertCoordsInner

    @_convertCoordinates
    def _interpNP(self, x, y, myVars, date=None, grid=False, returnXR=False):
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
        grid : boolean, optional
            If false, interpolate at x, y values. If true create grid with
            x and y 1-d arrays for each dimension. The default is False.
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
            xx1, yy1 = x, y
        elif not grid:
            xx1, yy1 = xarray.DataArray(x), xarray.DataArray(y)
        else:
            xx1, yy1 = np.array(x), np.array(y)
        myXR = self.subset
        # Array to receive results
        date = self.parseDate(date, defaultDate=False)
        if date is not None:
            myXR = self.subset.sel(time=date, method='nearest')
        #
        result = myXR.interp(x=xx1, y=yy1)
        if returnXR:
            if date is None and not grid:
                # Force band be first
                dims = list(result.dims)
                if dims[0] == 'time' and dims[1] == 'band':
                    dims[0] = 'band'
                    dims[1] = 'time'
                    result = result.transpose(*dims)
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
            result._getTimes()
            # result.time = [inst.datetime64ToDatetime(x) for x in
            #               myXR.time.data]
            return result
        return _applyInTimeInner

    @_applyInTime
    def anomaly(self):
        #
        myMean = self.subset.mean(dim='time')
        myAnomalyXR = xarray.concat(
            [self.subset.sel(time=t) - myMean for t in self.subset.time],
            dim='time', join='override', combine_attrs='drop')
        # fix coordinate order
        myAnomalyXR = myAnomalyXR.transpose('time', 'band', 'y', 'x')
        myAnomalyXR['time1'] = self.xr.time1.copy()
        myAnomalyXR['time2'] = self.xr.time2.copy()
        return myAnomalyXR

    @_applyInTime
    def mean(self, skipna=True, errors=[]):
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
        #
        # If bands to compute errors for mean do so
        if len(errors) > 0:
            # Regular mean
            noErrorBands = [x for x in self.variables if x not in errors]
            meanNoErrors = self.subset.sel(
                band=noErrorBands).mean(dim='time', skipna=skipna)
            # errors sqrt(mean(sigma**2)/N)
            squaredErrors = (self.subset.sel(band=errors)**2)
            N = self.numberValid().subset.sel(band=errors)
            meanSqErrors = np.sqrt(squaredErrors.mean(dim='time') / N)
            myMean = np.squeeze(xarray.concat([meanNoErrors, meanSqErrors],
                                              dim='band',
                                              join='override',
                                              combine_attrs='drop'))
            # Make sure band order not scrambled
            bands = self.subset.band.data
            bandOrder = dict(zip(bands, np.arange(0, len(bands))))
            return self._setBandOrder(bandOrder, myMean)
        else:
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

    def parseDate(self, date, defaultDate=True, returnString=False):
        ''' Accept date as either datetime or YYYY-MM-DD
        Parameters
        ----------
        date, str or datetime
        Returns
        -------
        date as datetime instance
        '''
        if date is None:
            if defaultDate:
                return self.datetime64ToDatetime(
                    np.datetime64(self.subset.time.item(0), 'ns'))
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

    def colorSetup(self, scale, cmap, vmin, vmax, backgroundColor=(1, 1, 1)):
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
            if cmap is None or cmap == 'hsv':
                cmap = self.logHSVColorMap()
                cmap.set_bad(color=backgroundColor)
            return norm, cmap
        # Pass back colormap for linear case
        elif scale == 'linear':
            cmap = plt.cm.get_cmap(cmap)
            cmap.set_bad(color=backgroundColor)
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
        # select band by date
        date = self.parseDate(date)
        if date is not None:
            myVar = self.subset.sel(time=date,
                                    method='nearest').sel(band=band).data
        else:
            myVar = self.subset.sel(band=band).data
        # compute max
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
                  colorBarPad, labelFontSize, plotFontSize, extend='max',
                  fontScale=1):
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

        cb = plt.colorbar(pos, cax=cbAx, orientation=orientation,
                          extend=extend)
        cb.set_label(colorBarLabel, size=labelFontSize * fontScale)
        cb.ax.tick_params(labelsize=plotFontSize * fontScale)
        if colorBarPosition in ['right', 'left']:
            cbAx.yaxis.set_ticks_position(colorBarPosition)
            cbAx.yaxis.set_label_position(colorBarPosition)
        elif colorBarPosition in ['top', 'tottom']:
            cbAx.xaxis.set_ticks_position(colorBarPosition)
            cbAx.xaxis.set_label_position(colorBarPosition)

    def hsvSpeedRender(self, speed, vmin=1, vmax=3000):
        '''
        Convert speed to rgb version of hsv rendering of image.

        Parameters
        ----------
        speed : np array
            Speed to be rendered, which will be clipped to (vmin, vmax)
        vmin : float, optional
            Minimum speed to clip to. The default is 1.
        vmax : float, optional
            Maximum speed to clip to. The default is 3000.
        Returns
        -------
        rgb image.
        '''
        # print(speed.shape)
        background = np.isnan(speed)
        # type(background))
        # Uniform value
        value = np.full(speed.shape, 1)
        # Reduce saturation on low end
        saturation = np.clip((speed/125 + .5)/1.5, 0, 1)
        # Force background to white
        # print(speed.shape, saturation.shape)
        saturation[background] = 0
        hue = np.log10(np.clip(speed, vmin, vmax)) / \
            (np.log10(vmax) - np.log(vmin))
        # order axes so rgb bands indexed last for imshow
        hsv = np.moveaxis(np.array([hue, saturation, value]), 0, 2)
        return colors.hsv_to_rgb(hsv)

    def logHSVColorMap(self, vmin=1, vmax=3000, ncolors=1024):
        ''' Create a log color map for displaying velocity'''
        # value
        value = np.full((ncolors), 1)
        # Compute values of log scale
        dv = (np.log10(vmax) - np.log10(vmin))/ncolors
        vrange = np.power(10., np.arange(0, ncolors) * dv)
        # Provides nice map
        saturation = np.clip((vrange/125 + .5)/1.5, 0, 1)
        # Use linear scale - norm will be used for log scale
        hue = np.arange(0, 1, 1/ncolors)
        # hsv to rgb
        hsv = np.array([hue, saturation, value]).transpose()
        rgb = colors.hsv_to_rgb(hsv)
        # return color map
        return colors.LinearSegmentedColormap.from_list('my', rgb, N=ncolors)

    def _dateTitle(self, date, midDate):
        '''

        Parameters
        ----------
        midDate : Boolean
            Use central date.
        Returns
        -------
        date as string for title.

        '''
        middleDate, date1, date2 = self._dates(date, asString=True)
        if midDate:
            title = middleDate
        else:
            title = f'{date1} - {date2}'
        return title

    def _labelAxes(self, ax, xLabel, yLabel, title=None,
                   labelFontSize=15, titleFontSize=16, plotFontSize=13,
                   fontScale=1, axisOff=False):
        '''
        Label and format axes

        Parameters
        ----------
        ax : axis
            matplotlib axes. The default is None.
        xLabel : tr, optional
            x-axis label. The default is 'Distance', use '' to disable.
        yLabel : tr, optional
            x-axis label. The default is band appropriate (e.g, Speed),
            use '' to disable.
        title : str, optional
            Plot title. The default is None.
        units : str, optional
            Units (m or km) for the x, y coordinates. The default is 'm'
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
        if axisOff:
            ax.axis('off')
        else:
            ax.set_xlabel(xLabel, size=labelFontSize * fontScale)
            ax.set_ylabel(yLabel, size=labelFontSize * fontScale)
            ax.tick_params(axis='both', labelsize=plotFontSize * fontScale)
            ax.tick_params(axis='y', labelsize=plotFontSize * fontScale)
        # Create title from dates
        if title is not None:
            ax.set_title(title, fontsize=titleFontSize * fontScale)

    def displayVar(self, band, date=None, ax=None, title=None,
                   colorBar=True,
                   labelFontSize=15,
                   titleFontSize=16,
                   plotFontSize=13,
                   fontScale=1,
                   axisOff=False,
                   vmin=0,
                   vmax=7000,
                   units='m',
                   scale='linear',
                   cmap=None,
                   midDate=True,
                   colorBarLabel='Speed (m/yr)',
                   colorBarPosition='right',
                   colorBarSize='5%',
                   colorBarPad=0.05,
                   wrap=None,
                   masked=None,
                   extend=None,
                   backgroundColor=(1, 1, 1),
                   **kwargs):
        '''
        Use matplotlib to show velocity in a single subplot with a color
        bar. Clip to absolute max set by maxv, though in practives percentile
        will clip at a signficantly lower value.

        Parameters
        ----------
        band : str
            band name (e.g., sigma, vx, vy).
        date : 'YYYY-MM-DD' or datetime, optional
            The date in the series to plot. The default is the first date.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        ax : axis, optional
            matplotlib axes. The default is None.
        colorBar : TYPE, optional
            DESCRIPTION. The default is True.
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
        vmax : number, optional
            max velocity to display. The default is 7000.
        vmin : number, optional
            min velocity to display. The default is 0.
        units : str, optional
            units of coordiinates (m or km). The default is 'm'.
        scale : str, optional
            Scale type ('linear' or 'log') The default is 'linear'.
        cmap : colormap specific, optional
            Colormap. The default is None.
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
        masked : Boolean, optional
            Masked for imshow. The default is None.
        extend : str, optional
            Colorbar extend ('both','min', 'max', 'neither').
            The default is None.
        backgroundColor : color, optional
            Background color. The default is (1, 1, 1).
        **kwargs : dict
            Keywords to imshow.

        Returns
        -------
        pos : matplotlib.image.AxesImage
            return value from imshow.
        '''
        if not self._checkUnits(units):
            return
        if ax is None:
            sx, sy = self.sizeInPixels()
            fig, ax = plt.subplots(1, 1, constrained_layout=True,
                                   figsize=(10.*float(sx)/float(sy), 10.))
        norm, cmap = self.colorSetup(scale, cmap, vmin, vmax,
                                     backgroundColor=backgroundColor)
        # Return if invalid var
        if band not in self.variables:
            print(f'{band} is not a valid, the choices are {self.variables}')
            return
        # Extract data for band
        displayVar = self.subset.sel(band=band)
        # Extract date for time
        date = self.parseDate(date)
        displayVar = displayVar.sel(time=date, method='nearest')
        # Display the data
        displayVar = np.squeeze(displayVar)
        # Wrap data
        if wrap is not None:
            displayVar = np.mod(displayVar, wrap)
        # Set default extend based on scale type
        if extend is None:
            try:
                extend = {'log': 'both', 'linear': 'max'}[scale]
            except Exception:
                print('Could not set extend for colorbar using scale={scale}')
        #
        if cmap == 'log':
            displayVar = self.hsvSpeedRender(displayVar.data)
        pos = ax.imshow(np.ma.masked_where(masked, displayVar,
                                           copy=True), norm=norm, cmap=cmap,
                        extent=self.extent(units=units), **kwargs)
        if title is None:
            title = self._dateTitle(date, midDate)
        # labels and such
        self._labelAxes(ax, f'x ({units})',  f'y ({units})',
                        labelFontSize=labelFontSize,
                        titleFontSize=titleFontSize, plotFontSize=plotFontSize,
                        axisOff=axisOff, title=title, fontScale=fontScale)
        if colorBar:
            self._colorBar(pos, ax, colorBarLabel, colorBarPosition,
                           colorBarSize, colorBarPad, labelFontSize,
                           plotFontSize, extend=extend, fontScale=fontScale)

        return pos

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

    def writeCloudOptGeo(self, tiffRoot, full=False, myVars=None, myXR=None,
                         **kwargs):
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
            myVars = self.variables
        else:
            if type(myVars) is str:  # Ensure its a str
                myVars = [myVars]
        # By default write subset unless none exists or full flagged
        if myXR is None:
            if full or self.subset is None:
                myXR = self.xr
            else:
                myXR = self.subset
        # Loop to write bands
        for band in myVars:
            # map np.nan to nodata value
            myBand = myXR.sel(
                    band=band).rio.write_nodata(self.noDataDict[band],
                                                encoded=True)
            #
            myBand.rio.to_raster(self.tiffFileName(tiffRoot, band), **kwargs)

    def writeSeriesToCOG(self, baseName, dateFormat='%d%b%y', suffix='V'):
        '''
        Write series as a sequence of geotiffs with format with
        baseName_date1_date2_band_suffix_band.tif

        Parameters
        ----------
        baseName : str
            First part of file name.
        dateFormat : str, optional
            Format for date strings in name. The default is '%b%d%Y'.
        Returns
        -------
        None.

        '''
        for time, date1, date2 in zip(self.time, self.time1, self.time2):
            if isinstance(date1, datetime):
                date1 = date1.strftime(dateFormat)
                if isinstance(date2, datetime):
                    date2 = date2.strftime(dateFormat)
        #
            tiffRoot = f'{baseName}_{date1}_{date2}_*_{suffix}'
            print(tiffRoot)
            subset = self.subset.sel(time=time)
            self.writeCloudOptGeo(tiffRoot, myXR=subset, myVars=self.variables)

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
        for x in self.subset.coords:
            if 'proj' in x or 'raster' in x:
                self.subset = self.subset.drop(x, dim=None)
        #
        self.subset.to_netcdf(path=cdfFile, mode='w')
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

    #
    # ---- Plot routines
    #

    def _plotPoint(self, x, y, band, *argv, ax=None, units='m',
                   sourceEPSG=None, **kwargs):
        '''
        Interpolate data set at point x, y, and plot result vs time

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        band : str
            band name (e.g., sigma, vx, vy).
        *argv : list
            Additional args to pass to plt.plot (e.g. 'r*').
        ax : axis, optional
            matplotlib axes. The default is None.
        units : str, optional
            units of coordiinates (m or km). The default is 'm'.
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax.

        '''
        self._checkUnits(units)
        # Create axes
        if ax is None:
            fig, ax = plt.subplots(1, 1,
                                   constrained_layout=True, figsize=(10, 8))
        if band is None:
            print('No band specified')
            return
        # Interpolate
        result = self.interp(x, y, returnXR=True, sourceEPSG=sourceEPSG,
                             units=units).sel(band=band)
        # Plot result
        ax.plot(result.time, np.squeeze(result), *argv, **kwargs)
        return ax

    def _plotProfile(self, x, y, band, date, *argv, label=None, ax=None,
                     sourceEPSG=None, distance=None, units='m', **kwargs):
        '''
        Interpolate data for profile x, y and plot as a function of distance

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        band : str
            band name (e.g., sigma, vx, vy).
        date : 'YYYY-MM-DD' or datetime, optional
            The date in the series to plot. The default is the first date.
        *argv : list
            Additional args to pass to plt.plot (e.g. 'r*').
        label : str, optional
            label for plot (same as plt.plot label). The default is YYYY-MM=DD.
        ax : axis, optional
            matplotlib axes. The default is None.
        distance : nparray, optional
            distance variable for plot.
            The default is None, which causes it to be calculated.
        units : str, optional
            units of coordiinates (m or km). The default is 'm'.
        **kwargs : dict
            kwargs pass through to plt.plot.

        Returns
        -------
        ax. Either the value passed on or the one created if none given.
        '''
        self._checkUnits(units)
        # Create axes
        if ax is None:
            fig, ax = plt.subplots(1, 1,
                                   constrained_layout=True, figsize=(10, 8))
        # Return if no band specified
        if band is None:
            print('No band specified')
            return
        #
        date = self.parseDate(date, returnString=True)
        if label is None:
            label = date.strftime('%Y-%m-%d')
        # Interpolate
        result = self.interp(x, y, returnXR=True,
                             units=units, sourceEPSG=sourceEPSG,
                             ).sel(band=band).sel(time=date, method='nearest')
        # Compute distance along xy profile
        if distance is None:
            distance = np.cumsum(
                np.concatenate(([0], np.sqrt(np.diff(result.x)**2 +
                                             np.diff(result.y)**2))))
        # Plot result
        ax.plot(distance, result, label=label, *argv, **kwargs)
        return ax

    def _removeNoData(self, t, v, band):
        ''' processs np array to remove no data and return as lists. If
        all no data return original'''
        if np.isnan(self.noDataDict[band]):
            keep = np.isfinite(v)
        else:
            keep = v > self.noDataDict[band]
        # Save valid values
        if len(keep) > 0:
            v = v[keep]
            t = t[keep]
        return list(t), list(v)

    def _extractData(self, x, y, band=None, plotOptions=None, units='m',
                     **kwargs):
        ''' Plot the time series, filtering out no data values '''
        dtf = DatetimeTickFormatter(years="%Y")
        # Get data
        result = self.interp(x, y, units=units, returnXR=True)
        vOrig = result.sel(band=band).values.flatten()
        tOrig = self.subset.time.values.flatten()
        t, v = self._removeNoData(tOrig, vOrig, band)
        # Plot points and lines- options need some work
        return hv.Curve((t, v)).opts(**plotOptions) * \
            hv.Scatter((t, v)).opts(color='red', size=4, framewise=True,
                                    xformatter=dtf, **plotOptions)

    def _view(self, band, imgOptions, plotOptions, ncols=2, date=None,
              markerColor='red', **kwargs):
        ''' Setup and return plot '''
        # Setup the image plot.
        if date is None:
            date = self.subset.time[-1]
        img = self.subset.sel(band=band).sel(time=date, method='nearest')
        imgDate = self.datetime64ToDatetime(
            np.datetime64(img.time.item(0), 'ns')).strftime('%Y-%m-%d')
        if 'title' not in imgOptions:
            imgOptions['title'] = f'{band} for {imgDate} '
        imgPlot = img.hvplot.image(rasterize=True, aspect='equal').opts(
            active_tools=['point_draw'], max_width=500, min_width=300,
            max_height=800, **imgOptions)
        # Setup up the time series plot
        xc = self.x0 + self.sx * self.dx * 0.5
        yc = self.y0 + self.sy * self.dy * 0.5
        points = hv.Points(([xc], [yc])).opts(size=6, color=markerColor)
        pointer = hv.streams.PointDraw(source=points,
                                       data=points.columns(), num_objects=1)
        # Create the dynamic map
        pointer_dmap = hv.DynamicMap(
            lambda data: self._extractData(data['x'][0], data['y'][0],
                                           band=band, plotOptions=plotOptions),
            streams=[pointer]).opts(width=500)
        # Return the result for display
        return pn.panel((imgPlot * points +
                         pointer_dmap).cols(ncols).opts(merge_tools=False))



