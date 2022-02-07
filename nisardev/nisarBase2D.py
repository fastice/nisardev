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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from dask.diagnostics import ProgressBar

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
        self.workingXR = None
        self.flipY = True
        self.useStackstac = False
        self.xforms = {}  # Cached pyproj transforms
        dask.config.set(num_workers=numWorkers)

    # ------------------------------------------------------------------------
    # Setup abtract methods for child classes
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # def setup xy coordinates
    # ------------------------------------------------------------------------

    def xyGrid(self):
        '''
        Computer grid version of coordinates.
        Save as 2-D arrays: self.xGrid, self.yGrid
        Returns
        -------
        None.
        '''
        #
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

    # -----------------------------------------------------------------------
    # Interpolate geo image.
    # -----------------------------------------------------------------------
    def _convertCoordinates(func):
        ''' This will wrap interpolators so that the input coordinates (e.g.
        lat lon) will automatically be mapped to xy as specificed by
        sourceEPSG=epsg.'''
        @functools.wraps(func)
        def convertCoordsInner(inst, xin, yin, *args, date=None, units='m',
                               **kwargs):
            # No source EPSG so just pass original coords back
            if 'sourceEPSG' not in kwargs:
                xout, yout = xin, yin
            else:
                # Ensure epsg a string
                sourceEPSG = str(kwargs['sourceEPSG'])
                # See if xform already cached
                if sourceEPSG not in inst.xforms:
                    inst.xforms[sourceEPSG] = \
                        pyproj.Transformer.from_crs(f"EPSG:{sourceEPSG}",
                                                    f"EPSG:{inst.epsg}")
                # Transform coordates
                xout, yout = inst.xforms[sourceEPSG].transform(xin, yin)
            # Unit conversion if needed
            if units == 'km':
                xout, yout = xout * 1000, yout * 1000
            return func(inst, xout, yout, *args, date=date, **kwargs)
        return convertCoordsInner

    def _checkUnits(self, units):
        '''
        Check units return True for valid units. Print message for invalid.
        '''
        if units not in ['m', 'km']:
            print('Invalid units: must be m or km')
            return False
        return True

    def interpGeo(self, x, y, myVars, date=None, units='m', **kwargs):
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
        return self._interpNP(x, y, myVars, date=date, units=units, **kwargs)

    @_convertCoordinates
    def _interpNP(self, x, y, myVars, date=None, units='m', **kwargs):
        '''
        Interpolate myVar(y,x) specified myVars (e.g., ['vx'..]) where myVars
        are nparrays.
        Parameters
        ----------
        x: np float array
            x coordinates in m to interpolate to.
        y: np float array
            y coordinates in m to interpolate to.
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        **kwargs : TBD
            Keyword pass through to interpolator.
        Returns
        -------
        myResults : float np array
            Interpolated valutes from x,y locations.
        '''
        if np.isscalar(x):
            x, y = [x], [y]
        xx1, yy1 = xr.DataArray(x), xr.DataArray(y)
        myXR = self.workingXR
        #
        nTime = myXR.shape[0]
        nBands = len(myVars)
        # Array to receive results
        date = self.parseDate(date)
        # No date or single time layer, so no time dimension
        if nTime <= 1 or date is not None:
            myResults = np.zeros((nBands, *xx1.shape))
        else:  # Include time dimension
            myResults = np.zeros((nBands, nTime, *xx1.shape))
        # Interp by band
        for myVar, i in zip(myVars, range(0, len(myVars))):
            if date is None:
                myResults[i][:] = myXR.sel(band=myVar).interp(
                    x=xx1, y=yy1).data
            else:
                myResults[i][:] = myXR.sel(
                    band=myVar).sel(time=date, method='nearest').interp(
                    x=xx1, y=yy1).data
        return myResults

    def readXR(self, fileNameBase, url=False, masked=False, stackVar=None,
               time=None, time1=None, time2=None, skip=[]):
        ''' Use read data into rioxarray variable
        Parameters
        ----------
        fileNameBase = str
            template with filename firstpart_*_secondpart (no .tif)
            the * will be replaced with the different compnents (e.g., vx,vy)
        url bool, optional
            Set true if fileNameBase is a link
        '''
        # Do a lazy open on the tiffs
        if stackVar is None:
            self.xr = self.lazy_openTiff(fileNameBase, url=url, masked=masked,
                                         time=time, skip=skip)
            self.workingXR = self.xr
            self.parseGeoInfo()
        else:
            items = self._construct_stac_items(fileNameBase, stackVar,
                                               skip=skip, url=url)
            self.xr = self.lazy_open_stack(items, stackVar)
            self.workingXR = self.xr
            self.parseGeoInfoStack()
        if time1 is not None:
            self.xr['time1'] = time1
        if time2 is not None:
            self.xr['time2'] = time2
        #
        self.xr.rename('None')
        # Save variables
        self._mapVariables()

    def readFromNetCDF(self, cdfFile):
        ''' Load data from netcdf file '''
        if '.nc' not in cdfFile:
            cdfFile = f'{cdfFile}.nc'
        xDS = xr.open_dataset(cdfFile)
        # Pull the first variable that is not spatial_ref
        for var in list(xDS.data_vars.keys()):
            if var != 'spatial_ref':
                self.xr = xr.DataArray(xDS[var])
                break
        try:
            self.xr['spatial_ref'] = xDS['spatial_ref']
        except Exception:
            print('warning missing spatial_ref')
        #
        self.workingXR = self.xr
        self.parseGeoInfo()
        self._mapVariables()
        
    def loadRemote(self):
        ''' Load the current XR, either full array if not subset,
        or the subsetted version '''
        with ProgressBar():
            self.workingXR.load()

    def _construct_stac_items(self, fileNameBase, stackVar, url=True, skip=[]):
        ''' construct STAC-style dictionaries of CMR urls for stackstac '''
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

    def lazy_open_stack(self, items, stackVar):
        ''' return stackstac xarray dataarray '''
        dtype = 'float32'
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
                             dtype=dtype,
                             xy_coords='center',
                             chunksize=CHUNKSIZE,
                             bounds=stackVar['bounds']
                             )
        da = da.rio.write_crs(f'epsg:{stackVar["epsg"]}')
        return da

    # @dask.delayed
    def lazy_openTiff(self, fileNameBase, masked=False, url=False, time=None,
                      skip=[], **kwargs):
        ''' Lazy open of a single velocity product
        Parameters
        ----------
        fileNameBase, str
            template with filename firstpart_*_secondpart (no .tif)
        '''
        # print(href)d
        das = []
        option = '?list_dir=no'
        for band in self.variables:
            if band in skip:
                continue
            bandTiff = fileNameBase
            if url:
                bandTiff = f'/vsicurl/{option}&url={fileNameBase}'
            bandTiff = bandTiff.replace('*', band) + '.tif'
            # create rioxarry
            chunks = {'band': 1, 'y': CHUNKSIZE, 'x': CHUNKSIZE}
            da = rioxarray.open_rasterio(bandTiff, lock=True,
                                         default_name=fileNameBase,
                                         chunks=chunks, masked=masked)
            if time is not None:
                da = da.expand_dims(dim='time')
                da['time'] = [time]
            da['band'] = [band]
            da['name'] = 'myData'
            da['_FillValue'] = self.noDataDict[band]
            das.append(da)
        # Concatenate bands (components)
        return xr.concat(das, dim='band', join='override',
                         combine_attrs='drop')

    def _mapVariables(self):
        ''' Map the xr variables to band variables (e.g., vx, vy) '''
        for myVar in self.workingXR.band.data:
            myVar = str(myVar)
            bandData = np.squeeze(self.workingXR.sel(band=myVar).data)
            setattr(self, myVar, bandData)

    def subSetData(self, bbox):
        ''' Subset dataArray using a box to crop limits
        Parameters
        ----------
        bbox, dict
            crop area {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        '''
        self.subset = self.xr.rio.clip_box(**bbox)
        self.workingXR = self.subset
        # Save variables
        self._mapVariables()
        # update the geo info (origin, epsg, size, res)
        # if self.useStackstac:
        #    self.parseGeoInfoStack()
        # else:
        self.parseGeoInfo()

    def parseDate(self, date):
        ''' Accept date as either datetime or YYYY-MM-DD
        Parameters
        ----------
        date, str or datetime
        Returns
        -------
        date in as datetime instance
        '''
        if date is None:
            return date
        try:
            if type(date) == str:
                date = datetime.strptime(date, '%Y-%m-%d')
        except Exception:
            print('Error: Either invalid date (datetime or "YYYY-MM-DD")')
            print(f'Or date outside range: {min(self.workingXR.date)}, '
                  f'{max(self.workingXR.date)}')
        return date

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

    def parseGeoInfoStack(self):
        ''' parse geoinfo for stackstac '''
        myXr = self.workingXR
        # get epsg
        self.epsg = int(myXr.crs.split(":")[1])
        gT = myXr.transform
        self.dx, self.dy = abs(gT[0]), abs(gT[4])
        #
        self._computeCoords(myXr)

    def parseGeoInfo(self):
        '''Parse geo info out of xr spatial ref'''
        myXr = self.workingXR
        # Get wkt and convert to crs
        self.wkt = myXr.spatial_ref.attrs['spatial_ref']
        myCRS = pyproj.CRS.from_wkt(self.wkt)
        # Save epsg
        self.epsg = myCRS.to_epsg()
        # get lower left corner, sx, sy, dx, dy
        gT = list(myXr.rio.transform())
        self.dx, self.dy = abs(gT[0]), abs(gT[4])
        self._computeCoords(myXr)

    # ----------------------------------------------------------------------
    # Output a band as geotif
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Plotting and display
    # ----------------------------------------------------------------------

    def _dates(self, date, asString=False):
        ''' Extract dates for a layer based on current date '''
        date1 = self.datetime64ToDatetime(
            self.workingXR.time.sel(time=date, method='nearest').time1.data)
        date2 = self.datetime64ToDatetime(
            self.workingXR.time.sel(time=date, method='nearest').time2.data)
        midDate = self.datetime64ToDatetime(
            self.workingXR.time.sel(time=date, method='nearest').data)
        if asString:
            date1 = date1.strftime('%Y-%m-%d')
            date2 = date2.strftime('%Y-%m-%d')
            midDate = midDate.strftime('%Y-%m-%d')
        return midDate, date1, date2

    def displayVar(self, var, date=None, ax=None, plotFontSize=14,
                   labelFontSize=12, titleFontSize=15, axisOff=False,
                   vmin=0, vmax=7000, units='m',
                   title=None, midDate=True, colorBarLabel='Speed (m/yr)',
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
                                   figsize=(10.*sx/sy, 10.))
        #
        divider = make_axes_locatable(ax)  # Create space for colorbar
        cbAx = divider.append_axes('right', size='5%', pad=0.05)
        # Return if invalid var
        if var not in self.variables:
            print(f'{var} is not a valid, the choices are {self.variables}')
            return
        # Plot map
        displayVar = self.workingXR.sel(band=var)
        #
        if date is not None:
            if type(date) == str:
                date = datetime.strptime(date, '%Y-%m-%d')
            displayVar = displayVar.sel(time=date, method='nearest')
        else:  # Single band case, so pull its date
            date = self.datetime64ToDatetime(displayVar.time[0].data)
        # Create title from dates
        if title is None:
            middleDate, date1, date2 = self._dates(date, asString=True)
            if midDate:
                title = middleDate
            else:
                title = f'{date1} - {date2}'
        displayVar = np.squeeze(displayVar)
        pos = ax.imshow(displayVar, vmin=vmin, vmax=vmax,
                        extent=self.extent(units=units), **kwargs)
        if axisOff:
            ax.axis('off')
        # labels and such.
        cb = plt.colorbar(pos, cax=cbAx, orientation='vertical', extend='max')
        cb.set_label(colorBarLabel, size=labelFontSize)
        cb.ax.tick_params(labelsize=plotFontSize)
        ax.set_xlabel(f'X ({units})', size=labelFontSize)
        ax.set_ylabel(f'Y ({units})', size=labelFontSize)
        ax.tick_params(axis='x', labelsize=plotFontSize)
        ax.tick_params(axis='y', labelsize=plotFontSize)
        ax.set_title(title, fontsize=titleFontSize)
        return ax

    # ----------------------------------------------------------------------
    # Return  geometry params in m and km
    # ----------------------------------------------------------------------

    def sizeInPixels(self):
        '''
        Return size in pixels
        Returns
        -------
        sx: int
            x size
        sy: int
            y size.
        '''
        return self.sx, self.sy

    def _toKM(self, x, y):
        return x/1000., y/1000.

    def size(self, units='m'):
        ''' Return size in meters '''
        self._checkUnits(units)
        if units == 'km':
            return self._toKM(self.sx*self.dx, self.sy*self.dy)
        return self.sx*self.dx, self.sy*self.dy

    def origin(self, units='m'):
        ''' Return origin in meters '''
        self._checkUnits(units)
        if units == 'km':
            return self._toKM(self.x0, self.y0)
        return self.x0, self.y0

    def bounds(self, units='m'):
        '''
        Determine data bounds in meters
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
        self.workingXR.to_netcdf(path=cdfFile)
        return cdfFile
