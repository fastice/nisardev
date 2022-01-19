#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:37:54 2020

@author: ian
"""
from abc import ABCMeta, abstractmethod
from nisarfunc import myError
import numpy as np
import functools
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from osgeo import gdal,  osr
import pyproj
import os


class nisarBase2D():
    ''' Abstract class to define xy polar stereo (3031, 3413) image objects
    such as nisarVel.

    Contains basic functionality to track xy coordinates and interpolate
    images give by myVariables. For example nisarVel as vx,vy,ex,ey, To
    interpolate just vx and vy pass myVars=['vx','vy'] to child. '''

    __metaclass__ = ABCMeta

    def __init__(self,  sx=None, sy=None, x0=None, y0=None, dx=None, dy=None,
                 verbose=True, epsg=None, useXR=False):
        ''' initialize a nisar velocity object'''
        self.sx, self.sy = sx, sy  # Image size in pixels
        self.x0, self.y0 = x0, y0  # Origin (center of lower left pixel) in m
        self.dx, self.dy = dx, dy  # Pixel size
        self.xx, self.yy = [], []  # Coordinate of image axis ( F(y,x))
        self.xGrid, self.yGrid = [], []  # Grid of coordinate for each pixel
        self.verbose = verbose  # Print progress messages
        self.epsg = epsg  # EPSG (sussm 3031 and 3413; should work for others)
        self.useXR = useXR  # Use XR arrays rather than numpy (not well tested)

    # ------------------------------------------------------------------------
    # Setup abtract methods for child classes
    # ------------------------------------------------------------------------
    @abstractmethod
    def _setupInterp():
        ''' Abstract setup interpolation method - define in child class. '''
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

    # ------------------------------------------------------------------------
    # def setup xy coordinates
    # ------------------------------------------------------------------------

    def xyCoordinates(self):
        '''
        Compute xy coordinates for image(y,x).
        Save as 1-D arrays: self.xx, self.yy.
        Returns
        -------
        None.
        '''
        # check specs exist
        if None in (self.x0, self.y0, self.sx, self.sy, self.dx, self.dy):
            myError('nisarVel.xyCoordinates: x0,y0,sx,sy,dx,dy undefined '
                    f'{self.x0},{self.y0},{self.sx},{self.sy},'
                    f'{self.dx},{self.dy}')
        # x0...x0+(sx-1)*dx, y0...
        self.xx = np.arange(self.x0, self.x0 + int(self.sx) * self.dx, self.dx)
        self.yy = np.arange(self.y0, self.y0 + int(self.sy) * self.dy, self.dy)

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
        if len(self.xx) == 0:
            self.xyCoordinates()
        # setup arrays
        self.xGrid = np.zeros((self.sy, self.sx))
        self.yGrid = np.zeros((self.sy, self.sx))
        # populate arrays
        for i in range(0, self.sy):
            self.xGrid[i, :] = self.xx
        for i in range(0, self.sx):
            self.yGrid[:, i] = self.yy

    # ------------------------------------------------------------------------
    # Projection routines
    # ------------------------------------------------------------------------

    def computePixEdgeCornersXYM(self):
        '''
        Return dictionary with corner locations. Note unlike pixel
        centered xx, x0 etc values, these corners are defined as the outer
        edges as in a geotiff
        Returns
        -------
        corners : dict
            corners in xy coordinates: {'ll': {'x': xll, 'y': yll}...}.
        '''
        nx, ny = self.sizeInPixels()
        x0, y0 = self.originInM()
        dx, dy = self.pixSizeInM()
        # Make sure geometry defined
        if None in [nx, ny, x0, y0, dx, dy]:
            myError(f'Geometry param not defined size {nx,ny}'
                    ', origin {x0,y0}), or pix size {dx,dy}')
        xll, yll = x0 - dx/2, y0 - dx/2
        xur, yur = xll + nx * dx, yll + ny * dy
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
        # xyproj = pyproj.Proj(f"EPSG:{self.epsg}")
        # llproj = pyproj.Proj("EPSG:4326")
        xytollXform = pyproj.Transformer.from_crs(f"EPSG:{self.epsg}",
                                                  "EPSG:4326")
        llcorners = {}
        # Loop to do coordinate transforms.
        for myKey in corners.keys():
            lat, lon = xytollXform.transform(np.array([corners[myKey]['x']]),
                                             np.array([corners[myKey]['y']]))
            llcorners[myKey] = {'lat': lat[0], 'lon': lon[0]}
        return llcorners

    def getWKT_PROJ(self, epsgCode):
        '''
        Get the wkt for the image
        Parameters
        ----------
        epsgCode : int
            epsg code.
        Returns
        -------
        wkt : str
            projection info as wkt string:
                'PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",...'
        '''
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsgCode)
        wkt = sr.ExportToWkt()
        return wkt

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
    # setup interpolation functions
    # -----------------------------------------------------------------------

    def _setupInterpolator(self, myVars):
        '''
        Set up interpolation for each variable specified by myVars.
        Parameters
        ----------
        myVars : list of str
            list of variable names as strings, e.g. ['vx',...].
        Returns
        -------
        None.

        '''
        if len(self.xx) <= 0:
            self.xyCoordinates()  # Setup coordinates if not done already.
        # XR, so nothing to do.
        if self.useXR:
            return
        # setup interpolation, Note y,x indexes for row colum
        for myVar in myVars:
            myV = getattr(self, myVar)  # Get variable. (e.g. vx,vy)
            setattr(self, f'{myVar}Interp',
                    RegularGridInterpolator((self.yy, self.xx), myV,
                                            method="linear"))

    # -----------------------------------------------------------------------
    # Interpolate geo image.
    # -----------------------------------------------------------------------

    def interpGeo(self, x, y, myVars, **kwargs):
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
        if self.useXR:
            return self._interpXR(x, y, myVars, **kwargs)
        else:
            return self._interpNP(x, y, myVars, **kwargs)

    def _toInterp(self, x, y):
        '''
        Return xy values of coordinates that within the image bounds for
        interpolation.
        Parameters
        ----------
        x : np float array
            x coordinates in m to interpolate to.
        y : np float array
            y coordinates in m to interpolate to.
        Returns
        -------
        x1 : np float array
            x coordinates for interpolation within image bounds.
        y1 : np float array
            y coordinates for interpolation within image bounds.
        igood : TYPE
            DESCRIPTION.
        '''
        # flatten, do bounds check, get locations of good (inbound) points.
        x1, y1 = x.flatten(), y.flatten()
        xgood = np.logical_and(x1 >= self.xx[0], x1 <= self.xx[-1])
        ygood = np.logical_and(y1 >= self.yy[0], y1 <= self.yy[-1])
        igood = np.logical_and(xgood, ygood)
        return x1[igood], y1[igood], igood

    def _interpXR(self, x, y, myVars, **kwargs):
        '''
        Interpolate myVar(y,x) specified myVars (e.g., ['vx'..]) where myVars
        are xarrays. This routine has had little testing.
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
        x1, y1, igood = self._toInterp(x, y)
        x1xr = xr.DataArray(x1)
        y1xr = xr.DataArray(y1)
        #
        myResults = [np.full(x1.transpose().shape, np.NaN) for x in myVars]
        for myVar, i in zip(myVars, range(0, len(myVars))):
            tmp = getattr(self,
                          f'{myVar}').interp(x=x1xr, y=y1xr, method='linear')
            myResults[i][igood] = tmp.values.flatten()
            myResults[i] = np.reshape(myResults[i], x.shape)
        #
        return myResults

    def _interpNP(self, x, y, myVars, **kwargs):
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
        x1, y1, igood = self._toInterp(x, y)
        # Save good points
        xy = np.array([y1, x1]).transpose()  # noqa
        #
        myResults = [np.full(x.shape, np.NaN).flatten() for v in myVars]
        for myVar, i in zip(myVars, range(0, len(myVars))):
            myResults[i][igood] = getattr(self, f'{myVar}Interp')(xy)
            myResults[i] = np.reshape(myResults[i], x.shape)
        #
        return myResults

    # ----------------------------------------------------------------------
    # Read geometry info
    # ----------------------------------------------------------------------

    def readGeodatFromTiff(self, tiffFile):
        '''
        Read geo information (x0,y0,sx,sy,dx,dy) from a tiff file.
        Assumes PS coordinates.
        Parameters
        ----------
        tiffFile : str
            Name of tiff file.
        Returns
        -------
        None.
        '''
        if not os.path.exists(tiffFile):
            myError(f"readGeodatFromTiff: {tiffFile} does not exist")
        try:
            gdal.AllRegister()
            ds = gdal.Open(tiffFile)
            proj = osr.SpatialReference(wkt=ds.GetProjection())
            self.epsg = int(proj.GetAttrValue('AUTHORITY', 1))
            self.sx, self.sy = ds.RasterXSize,  ds.RasterYSize
            gt = ds.GetGeoTransform()
            self.dx, self.dy = abs(gt[1]), abs(gt[5])
            # lower left corner corrected to pixel centered values
            self.x0 = (gt[0] + self.dx/2)
            # y-coord check direction
            if gt[5] < 0:
                self.y0 = (gt[3] - self.sy * self.dy + self.dy/2)
            else:
                self.y0 = (gt[3] + self.dy/2)
            self.xyCoordinates()
        except Exception:
            myError(f"readGeodatFromTiff: {tiffFile} exists but cannot parse")

    # ----------------------------------------------------------------------
    # Output a band as geotif
    # ----------------------------------------------------------------------

    def writeCloudOptGeo(self, tiffFile, myVar, gdalType=gdal.GDT_Float32,
                         overviews=None, predictor=1, noData=None):
        '''
        Write a cloud optimized geotiff with overviews if requested.
        Parameters
        ----------
        tiffFile : str
            Name of tiff file.
        myVar : str
            Name of internal variable (e.g., ".vx").
        gdalType : gdal type code, optional
            Gdal type code. The default is gdal.GDT_Float32.
        overviews : list, optional
            List of overvew sizes (e.g., [2, 4, 8, 16]). The default is None.
        predictor : int, optional
            Predictor used by gdal for compression. The default is 1.
        noData : optional - same as data, optional
            No data value. The default is None.
        Returns
        -------
        None.
        '''
        # use a temp mem driver for CO geo
        driverM = gdal.GetDriverByName("MEM")
        nx, ny = self.sizeInPixels()
        dx, dy = self.pixSizeInM()
        dst_ds = driverM.Create('', nx, ny, 1, gdalType)
        # set geometry
        tiffCorners = self.computePixEdgeCornersXYM()
        dst_ds.SetGeoTransform((tiffCorners['ul']['x'], dx, 0,
                                tiffCorners['ul']['y'], 0, -dy))
        # Set projection
        wkt = self.getWKT_PROJ(self.epsg)
        dst_ds.SetProjection(wkt)
        #  Handle no data value
        if noData is not None:
            getattr(self, myVar)[np.isnan(getattr(self, myVar))] = noData
            dst_ds.GetRasterBand(1).SetNoDataValue(noData)
        # write data
        dst_ds.GetRasterBand(1).WriteArray(np.flipud(getattr(self, myVar)))
        # Process overviews
        overFlag = 'No'
        if overviews is not None:
            overFlag = 'YES'
            dst_ds.BuildOverviews('AVERAGE', overviews)
        # now copy to a geotiff
        # mem -> geotiff forces correct order for c opt geotiff
        dst_ds.FlushCache()
        #
        # setup file driver and copy mem to it.
        driverT = gdal.GetDriverByName("GTiff")
        dst_ds2 = driverT.CreateCopy(tiffFile, dst_ds,
                                     options=[f'COPY_SRC_OVERVIEWS={overFlag}',
                                              'COMPRESS=LZW',
                                              f'PREDICTOR={predictor}',
                                              'TILED=YES'])
        # Flush cache and free memory to finish
        dst_ds2.FlushCache()
        dst_ds, dst_ds2 = None, None

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

    def _toKm(func):
        '''
        Decorator for unit conversion
        Parameters
        ----------
        func : function
            DESCRIPTION.
        Returns
        -------
        float
            Coordinates converted to km.
        '''
        @functools.wraps(func)
        def convertKM(*args):
            return [x*0.001 for x in func(*args)]
        return convertKM

    def sizeInM(self):
        ''' Return size in meters '''
        return self.sx*self.dx, self.sy*self.dy

    @_toKm
    def sizeInKm(self):
        '''
        Using decorator _toKm convert size to km
        Returns
        -------
        sx, sy: float
            size in km.
        '''
        return self.sizeInM()

    def originInM(self):
        ''' Return origin in meters '''
        return self.x0, self.y0

    @_toKm
    def originInKm(self):
        '''
        Using decorator _toKm convert origin to km
        Returns
        -------
        x0, y0: float
            origin in km.
        '''
        return self.originInM()

    def boundsInM(self):
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
        xmax = (self.x0 + (self.sx - 1) * self.dx)
        ymax = (self.y0 + (self.sy-1)*self.dy)
        return self.x0, self.y0, xmax, ymax

    def extentInM(self):
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
        bounds = self.boundsInM()
        return bounds[0], bounds[2], bounds[1], bounds[3]

    @_toKm
    def extentInKm(self):
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
        bounds = self.boundsInM()
        return bounds[0], bounds[2], bounds[1], bounds[3]

    @_toKm
    def boundsInKm(self):
        '''
        Determine data bounds in km using _toKM decorator.
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
        return self.boundsInM()

    def pixSizeInM(self):
        '''
        Return pixel size in m
        Returns
        -------
        dx : float
            pixel size in x dimension.
        dy : float
            pixel size in y dimension.
        '''
        return self.dx, self.dy

    @_toKm
    def pixSizeInKm(self):
        '''
        Return pixel size in km
        Returns
        -------
        dx : float
            pixel size in x dimension.
        dy : float
            pixel size in y dimension.
        '''
        return self.pixSizeInM()
