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
from scipy.interpolate import RegularGridInterpolator
import pyproj
import dask
import rioxarray


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
        self.xGrid, self.yGrid = [], []  # Grid of coordinate for each pixel
        self.verbose = verbose  # Print progress messages
        self.epsg = epsg  # EPSG (sussm 3031 and 3413; should work for others)
        self.xr = None
        self.subset = None
        self.flipY = True
        self.url = False
        dask.config.set(num_workers=numWorkers)

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
    def readDataFromURL():
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
        # setup interpolation, Note y,x indexes for row colum
        for myVar in myVars:
            myV = getattr(self, myVar).compute()
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

    def readXR(self, fileNameBase, url=False):
        ''' Use read data into rioxarray variable
        Parameters
        ----------
        fileNameBase = str
            template with filename firstpart_*_secondpart (no .tif)
            the * will be replaced with the different compnents (e.g., vx,vy)
        url bool, optional
            Set true if fileNameBase is a link
        '''
        self.url = url
        # Do a lazy open on the tiffs
        self.xr = dask.compute(self.lazy_openTiff(fileNameBase, url=url))[0]
        # get the geo info (origin, epsg, size, res)
        self.parseGeoInfo(subset=False)
        # Save variables
        self._mapVariables()

    def _mapVariables(self, subset=False):
        ''' Map the xr variables to band variables (e.g., vx, vy) '''
        if subset:
            myData = self.subset.data
        else:
            myData = self.xr.data
        for myVar, bandData in zip(self.variables, myData):
            if self.flipY:
                setattr(self, myVar, np.flipud(bandData))
            else:
                setattr(self, myVar, bandData)

    @dask.delayed
    def lazy_openTiff(self, fileNameBase, masked=False, url=False, **kwargs):
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
            bandTiff = fileNameBase
            if url:
                bandTiff = f'/vsicurl/{option}&url={fileNameBase}'
            bandTiff = bandTiff.replace('*', band) + '.tif'
            # create rioxarry
            chunks = {'band': 1, 'y': CHUNKSIZE, 'x': CHUNKSIZE}
            da = rioxarray.open_rasterio(bandTiff, lock=True,
                                         default_name=fileNameBase,
                                         chunks=chunks, masked=masked)
            da['band'] = [band]
            da['name'] = 'myData'
            da['_FillValue'] = self.noDataDict[band]
            das.append(da)
        # Concatenate bands (components)
        return xr.concat(das, dim='band', join='override',
                         combine_attrs='drop')

    def subSetData(self, bbox):
        ''' Subset dataArray using a box to crop limits
        Parameters
        ----------
        bbox, dict
            crop area {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
        '''
        self.subset = self.xr.rio.clip_box(**bbox)
        # Save variables
        self._mapVariables(subset=True)
        # update the geo info (origin, epsg, size, res)
        self.parseGeoInfo(subset=True)

    # ----------------------------------------------------------------------
    # Read geometry info
    # ----------------------------------------------------------------------

    def parseGeoInfo(self, subset=False):
        '''Parse geo info out of xr spatial ref'''
        if subset:
            myXr = self.subset
        else:
            myXr = self.xr
        if myXr is None:
            print('parseGeoInfo: xr not read yet')
            return
        # Get wkt and convert to crs
        self.wkt = myXr.spatial_ref.attrs['spatial_ref']
        myCRS = pyproj.CRS.from_wkt(self.wkt)
        # Save epsg
        self.epsg = myCRS.to_epsg()
        # get lower left corner, sx, sy, dx, dy
        gT = [float(x) for x in myXr.spatial_ref.attrs['GeoTransform'].split()]
        self.dx, self.dy = abs(gT[1]), abs(gT[5])
        if gT[5] > 0:
            self.flipY = False
        self.x0 = np.min(myXr.x).item()
        self.y0 = np.min(myXr.y).item()
        self.sx, self.sy = len(myXr.x), len(myXr.y)
        # coordinates
        self.xx = myXr.coords['x'].values
        if self.flipY:
            self.yy = np.flip(myXr.coords['y'].values)
        else:
            self.yy = myXr.coords['y'].values
        self.xyGrid()

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
