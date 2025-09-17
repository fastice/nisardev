# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:35:53 2019

@author: ian
"""

import os
import numpy as np
import pyproj
import functools
import matplotlib.pylab as plt
from nisardev import myError
import pandas as pd
import dask
from datetime import datetime
import pandas as pd
import matplotlib.colors as mcolors


class cvPoints:
    ''' cvpoints object - Has functionality for:
        input/output of cal/val points,
        filtering of points based on speed,
        computing the differences with a velocity map
        computing the statistics of these differences, and
        plotting the differencs and loctions'''

    labelFontSize = 14  # Font size for plot labels
    plotFontSize = 12  # Font size for plots
    legendFontSize = 12  # Font size for legends

    def __init__(self, cvFile=None, epsg=None, wktFile=None):
        '''
        Initialize an object to read a cvfile (cal/val points file) and
        manipulate cv points.
        Parameters
        ----------
        cvFile : TYPE, optional
            DESCRIPTION. The default is None.
        epsg : TYPE, optional
            DESCRIPTION. The default is None.
        wktFile : TYPE, optional
            DESCRIPTION. Wktfile instead of epsg None.
        Returns
        -------
        None.
        '''
        #
        # set everything to zero as default
        self.cvFile = None  # File with cal/val points
        self.lat, self.lon = np.array([]), np.array([])  # lat/lon of points
        self.x, self.y = np.array([]), np.array([])  # x/y PS of points in m
        self.z = np.array([])  # Elevatin of points
        self.nocull = None  # Used to mark points to cull (by default all 1)
        self.pound2 = False  # Flag to write pound 2 to output (historical)
        self.header = []  # In dictionary used as header - needs some work
        self.vx, self.vy, self.vz = np.array([]), np.array([]), np.array([])
        self.weight = np.array([])
        #
        self.setCVFile(cvFile)
        if wktFile is not None and epsg is not None:
            myError('Specify epsg or wktfile but not both')
        self.epsg = epsg  # Epsg set 3413 or 3031 depending on if N or S
        self.wktFile = wktFile
        self.llproj = "EPSG:4326"  # Used for conversions
        self.xyproj = None  # Defined once epsg set.
        #
        self.static = True
        if cvFile is not None:
            self.readCVs(cvFile=cvFile)

    def setCVFile(self, cvFile):
        '''
        Set the name of the CVlfile for the object.
        Parameters
        ----------
        cvFile : str
            Name of cv file with cal/val points.
        Returns
        -------
        None.
        '''
        if type(cvFile) is list:
            self.static = False
        self.cvFile = cvFile

    def checkCVFile(self):
        '''
        Check cvfile exists.
        Parameters
        ----------
        cvFile : str
            Name of cv file with cal/val points.
        Returns
        -------
        None.
        '''
        if self.cvFile is None:
            myError("No cvfile specified")
        #
        if self.static:
            cvFiles = [self.cvFile]
        else:
            cvFiles = self.cvFile
        for cvFile in cvFiles:
            if not os.path.exists(cvFile):
                myError("cvFile: {0:s} does not exist".format(cvFile))

    def readWKT(self, wktFile):
        ''' get wkt from a file '''
        with open(wktFile, 'r') as fp:
            return fp.readline()

    def setSRS(self):
        '''
        Set xy SRS proj based on wktFile or epsg,if neither defaults to
        northern  or southern lat of the cv points.
        Returns
        -------
        None.
        '''
        if len(self.lat) <= 0:
            myError("Cannot set epsg without valid latlon")
        if self.wktFile is not None:
            wkt = self.readWKT(self.wktFile)
            self.xyproj = wkt
        else:
            if self.epsg is None:
                self.epsg = [3031, 3413][self.lat[0] > 0]
            self.xyproj = f"EPSG:{self.epsg}"
        self.lltoxyXform = pyproj.Transformer.from_crs(self.llproj,
                                                       self.xyproj)
    #
    # ===================== Cv input/output stuff ============================
    #

    def readCVs(self, cvFile=None):
        ''' Read either single static file or multiple time varying files'''
        self.setCVFile(cvFile)
        self.checkCVFile()
        if self.static:
            self._readStaticCVs()
        else:
            self._readTimeVaryingCVs()

    def _readTimeVaryingCVs(self):
        ''' Read list of CV files and save result '''
        self.ptData = []
        for pointFile in self.cvFile:
            self.ptData.append(self._readPointCVFile(pointFile))
        # Force calculations with all data
        self.velocityForDateRange('1900-01-01', '2100-01-01')

    def velocityForDateRange(self, date1, date2):
        ''' Update velocity with new date range'''
        for var in ['lat', 'lon', 'vx', 'vy']:
            setattr(self, var, [])

        for pt in self.ptData:
            ptForDate = pt.loc[(pt.Date >= date1) & (pt.Date <= date2)]
            ptMean = ptForDate.loc[:, ['lat', 'lon', 'vx', 'vy', 'vv']].mean()
            #
            self.vx.append(ptMean['vx'])
            self.vy.append(ptMean['vy'])
            self.lat.append(ptMean['lat'])
            self.lon.append(ptMean['lon'])
        # convert to np
        for var in ['vx', 'vy', 'lat', 'lon']:
            setattr(self, var, np.array(getattr(self, var)))
        self.vv = np.sqrt(self.vx**2 + self.vy**2)
        self.setSRS()
        self.x, self.y = self.lltoxy(self.lat, self.lon)

    def _readPointCVFile(self, pointFile):
        ''' Read a single GPS Vel file '''
        points = []
        with open(pointFile) as fpPts:
            for line in fpPts:
                if '%' not in line:
                    pieces = line.split(',')
                    lineDate = [datetime.strptime(pieces[0].strip(),
                                                  '%Y-%m-%d')]
                    data = [float(x.strip()) for x in pieces[2:]]
                    points.append(lineDate + data)
        return pd.DataFrame(points, columns=['Date', 'lat', 'lon', 'vx',
                                             'vx_sigma', 'vy', 'vy_sigma',
                                             'vv', 'vv_sigma'])

    def _readStaticCVs(self):
        '''
        Read cv points, set projection based on hemisphere, convert to x,y (m)
        Parameters
        ----------
        cvFile : str, optional
            Name of cvFile in not already set. The default is None.
        Returns
        -------
        None.
        '''
        #
        cvCols = [latv, lonv, zv, vxv, vyv, vzv, weight] = \
            [], [], [], [], [], [], []
        # loop through file to read points
        with open(self.cvFile, 'r') as fpIn:
            for line in fpIn:
                if '#' in line and '2' in line:
                    self.pound2 = True
                if '&' not in line and ';' not in line and '#' not in line:
                    lineFields = [float(x) for x in line.split()[0:6]]
                    weight = 1.
                    # Update weight with value if present
                    if len(line.split()) == 7:
                        weight = float(line.split()[6])
                    lineFields.append(weight)
                    # print(lineFields)
                    for x, y in zip(cvCols, lineFields):
                        x.append(y)
        # append to any existing points
        for var, cvCol in \
                zip(['lat', 'lon', 'z', 'vx', 'vy', 'vz', 'weight'], cvCols):
            setattr(self, f'{var}', np.append(getattr(self, f'{var}'), cvCol))
        #
        self.vv = np.sqrt(self.vx**2 + self.vy**2)
        #
        self.setSRS()
        self.nocull = np.ones(self.vx.shape, dtype=bool)  # set all to no cull
        # to xy coords - x, y in meters internally
        self.x, self.y = self.lltoxy(self.lat, self.lon, units='m')

    def writeCVs(self, cvFileOut, fp=None, comment=None, keepOpen=False):
        '''
        Write a CV file as a list of points lat,lon,z,vx,vy,vz with a
        Parameters
        ----------
        cvFileOut : str
            Name of cvFile in not already set. The default is None.
        optional
            fpOut: ignore file name an using open file pointer
        Returns
        -------
        None.

        '''
        if comment is not None:
            self.header.append(comment)
        if fp is None:
            fpOut = open(cvFileOut, 'w')
            if self.pound2:
                print('# 2', file=fpOut)
        else:
            fpOut = fp
        # in case type file that needs this.
        for line in self.header:
            print(f'; {line}', file=fpOut)
        #
        myCVFields = zip(
            self.lat, self.lon, self.z, self.vx, self.vy, self.vz, self.weight,
            self.nocull)
        for lat, lon, z, vx, vy, vz, weight, nocull in myCVFields:
            if nocull:  # only print non-culled points
                print(f'{lat:10.5f} {lon:10.5f} {z:8.1f} '
                      f'{vx:9.2f} {vy:9.2f} {vz:8.5f} {weight:8.3f}',
                      file=fpOut)
        if not keepOpen:
            print('&', file=fpOut)  # End of data for historical reasons
            fpOut.close()
        else:
            return fpOut
    #
    # ===================== CV Select Stuff ============================
    #

    def zeroCVs(self):
        '''
        Return bool array of effectively stationary points.
        Returns
        -------
        bool
            List of all zero points (speed < 0.00001) indicated by T & F val.
        '''
        return np.abs(self.vv) < 0.00001

    def vRangeCVs(self, minv, maxv):
        '''
        Return bool array of points in a specified velocity range (minv,maxv).
        Returns
        -------
        bool
            List of points in range (vmin,vmax) indicated by T & F val.
        '''
        return np.logical_and(self.vv >= minv, self.vv < maxv)

    def allCVs(self):
        '''
        Return bool array of valid points.
        Returns
        -------
        bool
            List of all valid points (speed >=0) indicated by T & F val.
        '''
        return np.abs(self.vv) >= 0

    def NzeroCVs(self):
        '''
        Compute number of number of zero cvpoints
        Returns
        -------
        int
            Number of zero points.
        '''
        return sum(self.zeroCVs())

    def NVRangeCVs(self, minv, maxv):
        '''
        Compute number of cv points in range (minv,maxv)
        Returns
        -------
        int
            Number of points in range.
        '''
        return sum(self.vRangeCVs(minv, maxv))

    def NallCVs(self):
        '''
        Compute number of cv points
        Returns
        -------
        int
            Number of cv points.
        '''
        return sum(self.allCVs())

    #
    # ---- Interpolate Velocity Stuff
    #

    def _cvVels(func):
        ''' decorator to interpolate cv values from vel map '''
        @functools.wraps(func)
        def cvV(*args, units='m', date=None, **kwargs):
            x, y = func(*args)
            result = args[1].interp(x, y, units=units, date=date, **kwargs)
            return result
        return cvV

    @_cvVels
    def vRangeData(self, vel, minv, maxv, units='m', date=None):
        ''' Get velocity from vel map for points in range (vmin,vmax).'''
        return self.xyVRange(minv, maxv)

    @_cvVels
    def vAllData(self, vel, units='m', date=None):
        ''' Get velocity from vel map for all points.'''
        return self.xyAll()

    #
    # ---- Stats Stuff
    #

    def computeThreshMean(self, iPts, iGood, absError=1,
                          percentError=0.03):
        threshX = np.sqrt(np.mean(
            (np.ones(iGood.shape)[iGood] * absError)**2 +
            (self.vx[iPts][iGood] * percentError)**2))
        threshY = np.sqrt(np.mean(
            (np.ones(iGood.shape)[iGood] * absError)**2 +
            (self.vy[iPts][iGood] * percentError)**2))
        return threshX, threshY

    def computeThresh(self, absError=1, percentError=0.03):
        threshX = np.sqrt(absError**2 + (self.vx * percentError)**2)
        threshY = np.sqrt(absError**2 + (self.vy * percentError)**2)
        return threshX, threshY

    def _stats(func):
        ''' decorator for computing stats routines '''
        # create a table
        def pandasTable(muX, muY, sigX, sigY, rmsX, rmsY,  nPts,
                        threshX, threshY, date1, date2):
            midDateStr = \
                f'{date1.strftime("%Y-%m-%d")} - {date2.strftime("%Y-%m-%d")}'
            dfData = pd.DataFrame([[muX, muY, sigX, sigY, rmsX, rmsY]],
                                  columns=pd.MultiIndex.from_product(
                                  [['Mean', 'Sigma', 'RMS'],
                                   ['$$v_x-u_x$$', '$$v_y-u_y$$']]),
                                  index=pd.Index([midDateStr]))
            dfPoints = pd.DataFrame([[nPts]],
                                    columns=pd.MultiIndex.from_product(
                                        [['Count'], ['n']]),
                                    index=pd.Index([midDateStr]))
            dfThresh = pd.DataFrame([[threshX, threshY]],
                                    columns=pd.MultiIndex.from_product(
                                        [['Threshold'], ['x', 'y']]),
                                    index=pd.Index([midDateStr]))
            df = pd.concat([dfData, dfPoints, dfThresh], axis=1)
            return muX, muY, sigX, sigY, rmsX, rmsY, nPts, threshX, threshY, df
        # the actual stats

        @functools.wraps(func)
        def mstd(*args, table=False, date=None, absError=1,
                 percentError=0.03, **kwargs):
            if 'units' in kwargs:
                print('units is not an option for this function')
                return None
            x, y, iPts, date1, date2 = func(*args, date=date)
            dvx, dvy = args[0].cvDifferences(x, y, iPts, args[1], date=date)
            iGood = np.isfinite(dvx)
            #
            dvx = dvx[iGood]
            dvy = dvy[iGood]
            #
            # print(args[0].vx.shape, iPts.shape, iGood.shape)
            threshX, threshY = \
                args[0].computeThreshMean(iPts, iGood,
                                          absError=absError,
                                          percentError=percentError)
            muX, muY = np.average(dvx), np.average(dvy)
            rmsX = np.sqrt(np.average(dvx**2))
            rmsY = np.sqrt(np.average(dvy**2))
            sigX, sigY = np.std(dvx), np.std(dvy)
            if not table:
                return muX, muY, sigX, sigY, rmsX, rmsY, sum(iGood)
            else:
                return pandasTable(muX, muY, sigX, sigY, rmsX, rmsY,
                                   sum(iGood), threshX, threshY, date1, date2)
        return mstd

    def _processVelDate(self, date, vel):
        if date is None or len(vel.subset.time1.shape) == 0:
            if type(vel.time1) == list:
                date1 = vel.time1[0]
                date2 = vel.time2[0]
            else:
                date1 = vel.time1
                date2 = vel.time2
        else:
            date1 = vel.datetime64ToDatetime(
                    vel.subset.time1.sel(time=vel.parseDate(date),
                                         method='nearest').data)
            date2 = vel.datetime64ToDatetime(
                    vel.subset.time2.sel(time=vel.parseDate(date),
                                         method='nearest').data)
        return date1, date2

    @_stats
    def vRangeStats(self, vel, minv, maxv, date=None, units='m'):
        ''' get stats for cvpoints in range (minv,maxv) '''
        x, y = self.xyVRange(minv, maxv, units=units)
        iPts = self.vRangeCVs(minv, maxv)
        date1, date2 = self._processVelDate(date, vel)
        return x, y, iPts, date1, date2

    @_stats
    def noCullStats(self, vel, units='m', date=None):
        ''' get stats for cvpoints in range (minv,maxv) '''
        x, y = self.xyNoCull(units=units)
        if self.nocull is None:
            iPts = np.full(x.shape, True)
        else:
            iPts = self.nocull
        date1, date2 = self._processVelDate(date, vel)
        return x, y, iPts, date1, date2

    def timeSeriesStats(self, myVelSeries, minv, maxv):
        '''
        Compute stats for time series
        Parameters
        ----------
        myVelSeries : velSeries
            A nisardev velocity time series.
        minv : number
            Only includ points >= minv.
        maxv : number
            Only include points <- maxv.

        Returns
        -------
        df : pandas dataframe
            Data frame with the results.

        '''
        result = self.timeSeriesDifferences(myVelSeries, minv, maxv)
        nPts = result['dvx'].shape[1]
        sigmaX, sigmaY = np.zeros(nPts), np.zeros(nPts)
        meanX, meanY = np.zeros(nPts), np.zeros(nPts)
        threshX, threshY = np.zeros(nPts), np.zeros(nPts)
        nGood =  np.zeros((nPts, 1), dtype='i4')
        summary = {}
        summary['nPts'] = nPts
        for pt in range(0, nPts):
            for mean, sigma, thresh, band in zip([meanX, meanY],
                                                 [sigmaX, sigmaY],
                                                 [threshX, threshY],
                                                 ['vx', 'vy']):
                mean[pt] = np.nanmean(result[f'd{band}'][:, pt])
                sigma[pt] = np.nanstd(result[f'd{band}'][:, pt])
                thresh[pt] = np.sqrt(np.mean(
                    result[f'thresh{band}'][:, pt]**2))
            #
            nGood[pt] = int(np.sum(np.isfinite(result['vx'][:, pt])))

        rmsX = np.sqrt(meanX**2 + sigmaX**2)
        rmsY = np.sqrt(meanY**2 + sigmaY**2)
        summaryStats = np.array([meanX, meanY, sigmaX, sigmaY,
                                 rmsX, rmsY]).transpose()

        dfData = pd.DataFrame(summaryStats,
                              columns=pd.MultiIndex.from_product(
                              [['Mean', 'Sigma', 'RMS'],
                               ['$$v_x-u_x$$', '$$v_y-u_y$$']]),
                              index=range(1,nPts+1))
        dfPoints = pd.DataFrame(nGood,
                                columns=pd.MultiIndex.from_product(
                                    [['Count'], ['n']]),
                                index=range(1,nPts+1))
        dfThresh = pd.DataFrame(np.array([threshX, threshY]).transpose(),
                                columns=pd.MultiIndex.from_product(
                                    [['Threshold'], ['x', 'y']]),
                                index=range(1,nPts+1))
        df = pd.concat([dfData, dfPoints, dfThresh], axis=1)
        summary['nPassed'] = (
            (df["RMS"]['$$v_x-u_x$$'] <  df["Threshold"]['x']).sum() +
            (df["RMS"]['$$v_y-u_y$$'] < df["Threshold"]['y']).sum())
        summary['nGoodPoints'] = nGood.sum()
        summary['totalPoints'] = \
            result['vx'].shape[0] * result['vx'].shape[1]
        summary['percentCoverage'] = \
            summary['nGoodPoints'] / summary['totalPoints'] * 100.
        return df, summary

    def statsStyle(self, styler, thresh=1.0, caption=None):
        '''
        Setup the pandas style information to display a stats table.

        Parameters
        ----------
        styler : pandas.io.formats.style.Styler
            style for stats table.
        thresh : float, optional
            RMS values that exceed thresh are set to red. The default is 1.0.
        Returns
        -------
        styler : TYPE
            DESCRIPTION.
        '''
        if caption is not None:
            styler.set_caption(caption)
        # precesion for floating point columns
        styler.format({('Mean', '$$v_x-u_x$$'): "{:.2f}",
                       ('Mean', '$$v_y-u_y$$'): "{:.2f}",
                       ('Sigma', '$$v_x-u_x$$'): "{:.2f}",
                       ('Sigma', '$$v_y-u_y$$'): "{:.2f}",
                       ('RMS', '$$v_x-u_x$$'): "{:.2f}",
                       ('RMS', '$$v_y-u_y$$'): "{:.2f}",
                       ('Count', 'n'): "{:d}",
                       ('Threshold', 'x'): "{:.2f}",
                       ('Threshold', 'y'): "{:.2f}"})
        # Format specs for table
        styler.set_table_styles([
            {'selector': 'th.col_heading',
             'props': 'text-align: center; background-color: lightblue; '
             'border: 1px solid black; font-size: 12pt'},
            {'selector': 'th.col_heading.level1',
             'props': 'text-align: center; background-color: light blue; '
             'border: 1px solid black; font-size: 12pt'},
            {'selector': 'th.row_heading',
             'props': 'background-color: lightblue; text-align: center; '
             'border: 1px solid black; font-size: 12pt'},
            {'selector': 'td',
             'props': 'text-align: center; border: 1px solid black; '
             'font-size: 12pt'},
            {'selector': 'caption',
             'props': 'font-size: 14pt; text-align: center; '
             'caption-side: bottom'}]
            )
        # Use threshold to set RMS values that exceed thresh
        #

        def setColors(s, column='Count'):
            #
            result = ['color:black']*9
            if s.iloc[4] > s.iloc[7]:
                result[4] = 'color:red;'
            if s.iloc[5] > s.iloc[8]:
                result[5] = 'color:red;'
            return result
        styler.apply(setColors, axis=1, column='Count')
        return styler
    #
    # ---- Coordinate Stuff
    #

    def lltoxy(self, lat, lon, units='m'):
        '''
        Convert lat lon (deg) to x y (m)
        Parameters
        ----------
        lat : nparray
            Latitude nparray.
        lon : nparray
            Longitude nparray.
        Returns
        -------
        x : nparray
            x coordinate in m.
        y : nparray
            y coordinate in m..
        '''
        if not self._checkUnits(units):
            return None, None
        #
        if self.xyproj is not None:
            x, y = self.lltoxyXform.transform(lat, lon)
            if units == 'km':
                return self._toKM(x, y)
            return x, y
        else:
            myError("lltoxy: proj not defined")

    def llzero(self):
        '''
        Return lat and lon of zero CVs.
        Returns
        -------
        lat, lon : nparray
            lat and lon of zero CVs.
        '''
        iZero = self.zeroCVs()
        return self.lat[iZero], self.lon[iZero]

    def _toKM(self, x, y):
        return x/1000., y/1000.

    def xyZero(self, units='m'):
        '''
        Return x and y (m) coordinates of zero CVs.
        Returns
        -------
        x,y  : nparray
           x and y in m of zero CVs.
        '''
        if not self._checkUnits(units):
            return None, None
        # find zero points
        iZero = self.zeroCVs()
        if units == 'km':
            return self._tokm(self.x[iZero], self.y[iZero])
        return self.x[iZero], self.y[iZero]

    def xyAll(self, units='m'):
        '''
        Return x and y (m) coordinates of all CVs.
        Returns
        -------
        x,y  : nparray
           x and y in m of all CVs.
        '''
        if not self._checkUnits(units):
            return None, None
        #
        if units == 'km':
            return self._toKM(self.x, self.y)
        return self.x, self.y

    def xyNoCull(self, units='m'):
        '''
        Return x and y (m) coordinates of all CVs.
        Returns
        -------
        x,y  : nparray
           x and y in m of all CVs.
        '''
        if not self._checkUnits(units):
            return None, None
        # If no culling, return all
        if self.nocull is None:
            return self.xyAll(units=units)
        #
        if units == 'km':
            return self._toKM(self.x[self.nocull], self.y[self.nocull])
        return self.x[self.nocull], self.y[self.nocull]

    def xyVRange(self, minv, maxv, units='m'):
        '''
        Return x and y (m) coordinates for pts with speed in range (minv,maxv).
        Returns
        -------
        x,y  : nparray
           x and y in m of all CVs.
        '''
        if not self._checkUnits(units):
            return None, None
        # find points and return
        iRange = self.vRangeCVs(minv, maxv)
        if units == 'km':
            return self._toKM(self.x[iRange], self.y[iRange])
        return self.x[iRange], self.y[iRange]

    #
    # ===================== Plot Locations ============================
    #

    def _plotCVLocs(func):
        ''' Decorator for plotting locations in range (vmin,vmax). '''
        @functools.wraps(func)
        def plotCVXY(*args, vel=None, ax=None, nSig=3, units='m', date=None,
                     **kwargs):
            x, y = func(*args, units=units, nSig=nSig)
            #
            if vel is not None:
                result = vel.interp(x, y, date=date)
                iGood = np.isfinite(result[:, 0])
                # iGood = result[-1]
                x, y = x[iGood], y[iGood]

            for keyw, value in zip(['marker', 'linestyle'], ['.', 'None']):
                if keyw not in kwargs:
                    kwargs[keyw] = value
            if ax is None:
                plt.plot(x, y, **kwargs)
            else:
                ax.plot(x, y,  **kwargs)
        return plotCVXY

    @_plotCVLocs
    def plotVRangeCVLocs(self, minv, maxv, units='m', date=None, **kwargs):
        '''
        plot x,y locations for points where maxv > v > min.
        Parameters
        ----------
        minv : float
            minimum speed of desired range.
        maxv : float
            maximum speed of desired range.
        Returns
        -------
        None
            DESCRIPTION.
        '''
        if not self._checkUnits(units):
            return None, None
        return self.xyVRange(minv, maxv, units=units)

    @_plotCVLocs
    def plotAllCVLocs(self, units='m', date=None, **kwargs):
        '''
        plot x,y locations for points where maxv > v > min.
        Parameters
        ----------
        None.
        Returns
        -------
        None
            DESCRIPTION.
        '''
        if not self._checkUnits(units):
            return None, None
        return self.xyAll(units=units)

    def getOutlierLocs(self, minv, maxv, vel, units='m', nSig=3):
        ''' Get outliers where difference in either velocity component is >
        nSig*sigma for points in range (minv,maxv)
        Parameters
        ----------
        minv : float
            minimum speed of desired range.
        maxv : float
            maximum speed of desired range.
        vel : nisarVel
            A nisarVel object
        Returns
        -------
        None
            DESCRIPTION.
        '''
        if not self._checkUnits(units):
            return None, None
        x, y = self.xyVRange(minv, maxv, units=units)  # get points in range
        # Just points in range (minv,vmaxv)
        iPts = self.vRangeCVs(minv, maxv)
        dvx, dvy = self.cvDifferences(x, y, iPts, vel, units=units)
        # Compute valid points and reduce all variables to good points
        iGood = np.isfinite(dvx)
        dvx, dvy = dvx[iGood], dvy[iGood]
        x, y = x[iGood], y[iGood]  # points in range with valid diffs
        # status
        sigX, sigY = np.std(dvx), np.std(dvy)  # Sigmas
        # Find outliers
        iOut = np.logical_or(np.abs(dvx) > nSig * sigX,
                             np.abs(dvy) > nSig * sigY)
        return x[iOut], y[iOut]

    @_plotCVLocs
    def plotOutlierLocs(self, minv, maxv, vel, units='m', nSig=3):
        ''' Plot outliers where difference in either velocity component is >
        nSig*sigma for points in range (minv,maxv)
        Parameters
        ----------
        minv : float
            minimum speed of desired range.
        maxv : float
            maximum speed of desired range.
        vel : nisarVel
            A nisarVel object
        Returns
        -------
        None
            DESCRIPTION.
        '''
        return self.getOutlierLocs(minv, maxv, vel, units=units, nSig=nSig)

    #
    # ===================== CV plot differences ===============================
    #

    def _genPlotColors(self, nColors):
        '''
        Create a color table for plots with repeating colors if needed.

        Parameters
        ----------
        nColors : int
            The Numpber of colors to create.
        Returns
        -------
        colors : list
            List of colors.
        '''
        colors = mcolors.TABLEAU_COLORS.values()
        # Cycle colors if more are needed
        while len(colors) < nColors:
            colors += mcolors.TABLEAU_COLORS.values()
        return colors



    def plotTimesSeriesData(self, myVelSeries, minv, maxv,
                            bands=['vv'], figsize=None, labelFontSize=14,
                            nTicks=None):
        '''
        Plot a time varying series of cal/val points for a velocity series
        for a specified range of speeds.
        Parameters
        ----------
        myVelSeries : velSeries
            A nisardev velocity time series.
        minv : number
            Only includ points >= minv.
        maxv : number
            Only include points <- maxv.
        bands : list, optional
            List of bands ('vx', 'vy', or 'vv'). The default is ['vv'].
        figsize :  tuple, optional
            Size of figure The default is None, which yields (7*nbands, 5).
        labelFontSize : number, optional
            Fontsize for labels. The default is 14.
        nTicks : int, optional
            The number of ticks to use on x-axis. The default is None.
        Returns
        -------
        None.
        '''
        # Compute point by point differences
        result = self.timeSeriesDifferences(myVelSeries, minv, maxv)
        # Plot stuff
        if figsize is None:
            figsize = (7*len(bands), 5)
        colors = self._genPlotColors(len(result['vv']))

        fig, axes = plt.subplots(1, len(bands), figsize=figsize)
        ylabels = {'vx': '$v_x, u_x$', 'vy': '$v_y, u_y$',
                   'vv': '$|v|$'', ''$|u|$'}
        #
        if len(bands) == 1:
            axes = np.array([axes])
        # Loop over bands to plot
        for band, ax in zip(bands, axes.flatten()):
            # Loop over point to plot
            for i, color in zip(range(result['vv'].shape[1]), colors):
                ax.plot(myVelSeries.time, result[f'{band}'][:, i], '*',
                        color=color, markersize=12, label=f'Pt{i+1} data')
                ax.plot(myVelSeries.time, result[f'{band}GPS'][:, i], 'o-',
                        color=color, label=f'Pt{i+1} GPS')
                if band != 'vv':
                    ax.errorbar(myVelSeries.time, result[f'{band}'][:, i],
                                yerr=result[f'thresh{band}'][:, i])
            # Plot labels and legends
            ax.set_xlabel('Date', fontsize=labelFontSize)
            ax.set_ylabel(f'{ylabels[band]} (m/yr)',
                          fontsize=labelFontSize)
            ax.tick_params(axis='both', labelsize=int(labelFontSize*.8))
            if nTicks is not None:
                ax.xaxis.set_major_locator(plt.MaxNLocator(nTicks))
            ax.legend(ncol=3, bbox_to_anchor=(1, 1.15))

    def _plotDiffs(func):
        '''
        Decorator for plotting differences between tie points and
        velocities interpolated from in vel map
        Parameters
        ----------
        func : function
            Function being decorated.
        Returns
        -------
        ax
            Axis for the plot.
        '''
        @functools.wraps(func)
        def plotp(inst, *args,  ax=None, xColor='r', yColor='b', date=None,
                  legendKwargs={}, **kwargs):
            if 'units' in kwargs:
                print('units is not an option for this function')
                return
            x, y, iPts = func(inst, *args)
            dx, dy = dask.compute(inst.cvDifferences(x, y, iPts, args[0],
                                                     date=date))[0]
            # defaults
            for keyw, value in zip(['marker', 'linestyle'], ['.', 'None']):
                if keyw not in kwargs:
                    kwargs[keyw] = value
            #
            if 'fontsize' not in legendKwargs:
                legendKwargs['fontsize'] = inst.legendFontSize
            #
            if ax is None:
                ax = plt.subplot(111)
            #
            ax.plot(dx, color=xColor, label='$v_x-u_x$',  **kwargs)
            ax.plot(dy, color=yColor, label='$v_y-u_y$',  **kwargs)
            ax.set_xlabel('Point', size=inst.labelFontSize)
            ax.set_ylabel('$u_x-v_x, u_y-v_y$ (m/yr)',
                          size=inst.labelFontSize)
            ax.legend(**legendKwargs)
            ax.tick_params(axis='x', labelsize=inst.plotFontSize)
            ax.tick_params(axis='y', labelsize=inst.plotFontSize)
            return ax
        return plotp

    @_plotDiffs
    def plotVRangeCVDiffs(self, vel, minv, maxv, **kwargs):
        '''
        Plot differences between c/v points and interpolated values from
        v in range (minv,maxv).
        Originall written with decorator to accomodate multiple cases,
        but collapsed it down to one. Kept decorator for future mods.
        Parameters
        ----------
        vel : nisarVel
            Velocity map for comparison.
        minv : float
            minimum speed of desired range.
        maxv : float
            maximum speed of desired range.
        Returns
        -------
        x : nparray
            x coordinates.
        y : nparray
            y coordinates.
        iPts : bool array
            points.
        '''
        x, y = self.xyVRange(minv, maxv)
        iPts = self.vRangeCVs(minv, maxv)
        return x, y, iPts

    def _histDiffs(func):
        '''
        Decorator for plotting differences between tie points and
        velocities interpolated from in vel map
        Parameters
        ----------
        func : function
            Function being decorated.
        Returns
        -------
        axHistX, axHistY
            Axes for the hist plots.
        '''
        # the actual stats
        def clipTail(dx):
            ''' Clip differences to +/- 3 sigma '''
            threeSigma = 3 * np.std(dx)
            dx[dx < -threeSigma] = -threeSigma
            dx[dx > threeSigma] = threeSigma
            return dx

        @functools.wraps(func)
        def ploth(inst, *args, axes=None, xColor='r', yColor='b', date=None,
                  **kwargs):
            if 'units' in kwargs:
                print('units is not an option for this function')
                return
            x, y, iPts = func(inst, *args)
            dx, dy = dask.compute(inst.cvDifferences(x, y, iPts, args[0],
                                                     date=date))[0]
            # iGood = np.isfinite(dx)
            dx, dy = clipTail(dx), clipTail(dy)  # Set vals > 3sig to 3sig
            if axes is None:
                _, axes = plt.subplots(1, 2)
            colors = {'x': xColor, 'y': yColor}
            for keyw, value in zip(['edgecolor', 'density', 'bins'],
                                   ['k', 'True', 'auto']):
                if keyw not in kwargs:
                    kwargs[keyw] = value
            # produce and label plots for x and y compnents
            for var, lab, axH in zip([dx, dy], ['x', 'y'], axes):
                axH.hist(var, facecolor=colors[lab], **kwargs)
                axH.set_xlabel(f'$u_{lab}-v_{lab}$ (m/yr)',
                               size=inst.labelFontSize)
                axH.set_ylabel('Relative Frequency',
                               size=args[0].labelFontSize)
                axH.tick_params(axis='x', labelsize=inst.plotFontSize)
                axH.tick_params(axis='y', labelsize=inst.plotFontSize)
            return axes
        return ploth

    @_histDiffs
    def plotVRangeHistDiffs(self, vel, minv, maxv, date=None):
        '''
        Plot differences between c/v points and interpolated values from
        v in range (minv,maxv).
        Originall written with decorator to accomodate multiple cases,
        but collapsed it down to one. Kept decorator for future mods.
        Parameters
        ----------
        vel : nisarVel
            Velocity map for comparison.
        figD : matplot lib fig
            Figure.
        minv : float
            minimum speed of desired range.
        maxv : float
            maximum speed of desired range.
        Returns
        -------
        x : nparray
            x coordinates.
        y : nparray
            y coordinates.
        iPts : bool array
            points.
        '''
        x, y = self.xyVRange(minv, maxv)
        iPts = self.vRangeCVs(minv, maxv)
        return x, y, iPts

    def timeSeriesDifferences(self, myVelSeries, minv, maxv, absError=10,
                              percentError=0.03):
        '''
        Compute the difference between cal/val GPS points and data for points
        in range specified by minv and maxv.

        Parameters
        ----------
        myVelSeries : velSeries
            Velocity time series.
        minv : number
            Only includ points >= minv.
        maxv : number
            Only include points <- maxv.
        Returns
        -------
        result : dict
            {'vxGPS': [], ... 'vx': [],...'dvx': []...}.
        '''
        dates = zip(myVelSeries.time, myVelSeries.time1, myVelSeries.time2)
        result = {'vxGPS': [], 'vyGPS': [], 'vvGPS': [], 'vx': [],
                  'vy': [], 'vv': [], 'dvx': [], 'dvy': [], 'dvv': [],
                  'threshvx': [], 'threshvy': []}
        for myDate, myDate1, myDate2 in dates:
            # This will for set the tiepoints for the date range
            self.velocityForDateRange(myDate1, myDate2)
            # This will pull the coordinates for the date speed range
            x, y = self.xyVRange(minv, maxv)
            iPts = self.vRangeCVs(minv, maxv)
            r = myVelSeries.interp(x, y, returnXR=True).sel(time=myDate,
                                                            method='nearest')
            for band in ['vx', 'vy', 'vv']:
                result[band].append(r.sel(band=band).data)
                result[f'{band}GPS'].append(getattr(self, band)[iPts])
                d = r.sel(band=band).data - getattr(self, band)[iPts]
                result[f'd{band}'].append(d)
            # Compute thresholds for each point
            threshX, threshY = self.computeThresh(absError=absError,
                                                  percentError=percentError)
            result['threshvx'].append(threshX)
            result['threshvy'].append(threshY)
        for key in result:
            result[key] = np.array(result[key])
        return result

    def cvDifferences(self, x, y, iPts, vel, units='m', date=None):
        '''
        Interpolate the velocity components from velocity map at the c/v
        point locations and return the differences (vx_map - vx_cv).
        Parameters
        ----------
        x : nparray
            x coordinates.
        y : nparray
            y coordinates.
        iPts : bool array
            points to compare.
        vel : nisarVel
            Velocity map for comparison.
        Returns
        -------
        dvx nparray
            vx difference for good points.
        dvy nparray
            vy difference for good points.
        '''
        vx, vy, vv = vel.interp(x, y, units=units, date=date)[0:3]
        # subtract cvpoint values args[0] is self
        dvx, dvy = vx - self.vx[iPts], vy - self.vy[iPts]
        # Return valid points and locations where good
        return dvx, dvy

    #
    # ===================== CV culling stuff  ==============================
    #

    def readCullFile(self, cullFile):
        '''
        Read a cull file. For an original cvFile, a culling process was
        applied and the results saved in the cullFile.
        Parameters
        ----------
        cullFile : str
            File name for cull file.The file contains a
        Returns
        -------
        cullPoints: nparray
            Indices of culled points.
        '''
        with open(cullFile, 'r') as fp:
            myParams = eval(fp.readline())
            print(myParams["cvFile"])
            print(self.cvFile)
            cullPoints = []
            for line in fp:
                cullPoints.append(int(line))
            if len(cullPoints) != myParams["nBad"]:
                myError(f'reading culled points expected {myParams["nBad"]}'
                        ' but only found {len(cullPoints) }')
        #
        return np.array(cullPoints)

    def applyCullFile(self, cullFile):
        '''
        Read a cv point cull file and update nocull so that these points
        can be filtered out if needed.
        Parameters
        ----------
        cullFile : str
            File name for cull file.The file contains a
        Returns
        -------
        None
        '''
        #
        self.header.append(cullFile)
        toCull = self.readCullFile(cullFile)
        self.applyCull(toCull)

    def setNoCull(self, noCull):
        ''' Pass in list of points not to cull
         Parameters
        ----------
        cullFile : list of points to not cull (true means keep)
        '''
        self.nocull = noCull

    def applyCull(self, toCull):
        '''
        Read a cv point cull file and update nocull so that these points
        can be filtered out if needed.
        Parameters
        ----------
        cullFile : str
            File name for cull file.The file contains a
        Returns
        -------
        None
        '''
        #
        if len(toCull) > 0:
            self.nocull[toCull] = False

    def boundingBox(self, units='m', x=None, y=None, pad=10000.):
        '''
        Compute Bounding box for tiepoints
        Returns
        -------
        bounding box {'minx': minxm, 'miny': miny, 'maxx': maxx, 'maxy' : maxy}

        '''
        if x is None or y is None:
            x, y = self.xyAll(units=units)
        values = np.around([np.min(x) - pad, np.min(y) - pad,
                            np.max(x) + pad, np.max(y) + pad], -2)
        return dict(zip(['minx', 'miny', 'maxx', 'maxy'], values))

#
# ---- Error Checking
#

    def _checkUnits(self, units):
        '''
        Check units return True for valid units. Print message for invalid.
        '''
        if units not in ['m', 'km']:
            print('Invalid units: must be m or km')
            return False
        return True
