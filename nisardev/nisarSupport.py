#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:05:04 2020

@author: ian
"""
from datetime import datetime
from osgeo import gdal
import numpy as np
import sys
import os


def myError(message):
    '''
    Print error message and exit
    Parameters
    ----------
    message : str
        String with informative error message.
    Returns
    -------
    None.
    '''
    print(f'\n\t\033[1;31m *** {message} *** \033[0m\n')
    sys.exit()


def readGeoTiff(tiffFile, noData=-2.e9):
    '''
    Read a geotiff file and return the array
    Parameters
    ----------
    tiffFile : str
        Name of tif file with tif suffix.
    noData : data dependent, optional
        No data value. The default is -2.e9.
    Returns
    -------
    arr : data dependent
        Band from tiff.
    '''
    try:
        gdal.AllRegister()
        ds = gdal.Open(tiffFile)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        arr = np.flipud(arr)
        ds = None
    except Exception:
        myError(f"nisarSupport readMyTiff: error reading tiff file {tiffFile}")
    arr[np.isnan(arr)] = np.nan
    arr[arr <= noData] = np.nan
    return arr


def setKey(myKey, defaultValue, **kwargs):
    '''
    Unpack a keyword from **kwargs and if does not exist
    return defaultValue.
    Parameters
    ----------
    myKey : str
        The key of interest.
    defaultValue : key dependent
        The default value to return if key does not exist.
    **kwargs : TBD
        Keyword passthrough.
    Returns
    -------
    keywordvalue : key dependent
        Value for that keyword.

    '''
    if myKey in kwargs.keys():
        return kwargs(myKey)
    return defaultValue


def parseDatesFromDirName(dirName, dateTemplate, divider):
    '''
    Parse date from dir name with a template such as:
    Vel-2015-01-01.2015-12-31
    Parameters
    ----------
    dirName : str
        Name of the directory.
    dateTemplate : str
        Template for dir name. The default is 'Vel-%Y-%m-%d.%Y-%m-%d'.
    divider : str
        character that divides up name xyz.date1.date2. The default is '.'.
    Returns
    -------
    dates : [datetime, datetime]
        First and last dates from meta file.
    '''
    dates = []
    for dN, dT in zip(dirName.split(divider), dateTemplate.split(divider)):
        print(dN, dT)
        if '%' in dT:
            print(datetime.strptime(dN, dT))
            dates.append(datetime.strptime(dN, dT))
    if len(dates) != 2:
        myError(f'parseDatesFromDirName: could not parse dates from {dirName}')
    return dates


def parseDatesFromMeta(metaFile):
    '''
    Parse dates from a GIMP meta file.
    Parameters
    ----------
    metaFile : str, optional
        metaFile name, if not specified use basename. The default is None.
    Returns
    -------
    dates : [datetime, datetime]
        First and last dates from meta file.
    '''
    if not os.path.exists(metaFile):
        myError(f'parseDatesFromMeta: metafile {metaFile} does not exist.')
    fp = open(metaFile)
    dates = []
    for line in fp:
        if 'MM:DD:YYYY' in line:
            tmp = line.split('=')[-1].strip()
            dates.append(datetime.strptime(tmp, "%b:%d:%Y"))
            if len(dates) == 2:
                break
    fp.close()
    if len(dates) != 2:
        myError(f'parseDatesFromMeta: could parse 2 dates in {metaFile}')
    return dates
