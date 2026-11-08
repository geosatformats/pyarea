__author__ = 'jano'

"""
    This module contains navigation functions and constants to reproject
     ata using Proj4 library form geostationary projection GEOS(SSP) to LATLON



"""

import math

import numpy as np
import pyproj


def lc2YX(l=None,c=None,gt=None):

    xmin, xres, _, ymax, __, yres = gt
    return ymax + l*yres+yres/2., xmin + c*xres + xres/2.


def YX2lc(Y=None, X=None, gt=None):
    """
    Given a GDAL GeoTransform http://www.gdal.org/gdal_datamodel.html  and an image of certain size (nl, nc)
    converts Geostationary coordinates Y and X into raw|pixel coordinates (lc)

    :param Y: number, float, froy axis geostationary coordinate
    :param X: number, float, x axis geostationary coordinate
    :param gt: iterable, GDAL geotransform

    :return: tuple of ints representing the line&column coordinates corresponding to Y and X
    """

    xmin, xres, _, ymax, __, yres = gt
    ty = ((Y-ymax)/yres) + .5
    tx = ((X-xmin)/xres) + .5
    try:
        rc = int(math.floor(ty)), int(math.floor(tx))
    except TypeError:
        rc = np.floor(ty).astype(np.uint32), np.floor(tx).astype(np.uint32)
    return rc

def lc2ll(l=None, c=None, ssp=None, gt=None):
    Y,X = lc2YX(l=l, c=c, gt=gt)
    return YX2ll(Y=Y, X=X, ssp=ssp)

def YX2ll(Y=None, X=None, ssp=None):
    proj4_str = '+proj=geos +a=6378169.0 +b=6356583.8 +h=35785831.0 +lon_0=%.2f +units=m' % ssp
    geosp = pyproj.Proj(proj4_str)
    lon, lat = geosp(X,Y,inverse=True)
    lat = np.where(lat==1e30,np.nan, lat)
    lon = np.where(lon==1e30,np.nan, lon)
    return lat, lon

def ll2YX(lat=None, lon=None, ssp=None):
    """
    COnverts lat, lon coordinates to GEOS(ssp) projection using Proj4 library

    :param lat: number or numpy array, input latitude,
    :param lon:, number or numpy array, input longitude
    :param ssp: number, float, sub-satellite point
    :return:
    """

    proj4_str = '+proj=geos +a=6378169.0 +b=6356583.8 +h=35785831.0 +lon_0=%.2f +units=m' % ssp
    geosp = pyproj.Proj(proj4_str)
    try: # fix the bug in Proj4 that Tomas run into. The bug is related to not
        shp=lon.shape
        X, Y = geosp(list(lon.flatten()), list(lat.flatten()))
        X = np.array(X).reshape(shp)
        Y = np.array(Y).reshape(shp)
        return Y, X
    except AttributeError:
        X, Y = geosp(lon, lat)
        return Y,X

def ll2lc(lat=None, lon=None, gt=None, ssp=None):
    """
    Converts LATLON coordinates into line/column (RAW) coordinates using Proj4 lib
    :param lat: number of numpy array, input latitude
    :param lon: number or numpy array, input longitude
    :param gt:, iter, GDAL geotransform in the source (GEOS) projection
    :param ssp: number , float, sub-satellite point

    :return:
    """
    Y,X = ll2YX(lat=lat, lon=lon,ssp=ssp)
    return YX2lc(Y=Y, X=X,gt=gt)


def reproject_geos_2_latlon(array_in=None, bbox=None, ssp=None, gt=None):
    """
    Reprojects a geostationary image from GEOS(SSP) to LATLON using PROJ4

    :param array_in: 2D numpy array, input image
    :param bbox: instance of bounding_box object, the pyarea in geographic coordinates that will be reprojected from the input image
    :param ssp: number, float, input image sub-satellite point in degrees
    :param gt: inter, GDAL geotransform
    :return: 2D numpy array representing the  reprojected input image
    """
    lats = bbox.latitudes(array2d=True)
    lons = bbox.longitudes(array2d=True)

    l, c = ll2lc(lat=lats, lon=lons, gt=gt, ssp=ssp)

    return  array_in[l,c]

