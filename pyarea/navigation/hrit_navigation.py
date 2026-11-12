__author__ = 'jano'



"""
Generic HRIT based navigation of geostationary images.
This module performs navigation of images from geostatinary to LATLON anv viceversa
Four coordinate systems are used :

1. l-c, index grid coordinates, 0-number of lines, 0, number of columns
2. y-x, intermediary satellite coordinates, radians, the pixel size as viewed from the satellite
3. lat-lon, degrees
4. Y-X, meters

"""

import numpy as np

DEG 	= np.pi / 180
RAD 	= 1.0 / DEG
satdist = 42164.		#Distance from Earths center to virtual satellite (Rs)
req		= 6378.1370		#Earths equatorial radius (req)
rpol 	= 6356.7523		#Earths polar radius (rpol)


#the column offset and  line scaling factor for 1KM resolution

OFFSET_1000M = 5500
SCALE_FACTOR_1000M = 40932549


def compute_offset(resolution=None):
    """
    Computes the column/line offset of a given image
    :param resolution:, number float, the image resolution
    :return: the column/line offset navigation coefficient
    """
    scale_fact = 1000 / float(resolution)
    return int(round(OFFSET_1000M * scale_fact))


def compute_scale_factor(resolution=None):
    """
    Computes the column/line scaling factor for a given image
    :param resolution: number, float, the image resolution
    :return: the column line navigation scaling factor
    """
    scale_fact = 1000/float(resolution)
    return int(round(SCALE_FACTOR_1000M * scale_fact))


def get_navigation_dict(resolution=None, nl=None, nc=None):
    """
    COmputes a the navigation dictionary for a given image
    :param resolution: number, float, the image resolution
    :param nl: number, int, the number of lines in the image
    :param nc: number, nc the number of columns in the image
    :return: dict, the navigatio doictionary of  image.
     The navigation dictionary contains folowing parameters:
        RESOLUTION, NL, NC, COFF, LOFF, CFAC, CLFAC
    """
    nav_dict = dict()
    nav_dict['RESOLUTION'] = float(resolution)
    nav_dict['NL'] = int(nl)
    nav_dict['NC'] = int(nc)

    scaling_factor = compute_scale_factor(resolution=resolution)
    offset = compute_offset(resolution=resolution)

    nav_dict['COFF'] = offset
    nav_dict['LOFF'] = offset
    nav_dict['CFAC'] = scaling_factor
    nav_dict['LFAC'] = scaling_factor
    return nav_dict






def ll2yx(lat=None, lon=None, lon0=None ):
    """
        Transform lat/lon coordinates to intermediary or satellite(x,y) coordinates expressed in degrees
        These step difference between these coordinates is supposed to be the native resolution of the satellite
        (This is how satellite looks at the earth)
        For details consult http://www.cgms-info.org/documents/cgms-lrit-hrit-global-specification-%28v2-8-of-30-oct-2013%29.pdf
        section 4.4.3.1

        constants used here (http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v11.pdf)
        0.993243	- rpol2 / req2
        0.00675701	- (req2 - rpol2) / req2
        6356.5838	- Earths polar radius (rpol) [km]

        :param lat: latitude, float/int or 1D numpy array of dtype float, int,
        :param lon: longitude, float/int  or 1D numpy array of dtype float, int
        :param lon_0:, subsatellite longitude, float, int
        :return: a tuple representing (intermediary) satellite coordinates(degrees) corresponding to the lat/lon. It the arguments are numpy arrays
        then the tuple consists of 2 2D arrays, else 2 scalars
    """
    if lat is None or lon is None:
        return None, None

    try:
        #broadcasting is faster here
        lat = lat[:,np.newaxis]
        lon = lon[np.newaxis,:]
    except Exception: # scalar version, r
        pass


    c_lat = np.arctan(((rpol**2)/(req**2)) * np.tan(lat * DEG))
    rl = rpol / np.sqrt(1. - ((((req**2)-(rpol**2))/(req**2)) * np.cos(c_lat) * np.cos(c_lat)))

    r1 = satdist - (rl * np.cos(c_lat) * np.cos((lon - lon0) * DEG))
    r2 = -rl * np.cos(c_lat) * np.sin((lon - lon0) * DEG)
    r3 = rl * np.sin(c_lat)
    rn = np.sqrt((r1 * r1) + (r2 * r2) + (r3 * r3))

    y = None
    x = None
    #isvisible = rn*rn + np.square(rl*c_lat)
    #if isvisible> np.square(satdist):
    #	return y,x

    x = np.arctan(-r2 / r1) * RAD
    y = np.arcsin(-r3 / rn) * RAD
    return y, x

def yx2ll(y=None, x=None, lon0=None ):
    """
    Transform intermediary or satellite(x,y) coordinates expressed in degrees to lat/lon coordinates.
    http://2014.cgms-info.SatErthCenterDistanceorg/documents/cgms-lrit-hrit-global-specification-%28v2-8-of-30-oct-2013%29.pdf

    constants used here (http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v11.pdf)
    42164. 		- Distance from satellite to Earth center in kilometers
    1.006739501	- req2 / rpol2
    1737122264.0- Coefficient for Sd (Rs2 - req2)

        :param y: satellite coordinates, float
        :param x: satellite coordinates, float
        :param lon_0:, subsatellite longitude, float, int
        :return: longitude, latitude
    Note Jano. if the input is 1D thne broadcasting is used and, i have found out that LAT/LON indices were flipped so
    I transposed thme back to tor correct position.
    This behaviour might not be present in case 2D input is used
    TODO: verify this 1D/2D behaviour
    """
    if y is None or x is None:
        return None, None
    try:
        # broadcasting is faster here
        x = x[:, np.newaxis]
        y = y[np.newaxis, :]
    except Exception:  # scalar version, r
        pass


    # converts to radians
    x = x*DEG
    y = y*DEG

    lon0Rad = lon0*DEG

    #### PARTIAL CALCULATIONS for better overview ####
    sdp1 = satdist * np.cos(x) * np.cos(y)
    sdp1sq = (sdp1**2)
    sdp2 = (np.cos(y)**2) + (((req**2)/(rpol**2)) * (np.sin(y)**2))

    sdp3 = sdp1sq - sdp2 * ((satdist**2)-(req**2))
    vis_mask = sdp3 < 0
    # if sdp3 < 0:
    # 	return None, None
    ####
    sdp3[vis_mask] = np.nan
    sd = np.sqrt(sdp3)

    snp1 = (satdist * np.cos(x) * np.cos(y)) - sd
    snp2 = (np.cos(y) * np.cos(y)) + (((req**2)/(rpol**2)) * np.sin(y) * np.sin(y))

    sn = snp1 / snp2
    s1 = satdist - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    #s3 = sn * np.sin(y)
    sxy = ((s1 * s1) + (s2 * s2))**0.5
    lon = np.arctan(s2 / s1) + lon0Rad
    lat = np.arctan(((req**2)/(rpol**2)) * (s3 / sxy))

    # converts to degrees
    lonDeg = lon * RAD
    latDeg = lat * RAD

    # Solution of 180deg meridian
    lonDeg = ((lonDeg + 180.0) % 360.0) - 180.0
    # rlatDeg = np.where(vis_mask, latDeg, np.nan)
    # rlonDeg = np.where(vis_mask, lonDeg, np.nan)
    return latDeg.T, lonDeg.T #this is a bug??? I had to do this because when using broadcasting the resulting index matrices were rotated

def yx2lc(y=None, x=None, lfac=None, cfac=None, coff=None, loff=None):
    """
        Scaling function that transforms intermediary satellite coordinates expressed in degrees into satellite line column coordinates
        or pixel indices of the  original satellite image.
        The definition is as follows:
            c = COFF + nint(x * 2**-16 * CFAC)
            l = LOFF + nint(y * 2 **-16 * LFAC)

        For details consult http://2014.cgms-info.org/documents/cgms-lrit-hrit-global-specification-%28v2-8-of-30-oct-2013%29.pdf
        section 4.4.4
        TODO: I really need to do some experimentation to understand precisely how are the args derived


        :param y: satellite coordinates for Y axis corresponding to a specific LATLON
        :param x: satellite coordinates for X axis corresponding to a specific LATLON
        :param lfac: line scaling factor,
        :param cfac: column scaling factor
        :param coff: column offset
        :param loff: line offset offset
        :return a tuple representing line coordinates (indices) and column coordinates (indices)
        These coordinates can be used to extract counts form the original hrit data


    """
    if y is None or x is None:
        return None, None

    return np.floor(0.5 + loff+ y*lfac*2**-16).astype(np.uint16), np.floor(0.5 + coff+ x*cfac*2**-16).astype(np.uint16)

def lc2yx(c=None,l=None,loff=None,coff=None,cfac=None,lfac=None):
    """
    Scaling function that transforms satellite line column coordinates or pixel indices of the
    original satellite image into intermediary satellite coordinates expressed in degrees.
        Derived from:
        c = COFF + nint(x * 2**-16 * CFAC)
        l = LOFF + nint(y * 2 **-16 * LFAC)
        to:
        x = c - COFF / 2**-16 * CFAC
    :param c: column coordinates, int
    :param l: line coordinates, int
    :param lfac: line scaling factor
    :param cfac: column scaling factor
    :param coff: column offset
    :param loff: line offset offset
    :return satellite coordinates x,y

    """
    if c is None or l is None:
        return None, None

    x = (c - coff) / ((2**-16) * cfac)
    y = (l - loff) / ((2**-16) * lfac)
    return y,x

def ll2lc(lat=None, lon=None, lon0=None, cfac=None, lfac=None, coff=None, loff=None, nl=None, nc=None):
    """
    Converts lat/lon to raw  line/col. Basically a container forlatlon2yx and yx2lc

    Many JMA HRIT images have geometry registration issues. These issues can be corrected by using information from Header 130 located in channel corresponding segments


    :param lat: latitude, float/int or 1D numpy array of dtype float, int,
    :param lon: longitude, float/int  or 1D numpy array of dtype float, int
    :param lfac: line scaling factor
    :param cfac: column scaling factor
    :param coff: column offset
    :param loff: line offset offset
    :param nl: int, number of lines in the image
    :param nc: int, number of columns in the image
    :return a tuple representing line coordinates (indices) and column coordinates (indices) or a typke of None if the coordinates are outside the nl and nc .
    LC space is bigger than LATLON therefore return coords are checked to be in 0-NL|NC range.
    """

    y, x = ll2yx(lat=lat, lon=lon, lon0=lon0 )
    l, c = yx2lc(y=y, x=x, lfac=lfac, cfac=cfac, coff=coff, loff=loff)

    if np.all(np.logical_and(l>=0, l<nl)) and np.all(np.logical_and(c>=0, c<nc)):
        return l, c
    else:
        return None, None

def lc2ll(c=None, l=None, lon0=None, cfac=None, lfac=None,coff=None, loff=None):
    """
    Converts raw line/col to lat/lon. A container for	lc2yx and yx2ll

    :param c: column coordinates, int or numpy array of ints
    :param l: line coordinates, int or numpy array of ints
    :param lon0: subsatelitte point
    :param lfac: line scaling factor
    :param cfac: column scaling factor
    :param coff: column offset
    :param loff: line offset offset
    :return
    """
    if c is None or l is None:
        return None, None

    y,x = lc2yx(c=c, l=l, loff=loff, coff=coff, cfac=cfac, lfac=lfac)
    return  yx2ll(y=y, x=x, lon0=lon0)


def offset_lc(l=None, c=None, loff=None, coff=None, orig_loff=None, orig_coff=None):
    """
    Offsets original line/col coordinates with offset values from the header 130.

    :param l: line coordinates, int or numpy array of ints,
    :param c: column coordinates, int or numpy array of ints
    :param loff: line offets, numy array of floats, having same size as l
    :param coff: column offets, numpy array of floats having same size as c
    :param orig_loff:, int, float, the default line offset form header 3
    :param orig_coff:, int, float,, the defauylt column offset
    :return:
    """

    return np.floor(l-orig_loff + loff).astype(np.uint16), np.floor(c-orig_coff + coff).astype(np.uint16)

def YX2lc(X=None,Y=None, coff=None, loff=None, resolution=None):
    """
    Transforms Geostationary coordinates X/Y to raw line/col
    http://lists.osgeo.org/pipermail/gdal-dev/2007-November/014954.html
    c = COFF + X / ColumnDirGridStep
    :param X: geostationary MTSAT x axis coordinate
    :param Y: geostationary MTSAT y axis coordinate
    :param loff: line offset
    :param coff: column offset
    :param resolution:
    :return:
    """

    if Y is None or X is None:
        return None, None

    c = np.floor(0.5 + coff + X / resolution).astype(np.uint16)
    l = np.floor(0.5 + abs(loff - Y / resolution)).astype(np.uint16)

    return l, c

def lc2YX(c=None,l=None, coff=None, loff=None, resolution=None):
    """
    Derived DIRECTLY from the linear relationship between the lc coordinates and geostationary coordinates
    Transforms raw LC into geostationary coordinates XY
    derived from http://lists.osgeo.org/pipermail/gdal-dev/2007-November/014954.html
    c = COFF + X / resolution


    :param l: line coordinates, int or numpy array of ints,
    :param c: column coordinates, int or numpy array of ints
    :param loff: line offset
    :param coff: column offset
    :param resolution:
    :return: geostationary  X,Y coordinates in meters
    Note by Jano
        -loff -l  is necessary here  not l-loff because  in geostationary projection Y and X have both negative values
         depending on which quadrant they do fall
    """
    if c is None or l is None:
        return None, None
    try:
        c = np.asarray(c)
        l = np.asarray(l)
    except TypeError:
        pass
    X = (c-coff)*float(resolution) + resolution/2.
    Y = (loff-l)*float(resolution) - resolution/2.
    return Y, X

def yx2YX(y=None, x=None, cfac=None, lfac=None, resolution=None):
    """
    Transforms intermediary satellite coordinates expressed in degrees into geostationary coordinates XY
    http://lists.osgeo.org/pipermail/gdal-dev/2007-November/014954.html
    X = x * CFAC * resolution / (2**16)

    :param y: satellite coordinates for Y axis
    :param x: satellite coordinates for X axis
    :param lfac: line scaling factor
    :param cfac: column scaling factor
    :param resolution:
    :return: geostationary MTSAT YX
    Note by Jano
        -lfac is necessary here because the in geostationary projection Y and X have both negative values
         depending on which quadrant they do fall
         this ahould be actually handled by 2 resoilutions, one for each axis and the Y resolution is NEGATIVE

    """
    if y is None or x is None:
        return None, None

    X = (x * cfac * resolution) / (2**16)
    Y = (y * -lfac * resolution) / (2**16)

    return Y,X

def YX2yx(Y=None, X=None, cfac=None, lfac=None, resolution=None):
    """
    Geostationary  YX to intermediary satellite coordinates expressed in degrees
    http://lists.osgeo.org/pipermail/gdal-dev/2007-November/014954.html
    x = X * (2**16) / (CFAC * resolution)

    :param l: line coordinates, int or numpy array of ints,
    :param c: column coordinates, int or numpy array of ints
    :param lfac: line scaling factor
    :param cfac: column scaling factor
    :param resolution
    :return intermediary satellite coordinates expressed in degrees
    #note by jano
        -lfac is necessary here to account for quadrants as the LC space does not change sign and the geostationary space does

    """
    if Y is None or X is None:
        return None, None

    x = X*(2**16) / (cfac * resolution)
    y = Y*(2**16) / (-lfac * resolution)

    return y, x

def YX2ll(Y=None, X=None, cfac=None, lfac=None, resolution=None, lon0=None):
    """
        Geostationary  YX to latitude longitude, container for two functions
    :param X: geostationary  x axis coordinate
    :param Y: geostationary  y axis coordinate
    :param loff: line offset
    :param coff: column offset
    :param resolution:
    :return: latitude, longitude
    """

    if Y is None or X is None:
        return None, None

    y,x = YX2yx(Y=Y, X=X, cfac=cfac, lfac=lfac, resolution=resolution)
    return  yx2ll(y=y, x=x,lon0=lon0)


def ll2YX(lat=None, lon=None, lon0=None, cfac=None, lfac=None, resolution=None):
    """
    Latitude/Longitude to Geostationary YX
    :param lat: latitude, int,
    :param lon: longitude, int
    :param loff: line offset
    :param coff: column offset
    :param resolution:
    :return: geostationary MTSAT X,Y coordinates in meters
    """

    if lat is None or lon is None:
        return None, None

    y,x = ll2yx(lat=lat, lon=lon, lon0=lon0)
    Y,X = yx2YX(y=y, x=x, cfac=cfac, lfac=lfac, resolution=resolution)

    return Y,X


def reproject_geos_2_latlon(array_in=None, bbox=None, ssp=None, resolution=None):
    """
    Reprojects a geostatioanry  image (2D array) in a spatial context defined by input image
    a sub-satellite point and resolution and output  geographic bounding box.
    :param array_in: 2D numpy array, input image
    :param bbox: instance of bounding_box object, the pyarea in geographic coordinates that will be reprojected from the input image
    :param ssp: number, float, input image sub-satellite point in degrees
    :param resolution: number, float, input image resolution in meters
    :return: 2D numpy array representing the  reprojected input image
    """
    nl, nc = array_in.shape
    nav_dict = get_navigation_dict(resolution=resolution, nl=nl, nc=nc)
    cfac = nav_dict['CFAC']
    lfac = nav_dict['LFAC']
    coff = nav_dict['COFF']
    loff = nav_dict['LOFF']

    # get 1D lats and lons
    lats = bbox.latitudes()

    lons = bbox.longitudes()
    valid_l, valid_c = ll2lc(lat=lats, lon=lons, lon0=ssp, cfac=cfac, lfac=lfac, coff=coff, loff=loff, nl=nl, nc=nc)
    return array_in[valid_l, valid_c]
