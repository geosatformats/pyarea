__author__ = 'jano'
import collections
import gc
import math
import struct
from multiprocessing import cpu_count
import logging
import datetime
import numexpr as ne
import numpy as np
from numpy import nan

from pyarea import utils

import numba

_njit = numba.njit


ne.set_num_threads(cpu_count())
logger = logging.getLogger('nav')

class AreaNav():


    # Code value in AREA files used to designate DMSP navigation
    DMSP = 'DSMP'

    # Code value in AREA files used to designate GMSX (GMS) navigation
    GMSX = 'GMSX'

    # Code value in AREA files used to designate GOES (GOES D-H) navigation
    GOES = 'GOES'

    # Code value in AREA files used to designate GEOS navigation
    GEOS = 'GEOS'

    # Code value in AREA files used to designate GVAR (GOES I-M) navigation
    GVAR = 'GVAR'

    # Code value in AREA files used to designate MOLL (Mollweide) navigation
    MOLL = 'MOLL'

    # Code value in AREA files used to designate MSAT (Meteosat) navigation
    MSAT = 'MSAT'

    # Code value in AREA files used to designate MSGT  navigation
    MSGT = 'MSGT'

    # Code value in AREA files used to designate MSG navigation
    MSG  = 'MSG'

    # Code value in AREA files used to designate POES navigation
    POES =  'POES'

    # Code value in AREA files used to designate RADR (radar) navigation
    RADR = 'RADR'

    # Code value in AREA files used to designate RECT (rectilinear) navigation
    RECT =  'RECT'

    # Code value in AREA files used to designate PS (polar stereographic) navigation
    PS = 'PS'

    # Code value in AREA files used to designate MERC (mercator) navigation
    MERC = 'MERC'

    # Code value in AREA files used to designate TANC (tangent cone) navigation
    TANC =  'TANC'

    # Code value in AREA files used to designate SIN (sinusoidal cone) navigation
    SIN =  'SIN'

    # Code value in AREA files used to designate LAMB (lambert conformal)navigation
    LAMB = 'LAMB'

    # Code value in AREA files used to designate Lat/Lon
    LALO = 'LALO'

    # Code value in AREA files used to designate KALP
    KALP = 'KALP'

    # Code value in AREA files used to designate ABIS
    ABIS = 'ABIS'

    #ABI GOES R
    ABIN = 'ABIN'

    # Code value for specifying Latitude/Longitude transformations
    LL = 123

    # Code value for specifying Cartesian (X/Y) transformations
    XY = 234

    # "Line" index in line/element array
    indexLine=1
    # "Element" index in line/element array
    indexEle=0
    # "Latitude" index in latitude/longitude array
    indexLat=0
    # "Longitude" index in latitude/longitude array
    indexLon=1

    isLineFlipped = False
    lineOffset = 0.0

    # the following are ancillary info set by the set/get
    # public methods Res, Mag, and Start
    resLine = 1.
    resElement = 1
    magLine = 1.
    magElement = 1.
    startLine = 0
    startElement = 0.
    startImageLine = 0.
    startImageElement = 0.
    ##COMMON attrs added by jano purposefully
    PI = math.pi
    DEG = 180. / PI
    RAD = PI / 180.  # degrees to radians conversion pi/180
    STTYPE = 0  # word position of satellite type

    #java project uses makeAreaNav factory static method. I prefer different alternative,
    #to use __new__()
    def __new__(cls, *args, **kwargs):
        if not 'navblock' in kwargs:
            raise RuntimeError('Can not create {} instance without a navigation block'.format( cls.__name__))
        if not 'byte_order' in kwargs:
            raise RuntimeError('Can not create {} instance without knowing byte_order. Please supply "byte_order" argument.'.format( cls.__name__))
        if not 'int_format' in kwargs:
            raise RuntimeError(
                'Can not create {} instance without knowing format. Please supply "int_format" argument.'.format(cls.__name__))

        navblock= kwargs.get('navblock')


        byte_order= kwargs.get('byte_order')
        int_format= kwargs.get('int_format')
        #navtype = struct.unpack('<4s', struct.pack('<i', navblock[0]))[0]
        #navtype = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=byte_order, size=4,format_str='s'), navblock[:4])[0]
        navtype = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=byte_order, size=4, format_str='s'),
                      struct.pack('{byte_order:s}{format_str:s}'.format(byte_order=byte_order, format_str=int_format),
                                  navblock[0]))[0]
        #navtype = .from_bytes(navblock[:4])
        navtype = navtype.decode(errors='replace')

        #evaluate dynamically, does have the advantage of being self updateable
        #nav_classes = [klass.code_string() for name, klass in globals().items() if 'nav' in name.lower() and issubclass(klass, AreaNav)]
        # a concrete subclass is built directly (its __init__ validates the block);
        # only AreaNav itself has to find the right subclass from the nav type
        nav_classes = cls.get_subclasses() if cls is AreaNav else [cls]



        # dynamically create the new class
        for klass in nav_classes:
            if cls is not AreaNav or klass.__name__.startswith(navtype):
                inst = object.__new__(klass)
                inst.navtype = navtype
                inst.byte_order = byte_order
                inst.int_format = int_format
                return  inst



        #raise RuntimeError(f'Navigation for {navtype} data is not implemented. Supported implementations are: {str(nav_classes)}')


    def toLatLon(self, linele=None):
        raise RuntimeError('Abstract method. Should be implemented by subclasses')

    def toLinEle(self, latlon=None):
        raise RuntimeError('Abstract method. Should be implemented by subclasses')

    def areaCoordtoImageCoord_np(self, linele=None):
        c, l = linele
        l1 =  self.lineOffset - l if self.isLineFlipped else l
        l1 = self.startImageLine + (self.resLine * (l1 - self.startLine)) / self.magLine
        c1 = self.startImageElement + (self.resElement * (c - self.startElement)) / self.magElement
        return [l1, c1]

    def areaCoordtoImageCoord(self, linele=None, newvals = None):
        l = len(linele[0])
        if newvals is None:
            newvals = [[nan for i in range(l)], [nan for i in range(l)]]

        for i in  range(l):
            if linele[self.indexLine][i] == linele[self.indexLine][i]:
                # account for flipped coordinates
                line  =  self.lineOffset - linele[self.indexLine][i] if self.isLineFlipped else linele[self.indexLine][i]
                #print 'line %s startImageLine %s startLine %s resLine %s magLine %s' % (line, self.startImageLine, self.startLine, self.resLine, self.magLine)
                newvals[self.indexLine][i] = self.startImageLine + (self.resLine * (line - self.startLine)) / self.magLine

            if linele[self.indexEle][i] == linele[self.indexEle][i]:
                #print 'elem %s startImageElement %s startElement %s resElement %s magElement %s' % (linele[self.indexEle][i], self.startImageElement, self.startElement, self.resElement, self.magElement)
                newvals[self.indexEle][i] = self.startImageElement + (self.resElement * (linele[self.indexEle][i] - self.startElement)) / self.magElement


        return newvals


    def imageCoordToAreaCoord(self, linele=None, newvals = None):
        l = len(linele[0])
        if newvals is None:
            newvals = [[nan for i in range(l)], [nan for i in range(l)]]

        for i in range(l):
            if linele[self.indexLine][i] == linele[self.indexLine][i]:
                newvals[self.indexLine][i] = self.startLine + ( self.magLine * (linele[self.indexLine][i] - self.startImageLine)) / self.resLine
            # account for flipped coordinates
                if self.isLineFlipped:
                    newvals[self.indexLine][i] =  self.lineOffset - newvals[self.indexLine][i]

            if linele[self.indexEle][i] == linele[self.indexEle][i]:
                newvals[self.indexEle][i] = self.startElement + ( self.magElement * (linele[self.indexEle][i] - self.startImageElement)) / self.resElement
        return newvals


    def imageCoordToAreaCoord_np(self, linele=None):
        l, c = linele
        l1 = self.startLine + ( self.magLine * (l - self.startImageLine)) / self.resLine
        if self.isLineFlipped:
            l1 = self.lineOffset - l1
        c1 = self.startElement + ( self.magElement * (c - self.startImageElement)) / self.resElement

        return [l1,c1]

    def canApproximateWithSpline(self):
        return True

    def __str__(self):
        return self.navtype

    def setImageStart(self, start_line, start_elem):
        self.startImageLine = float(start_line)
        self.startImageElement = float(start_elem)

    def setStart(self, start_line, start_elem):
        self.startLine = start_line
        self.startElement = start_elem
    def setRes(self, resLine, resElement):
        self.resLine = resLine
        self.resElement = resElement
    def setMag(self, magLine, magElement):
        self.magLine = magLine
        self.magElement = magElement

    @classmethod
    def get_subclasses(cls):

        subs = set(cls.__subclasses__())
        return subs.union(*(i.get_subclasses() for i in subs))



_GvarParams = collections.namedtuple(
    "_GvarParams",
    "instr ff doff elvmax elvln scnpx scnmax pma rma "
    "b00 b01 b02 b10 b11 b12 b20 b21 b22 xs0 xs1 xs2 aebe2c aebe3c aebe4c q3 east",
)


@_njit
def _gvar_to_latlon(rl, rp, p):
    """Image line/pixel (flat float arrays, GVAR image coordinates) to geodetic lat/lon in degrees.

    This is the GVAR "lpoint" computation of the Operations Ground Equipment interface document, ported from the
    McIDAS ``nvxgvar`` code. NaN in, or a line of sight that misses the Earth, gives NaN.
    """
    n = rl.size
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    for i in range(n):
        line = rl[i]
        pixel = rp[i]
        if not (np.isfinite(line) and np.isfinite(pixel)):
            continue
        # the sounder routines take the sounder resolution into account, so pretend the image is at resolution 1
        if p.instr == 2:
            line = (line + 9.0) / 10.0
            pixel = (pixel + 9.0) / 10.0
        # elevation and scan angles of the line and pixel
        if p.instr == 1:
            alpha0 = p.elvmax - (line - 4.5) * p.elvln
        else:
            alpha0 = p.elvmax - (line - 2.5) * p.elvln
        zeta0 = (pixel - 1.0) * p.scnpx - p.scnmax
        # second order origin offset correction
        alpha = alpha0 - alpha0 * zeta0 * p.doff
        zeta = zeta0 + 0.5 * alpha0 * alpha0 * p.doff
        # trigonometric functions of the angles, corrected for the roll and pitch misalignments
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cz = math.cos(zeta)
        da = alpha - p.pma * sa * (p.ff / cz + math.tan(zeta)) - p.rma * (1.0 - ca / cz)
        dz = zeta + p.ff * p.rma * sa
        cz = math.cos(dz)
        # pointing vector in instrument coordinates, then in earth fixed coordinates
        g0 = math.sin(dz)
        g1 = -cz * math.sin(da)
        g2 = cz * math.cos(da)
        e0 = p.b00 * g0 + p.b01 * g1 + p.b02 * g2
        e1 = p.b10 * g0 + p.b11 * g1 + p.b12 * g2
        e2 = p.b20 * g0 + p.b21 * g1 + p.b22 * g2
        # intersect the pointing vector with the earth surface (a quadratic equation)
        q1 = e0 * e0 + e1 * e1 + p.aebe2c * e2 * e2
        q2 = p.xs0 * e0 + p.xs1 * e1 + p.aebe2c * p.xs2 * e2
        d = q2 * q2 - q1 * p.q3
        if abs(d) < 1e-9:
            d = 0.0
        if d < 0.0:  # points off the earth
            continue
        d = math.sqrt(d)
        h = -(q2 + d) / q1
        u0 = p.xs0 + h * e0
        u1 = p.xs1 + h * e1
        u2 = p.xs2 + h * e2
        # sine of the geocentric latitude, then the geodetic latitude and the longitude
        d1 = u2 / math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        lat[i] = math.atan(p.aebe2c * d1 / math.sqrt(1.0 - d1 * d1)) * 57.29577951308232
        lon[i] = math.atan2(u1, u0) * 57.29577951308232 * p.east
    return lat, lon


@_njit
def _gvar_to_image(lat, lon, p):
    """Geodetic lat/lon in degrees (flat float arrays) to image line/pixel (GVAR image coordinates).

    The inverse of :func:`_gvar_to_latlon` (the "gpoint" routine). NaN where the point is not visible from the
    satellite, or where lat or lon is NaN.
    """
    n = lat.size
    line = np.full(n, np.nan)
    pixel = np.full(n, np.nan)
    for i in range(n):
        if not (abs(lat[i]) <= 90.0 and np.isfinite(lon[i])):
            continue
        rlat = lat[i] * 0.017453292519943295
        rlon = lon[i] * 0.017453292519943295 * p.east
        # sine of the geodetic latitude and of the geocentric latitude
        sing = math.sin(rlat)
        w1 = p.aebe4c * sing * sing
        slat = ((0.375 * w1 - 0.5) * w1 + 1.0) * sing / p.aebe2c
        # local earth radius at the point, and its cartesian coordinates
        w2 = slat * slat
        w1 = p.aebe3c * w2
        w1 = (0.375 * w1 - 0.5) * w1 + 1.0
        u2 = slat * w1
        w2 = w1 * math.sqrt(1.0 - w2)
        u0 = w2 * math.cos(rlon)
        u1 = w2 * math.sin(rlon)
        # pointing vector from the instrument to the point
        f0 = u0 - p.xs0
        f1 = u1 - p.xs1
        f2 = u2 - p.xs2
        w2 = u0 * f0 + u1 * f1 + u2 * f2 * p.aebe2c
        if not w2 <= 0.0:  # not visible
            continue
        # the pointing vector in instrument coordinates
        t0 = p.b00 * f0 + p.b10 * f1 + p.b20 * f2
        t1 = p.b01 * f0 + p.b11 * f1 + p.b21 * f2
        t2 = p.b02 * f0 + p.b12 * f1 + p.b22 * f2
        # scan and elevation angles, corrected for the roll and pitch misalignments
        gam = math.atan(t0 / math.sqrt(t1 * t1 + t2 * t2))
        alf = -math.atan(t1 / t2)
        w1 = math.sin(alf)
        w2 = math.cos(gam)
        alpha1 = alf + p.rma * (1.0 - math.cos(alf) / w2) + p.pma * w1 * (p.doff / w2 + math.tan(gam))
        gam = gam - p.ff * p.rma * w1
        alf = alpha1 + alpha1 * gam * p.doff
        gam = gam - 0.5 * alpha1 * alpha1 * p.doff
        # fractional line and pixel numbers
        tmplin = (p.elvmax - alf) / p.elvln
        tmplin = tmplin + 4.5 if p.instr == 1 else tmplin + 2.5
        tmpele = (p.scnmax + gam) / p.scnpx + 1.0
        # sounder navigation returns resolution 10 values
        if p.instr == 2:
            tmplin = tmplin * 10.0 - 9.0
            tmpele = tmpele * 10.0 - 9.0
        line[i] = tmplin
        pixel[i] = tmpele
    return line, pixel


class GVARNav(AreaNav):

    isEastPositive = True


    NOMORB=42164.365 # nominal radial distance of instrument (km)
    AE=6378.137 # earth equatorial radius (km)
    FER=1.-(6356.7533/AE) # earth flattening coeff = 1-(be/ae)
    AEBE2= 1./(1-FER)**2
    AEBE3= AEBE2-1.
    AEBE4=((1.-FER)**4.)-1.

    xs = [nan]  * 3 # normalized s/c position in ecef coordinates
    #bt= [[nan for i in range(3)], [nan for i in range(3)], [nan for i in range(3)] ]
    bt = np.empty((3,3))
    bt[:] = np.nan
    #print bt
    # ecef to instrument coordinates transformation
    q3 = nan # used in subrtn lpoint
    pitch = roll = yaw = nan # pitch,roll,yaw angles of instrument (rad)
    pma = nan # pitch misalignment of instrument (rad)
    rma = nan # roll misalignment of instrument (rad)

    incmax = [6136, 2805] # number of increments per cycle

    elvmax = [0.220896, 0.22089375] # bounds in elevation (radians)
    scnmax= [0.24544,0.2454375] # bounds in scan angle (radians)
    elvinc = [8.e-6, 17.5e-6] # change in elevation angle per increment (rad)
    scninc = [16.e-6, 35.e-6] # change in scan angle per increment (radians)
    elvln = [28.e-6, 280.e-6] # elevation angle per detector line (radians)
    scnpx = [16.e-6, 280.e-6 ]# scan angle per pixel (radians)
    nsnom = [.220896, .22089375] # north-south center of instrument (=4.5 x incmax x elvinc)

    ewnom = [.24544, .2454375] # east-west center of instrument (=2.5 x incmax x sncinc)

    STTYPE = 0 # position of instrument type
    IDNTFR = 1
    IMCACT = 2 # position of imc active flag
    IYFLIP = 3 # position of yaw flip enabled flag
    REFLON = 5 # position of reference longitude
    REFDIS = 6 # position of reference distance from nominal
    REFLAT = 7 # position of reference latitude
    REFYAW = 8 # position of reference yaw
    RATROL = 9 # position of reference attitude roll
    RATPTC = 10 # position of reference attitude pitch
    RATYAW = 11 # position of reference attitude yaw
    ETIME  = 12 # position of epoch time
    EDTIME = 14 # location of delta from epoch time
    IMCROL = 15 # location of image motion compensation roll
    IMCPTC = 16 # location of image motion compensation pitch
    IMCYAW = 17 # location of image motion compensation yaw

    # ldr1-13: location of longitude delta from reference parameters
    LDR1   = 18
    LDR2   = 19
    LDR3   = 20
    LDR4   = 21
    LDR5   = 22
    LDR6   = 23
    LDR7   = 24
    LDR8   = 25
    LDR9   = 26
    LDR10  = 27
    LDR11  = 28
    LDR12  = 29
    LDR13  = 30

    # rddr1-11: location of radial distance delta from reference parameters
    RDDR1  = 31
    RDDR2  = 32
    RDDR3  = 33
    RDDR4  = 34
    RDDR5  = 35
    RDDR6  = 36
    RDDR7  = 37
    RDDR8  = 38
    RDDR9  = 39
    RDDR10 = 40
    RDDR11 = 41

    # dgl1-9: location of geocentric latitude delta parameters
    DGL1   = 42
    DGL2   = 43
    DGL3   = 44
    DGL4   = 45
    DGL5   = 46
    DGL6   = 47
    DGL7   = 48
    DGL8   = 49
    DGL9   = 50

    # doy1-9: location of orbit yaw delta parameters
    DOY1   = 51
    DOY2   = 52
    DOY3   = 53
    DOY4   = 54
    DOY5   = 55
    DOY6   = 56
    DOY7   = 57
    DOY8   = 58
    DOY9   = 59

    EXPTIM = 61 # exponential start time from epoch
    RAAWDS = 62 # location of start of roll attitude angle words
    PAAWDS = 129 # location of start of pitch attitude angle words
    YAAWDS = 184 # location of start of yaw attitude angle words
    RMAWDS = 257 # location of start of roll misalignment angle words
    PMAWDS = 312 # location of start of pitch misalignment angle words

    IMGDAY = 367 # position of image day value  (yyddd)
    IMGTM  = 368 # position of image time value (hhmmss)
    IMGSND = 369 # location of imager/sounder instrument flag
    ISTIM = 226

    # the following four words were added 5-26-94 to comply w/ the new elug...
    # numbering started at 380 because these same parameters are used
    # in the nav message sent from the ingestor to evx, and we had to
    # start somewhere after the 378 nav parameters

    IOFNC = 379
    IOFEC = 380
    IOFNI = 381
    IOFEI = 382

    MXCDSZ=5*128 # maximum directory block entry size

    OASIZE = 336 # size of gvar orbit and attitude set
    PCOEFS = 117 # start of pitch coefficients
    RMACFS = 227 # start of rma coefficients
    CUTOF1 = 115 # first dividing poin o&a set (128 word sets)
    CUTOF2 = 225 # second dividing poin o&a set (128 word sets)

    IMCFLG = 7 #  bit # in sscan for imc active flag
    FLPFLG = 15 #  bit # in sscan for yaw flip enabled flat
    iflip = None # value of FLPFLG

    # now some variables that are shared among the methods...

    aec = ts = dr = lam = dlat = dyaw = phi = psi = imc = nan
    aebe2c = aebe3c = aebe4c = ferc = nan
    instr = itype = None
    sublat = sublon = nan
    rlat = rlon = gam = alf = nan
    subpoint = [nan] * 2

    RELLST = (  ( 4,  10), ( 13,  63), ( 65,  94),
                 ( 98, 100), (103, 105), (108, 110), (113, 115),
                 (116, 118), (120, 149), (153, 155), (158, 160),
                 (163, 165), (168, 170), (171, 173), (175, 204),
                 (208, 210), (213, 215), (218, 220), (223, 225),
                 (226, 228), (230, 259), (263, 265), (268, 270),
                 (273, 275), (278, 283), (285, 314), (318, 320),
                 (323, 325), (328, 330), (333, 335))

    def __init__(self, navblock=None, auxblock=None, **kwargs):
        """
            doc
        """

        #python is not java but i declare this so I do not lose the track of instance vars

        te  = u = sinu = cosu = nan
        sinoi = cosoi = slat = asc = sinasc = cosasc = nan
        syaw = w = sw = cw = s2w = c2w = nan
        sw1 = cw1 = sw3 = cw3 = nan
        secs = wa = nan
        imgtim = epoch = timex = time50 = r = nan
        #b = [[nan for i in range(3)], [nan for i in range(3)], [nan for i in range(3)] ]
        b = np.empty((3,3))
        b[:] = np.nan
        lit = year = day = nan
        iftok = hour = minuts = nan
        count = offset = loop = nan
        time = [nan] * 2
        rparms = [nan] * self.MXCDSZ
        # rellst two dimensional array of location of float in o&a set

        # initialize the earth radius constants
        self.aec = self.AE
        self.ferc = self.FER
        self.aebe2c = self.AEBE2
        self.aebe3c = float(self.AEBE3)
        self.aebe4c = float(self.AEBE4)


        #navtype = struct.unpack( '<4s', struct.pack('<i', navblock[self.STTYPE]))[0]

        navtype = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=4, format_str='s'), struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=1, format_str=self.int_format), navblock[self.STTYPE]))[0]
        navtype = navtype.decode()

        if  navtype != self.GVAR:
            raise Exception(f'Invalid navigation type {navtype}')


        self.itype = 1
        #is this used somewhere???
        self.flywheel = None
        #convert to float, no Gould bullshit like in C, just int to float, same order of course

        for i in range(self.MXCDSZ):
            #rparms[i] = struct.unpack('<f', struct.pack('<i', navblock[i]))[0]
            rparms[i] = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=1, format_str='f'), struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=1, format_str=self.int_format), navblock[i]))[0]



        #divide floats as per OGE
        for i, e in enumerate(self.RELLST):
            li, ui = e
            offset = 1
            if li > self.CUTOF1: offset = 13
            if li > self.CUTOF2: offset = 31
            for j in range(li, ui):

                if (j ==  13 or j == 60 or (j - 7) % 55 == 0) and j != 7:
                    rparms[j+offset] = navblock[j+offset] / 100.
                else:
                    rparms[j + offset] = navblock[j + offset] / 10000000.

        self.instr = navblock[self.IMGSND]




        # ----------------------------------------------------------
        # new code from kath kelly for sounder nav - 10/27/??
        # because change to elug -- nadir postion is avabile in
        # signal and should be used to compute values instead
        # of making them hardwired...


        self.nadnsc = navblock[self.IOFNC]
        self.nadnsi = navblock[self.IOFNI]
        self.nadewc = navblock[self.IOFEC]
        self.nadewi = navblock[self.IOFEI]

        if self.nadnsc != 0 and self.nadnsi != 0 and self.nadewc != 0 and self.nadewi != 0:
            if self.instr == 1:
                self.elvmax[0] = (self.nadnsc* self.incmax[0]+self.nadnsi)*self.elvinc[0]
            else:
                self.elvmax[1]=( (9-self.nadnsc)*self.incmax[1]-self.nadnsi)*self.elvinc[1]
            self.scnmax[self.instr-1] = (self.nadewc*self.incmax[self.instr-1]+self.nadewi)*self.scninc[self.instr-1]


        # end of new code from kathy kelly
        # ----------------------------------------------------------
        #compute image time and epoch

        #java and c code uses this computation to get to epoch time in minutes
        #this could nevertheless be a source of rounding error
        #but hey, python has all this figured out through timedelta

        year = 1900 + navblock[self.IMGDAY] // 1000
        day = navblock[self.IMGDAY] - navblock[self.IMGDAY] / 1000 * 1000
        hour = int(rparms[self.IMGTM] / 10000)
        minuts = int(rparms[self.IMGTM] / 100 - hour * 100)
        secs = rparms[self.IMGTM] - float(100 * minuts) - float(10000 * hour)





        j = day + 1461 * (year + 4799) / 4 - 3 * ((year + 4899) / 100) / 4 - 2465022

        imgtim = float( j * 1440. + hour * 60. + minuts + (secs / 60.))


        #sdt = '%4d-%02d-%02d %02d:%02d:%0.2f' % (year, month, mday, hour, minuts, secs)
        #self.scan_datetime = datetime.datetime.strptime(sdt, '%Y-%m-%d %H:%M:%S.%f')
        self.scan_datetime = None

        jan_1_1950 = datetime.datetime(1950, 1, 1)
        '''
        my version of imgtime in minutres is not quite the same like the java code so i use that one

        imgtime = utils.yyyddd_hhmmss2datetime(navblock[self.IMGDAY], navblock[self.IMGTM])

        diff = imgtime - jan_1_1950
        imgtim1 = diff.total_seconds()/60.
        '''
        bcdtime = self.bcd2datetime(navblock[self.ETIME], navblock[self.ETIME+1])



        diff1 = bcdtime - jan_1_1950
        epoch1 = diff1.total_seconds()/60.
        '''
        # convert the BCD to binary integer
        #heer the my code and java is identical so i use mine
        #gain this is a port from java, but because the results are identical to my own code
        t0 = 0
        t1 = 0
        power10 = 1
        e0 = navblock[self.ETIME]
        e1 = navblock[self.ETIME + 1]
        for i in range(8):
            t0 = t0 + ( e0 & int('0xf', 16) ) * power10
            t1 = t1 + ( e1 & int('0xf', 16) ) * power10
            e0 = e0 >>  4
            e1 = e1 >>  4
            power10 = power10 * 10
        year = t0 / 10000
        day = int((t0 - (year * 10000)) * 0.1)
        iaa = t0 - (year * 10000)
        iab = (iaa - (day * 10)) * 10
        nbc = t1 / 10000000
        iac = t1 - (nbc * 10000000)
        def1 = t1 - iac
        hour = iab + nbc
        minuts = int(iac * 0.00001)
        s = (t1 - (def1 + (minuts * 100000))) * 0.001;
        j = day + 1461 * (year + 4799) / 4 - 3 * ((year + 4899) / 100) / 4 - 2465022
        epoch = float(j * 1440. + hour * 60. + float(minuts) + (s / 60.))
        print epoch, epoch1
        '''


        self.imc = 0 if ( (navblock[self.IMCACT] & (1 << self.IMCFLG)) != 0 ) else  1
        self.iflip = -1 if ( (navblock[self.IYFLIP] & (1 << self.FLPFLG)) != 0) else 1
        #print self.imc, self.iflip, imgtim
        # print self.imc
        # print self.iflip

        # assign reference values to the subsatellite longitude and
        # latitude, the radial distance and the orbit yaw.
        self.lam = rparms[self.REFLON]

        self.dr = rparms[self.REFDIS]

        self.phi = rparms[self.REFLAT]

        self.psi = rparms[self.REFYAW]


        self.subpoint[0] = rparms[self.REFLAT] / self.RAD
        self.subpoint[1] = rparms[self.REFLON] / self.RAD


        # assign reference values to the attitudes and misalignments
        self.roll = rparms[self.RATROL]
        self.pitch = rparms[self.RATPTC]
        self.yaw = rparms[self.RATYAW]


        self.rma = 0.
        self.pma = 0.
        # if imc is off, compute changes in the instrument orbit
        if self.imc != 0:
            # set reference radial distance, latitude and orbit yaw to zero
            self.dr = 0.
            self.phi = 0.
            self.psi = 0.

            # compute time since epoch (in minutes)
            self.ts = imgtim - epoch
            # computes orbit angle and the related trigonometric functions.
            # earth rotational rate=.729115e-4 (RAD/s)
            w = 0.729115e-4 * 60.0 * self.ts
            sw = math.sin(w)
            cw = math.cos(w)
            sw1 = math.sin(0.927 * w)
            cw1 = math.cos(0.927 * w)
            s2w = math.sin(2. * w)
            c2w = math.cos(2. * w)
            sw3 = math.sin(1.9268 * w)
            cw3 = math.cos(1.9268 * w)

            # computes change in the imc longitude from the reference
            self.lam = self.lam + rparms[self.LDR1] + (rparms[self.LDR2] + rparms[self.LDR3] * w) * w \
                        + (rparms[self.LDR10] * sw1 + rparms[self.LDR11] * cw1 + rparms[self.LDR4] * sw \
                        + rparms[self.LDR5] * cw + rparms[self.LDR6] * s2w + rparms[self.LDR7] * c2w \
                        + rparms[self.LDR8] * sw3 + rparms[self.LDR9] * cw3 + w * (rparms[self.LDR12] * sw + rparms[self.LDR13] * cw)) * 2.

            # computes change in radial distance from the reference (km)
            self.dr = self.dr + rparms[self.RDDR1] + rparms[self.RDDR2] * cw + rparms[self.RDDR3] * sw \
                        + rparms[self.RDDR4] * c2w + rparms[self.RDDR5] * s2w + rparms[self.RDDR6]*cw3 \
                        + rparms[self.RDDR7] * sw3 + rparms[self.RDDR8] * cw1 \
                        + rparms[self.RDDR9] * sw1 + w * (rparms[self.RDDR10] * cw + rparms[self.RDDR11] * sw)
            # computes the sine of the change in the geocentric latitude
            self.dlat = rparms[self.DGL1] + rparms[self.DGL2] * cw + rparms[self.DGL3] * sw \
                        + rparms[self.DGL4] * c2w + rparms[self.DGL5] * s2w + w * (rparms[self.DGL6] * cw \
                        + rparms[self.DGL7] * sw) + rparms[self.DGL8] * cw1 + rparms[self.DGL9] * sw1

            # computes geocentric latitude by using an expansion for arcsine
            self.phi = self.phi + self.dlat * (1. + self.dlat * self.dlat / 6.)

            # computes sine of the change in the orbit yaw
            self.dyaw = rparms[self.DOY1] + rparms[self.DOY2] * sw + rparms[self.DOY3] * cw \
                        + rparms[self.DOY4] * s2w + rparms[self.DOY5] * c2w \
                        + w * (rparms[self.DOY6] * sw + rparms[self.DOY7] * cw) \
                        + rparms[self.DOY8] * sw1 + rparms[self.DOY9] * cw1
            # computes the orbit yaw by using an expansion for arcsine.
            self.psi = self.psi + self.dyaw * (1. + self.dyaw * self.dyaw / 6.)

            # calculation of changes in the instrument orbit ends here

        # conversion of the imc longitude and orbit yaw to the subinstrument
        # longitude and the orbit inclination (ref: goes-pcc-tm-2473, inputs
        # required for earth location and gridding by sps, june 6, 1988)
        slat = math.sin(self.phi)
        syaw = math.sin(self.psi)
        sinoi = slat * slat + syaw * syaw
        cosoi = math.sqrt(1. - sinoi)
        sinoi = math.sqrt(sinoi)

        if slat == 0.0 and syaw == 0.0:
            u = 0.0
        else:
            u = math.atan2(slat, syaw)
        sinu = math.sin(u)
        cosu = math.cos(u)

        # computes longitude of the ascending node
        asc = self.lam - u
        sinasc = math.sin(asc)
        cosasc = math.cos(asc)

        # computes the subinstrument geographic latitude
        self.sublat = math.atan(self.aebe2c * math.tan(self.phi))

        # computes the subinstrument longitude
        self.sublon = asc + math.atan2(cosoi * sinu, cosu)

        # computes the spacecraft to earth fixed coordinates transformation
        # matrix:
        #     (vector in ecef coordinates) = b * (vector in s/c coordinates)

        b[0][1] = -sinasc * sinoi
        b[1][1] = cosasc * sinoi
        b[2][1] = -cosoi
        b[0][2] = -cosasc * cosu + sinasc * sinu * cosoi
        b[1][2] = -sinasc * cosu - cosasc * sinu * cosoi
        b[2][2] = -slat
        b[0][0] = -cosasc * sinu - sinasc * cosu * cosoi
        b[1][0] = -sinasc * sinu + cosasc * cosu * cosoi
        b[2][0] = cosu * sinoi

        # computes the normalized spacecraft position vector in earth fixed
        # coordinates - xs.
        r = (self.NOMORB + self.dr) / self.aec
        self.xs[0] = -b[0][2] * r
        self.xs[1] = -b[1][2] * r
        self.xs[2] = -b[2][2] * r

        # precomputes q3 (used in lpoint funciton (now in navToLatLon() )

        self.q3 = self.xs[0] * self.xs[0] + self.xs[1] * self.xs[1] + self.aebe2c * self.xs[2] * self.xs[2] - 1.0

        # computes the attitudes and misalignments if imc is off
        if self.imc != 0: # computes the solar orbit angle
            wa = rparms[61-1] * self.ts  # computes the difference between current time, ts, and the  # exponential time, iparms(62). note that both times are since epoch.
            te = self.ts - rparms[self.EXPTIM]  # computes roll + roll misalignment
            self.roll = self.roll + self.gatt(self.RAAWDS, rparms, navblock, wa, te)  # computes pitch + pitch misalignment
            self.pitch = self.pitch + self.gatt(self.PAAWDS, rparms, navblock, wa, te)  # computes yaw
            self.yaw = self.yaw + self.gatt(self.YAAWDS, rparms, navblock, wa, te)  # computes roll misalignment
            self.rma = float(self.gatt(self.RMAWDS, rparms, navblock, wa, te))# computes pitch misalignment
            self.pma = float(self.gatt(self.PMAWDS, rparms, navblock, wa, te)) # apply the earth sensor compensation if needed
            self.roll = self.roll + rparms[self.IMCROL]
            self.pitch = self.pitch + rparms[self.IMCPTC]
            self.yaw = self.yaw + rparms[self.IMCYAW]
            # end if (imc...)
        #this needs to be fixed, those params are passed for no reason as they are instance
        self.inst2e(self.roll, self.pitch, self.yaw, b, self.bt)




    def bcd2datetime(self, w1, w2):
        """
            converts a 2word 4 bytes binary string representing a datetime in  BCD format
            into a datetime object

        """
        names = 'year_1000', 'year_100', 'year_10', 'year_1', 'day_100', 'day_10', 'day_1', 'hour_10', 'hour_1', 'min_10', 'min_1', 'sec_10', 'sec_1', 'msec_100', 'msec_10', 'msec_1'
        #w1 and 2 should be  little endian, the code handles big endian so convert conditionally
        if self.byte_order == '<':
            #chars = struct.pack('>i', w1) + struct.pack('>i', w2)
            chars = struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order='>', size=1, format_str=self.int_format), w1) + struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order='>', size=1, format_str=self.int_format), w2)
        else:
            chars = struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order='>', size=1, format_str=self.int_format),w1) + struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order='>', size=1, format_str=self.int_format), w2)
        values = list()
        k = 0
        for i, byte in enumerate(chars):
            #char = ord(byte)  # int values
            for val in (byte >> 4, byte & 15):  # split in 2 nibbles, use shift by 4 and bitewise & 15 to get the values
                if i == 3:  # the most significant bit of the first nibble day_100 is the flywheel. GOD knows what that means
                    #set flywheel
                    mask = 1 << 0
                    self.flywheel = True if (val & mask) else False


                values.append(val)
        #create a dict from names and nibbles
        d = dict(zip(names, values))

        #compose year
        year = d['year_1000'] * 1000 + d['year_100'] * 100 + d['year_10'] * 10 + d['year_1']
        yday = d['day_100'] * 100 + d['day_10'] * 10 + d['day_1']
        month, day = utils.get_month_day(year, yday, True)
        hour = d['hour_10'] * 10 + d['hour_1']
        minute = d['min_10'] * 10 + d['min_1']
        second = d['sec_10'] * 10 + d['sec_1']
        milisecond = d['msec_100'] * 100 + d['msec_10'] * 10 + d['msec_1']
        return datetime.datetime(year, month, day, hour, minute, second, milisecond*1000)


    def gatt(self, k, floatblock, intblock, wa, te):
        """
            k =  starting position of a parameter subset in the real o&a set
            floatblock(mxcdxz) = input o&a parameter set
            intblock(mxcdxz) = input o&a parameter set
            double wa = input solar orbit angle in radians
            double te = input exponential time delay from epoch (minutes)
        """
        # local variables

        #ir, jr = mr = att

        # constant component
        att = floatblock[k + 2]

        #  computes the exponential term
        if te >= 0:
            att = att + floatblock[k] * math.exp(-te / floatblock[k + 1])


        # extracts the number of sinusoids
        ir = float(intblock[k + 3])
        i = int(ir)

        # calculation of sinusoids

        for j in range(1, i + 1):
            att = att + floatblock[k + 2 * j + 2] * math.cos(wa * float(j) + floatblock[k + 2 * j + 3])


        # pointer to the number of monomial sinusoids
        k = k + 34

        # extarct number of monomial sinusoids
        ir = float(intblock[k])
        kkk = int(intblock[k])


        # computes monomial sinusoids

        for l in range(1, kkk+1):

            ll = k + 5 * l  # order of sinusoid
            jr = float(intblock[ll - 4])

            # order of monomial sinusoid
            mr = float(intblock[ll - 3])
            att = att + floatblock[ll - 2] * math.pow((wa - floatblock[ll]), mr) * math.cos(jr * wa + floatblock[ll - 1])


        return att

    def inst2e(self, r, p, y, a, at):
        """
         r =  roll angle in radians
         p = pitch angle in radians
         y = yaw angle in radians
         a(3,3) = spacecraft to ecef coordinates transformation matrix
         at(3,3)= instrument to ecef coordinates transformation matrix
        """
        #rpy = [[nan for i in range(3)], [nan for i in range(3)], [nan for i in range(3)]]
        rpy = np.empty((3,3))
        rpy[:] = np.nan

        # we compute instrument to body coordinates transformation
        # matrix by using a small angle approximation of trigonometric
        # funktions of the roll, pitch and yaw.
        rpy[0][0] = 1. - 0.5 * (p * p + y * y)
        rpy[0][1] = -y
        rpy[0][2] = p
        rpy[1][0] = y + p * r
        rpy[1][1] = 1. - 0.5 * (y * y + r * r)
        rpy[1][2] = -r
        rpy[2][0] = -p + r * y
        rpy[2][1] = r + p * y
        rpy[2][2] = 1. - 0.5 * (p * p + r * r)

        # multiplication of matrices a and rpy


        for i in range(3):
            for j in range(3):
                at[i][j] = a[i][0] * rpy[0][j] + a[i][1] * rpy[1][j] +a[i][2] * rpy[2][j]


    def _kernel_params(self):
        """The scalar constants the numba kernels need, for the instrument (imager or sounder) of this file."""
        k = self.instr - 1
        bt, xs = self.bt, self.xs
        return _GvarParams(
            instr=int(self.instr),
            ff=float(-self.iflip) if self.instr == 2 else float(self.iflip),
            doff=float(self.scnmax[k] - self.ewnom[k]),
            elvmax=float(self.elvmax[k]), elvln=float(self.elvln[k]),
            scnpx=float(self.scnpx[k]), scnmax=float(self.scnmax[k]),
            pma=float(self.pma), rma=float(self.rma),
            b00=float(bt[0][0]), b01=float(bt[0][1]), b02=float(bt[0][2]),
            b10=float(bt[1][0]), b11=float(bt[1][1]), b12=float(bt[1][2]),
            b20=float(bt[2][0]), b21=float(bt[2][1]), b22=float(bt[2][2]),
            xs0=float(xs[0]), xs1=float(xs[1]), xs2=float(xs[2]),
            aebe2c=float(self.aebe2c), aebe3c=float(self.aebe3c), aebe4c=float(self.aebe4c),
            q3=float(self.q3), east=1.0 if self.isEastPositive else -1.0,
        )

    def _area_to_image(self, line, elem):
        """AREA (file) coordinates to GVAR image coordinates."""
        if self.isLineFlipped:
            line = self.lineOffset - line
        return (self.startImageLine + self.resLine * (line - self.startLine) / self.magLine,
                self.startImageElement + self.resElement * (elem - self.startElement) / self.magElement)

    def _image_to_area(self, line, elem):
        """GVAR image coordinates to AREA (file) coordinates."""
        line = self.startLine + self.magLine * (line - self.startImageLine) / self.resLine
        if self.isLineFlipped:
            line = self.lineOffset - line
        return line, self.startElement + self.magElement * (elem - self.startImageElement) / self.resElement

    def to_latlon(self, line, elem):
        """Convert line/element (AREA/file coordinates, any shape, broadcast together) to ``(lat, lon)`` degrees.

        East longitude is positive. NaN where the line of sight misses the Earth, or the input is NaN.
        """
        line, elem = np.broadcast_arrays(np.asarray(line, dtype=float), np.asarray(elem, dtype=float))
        image_line, image_elem = self._area_to_image(line, elem)
        lat, lon = _gvar_to_latlon(np.ascontiguousarray(image_line).ravel(), np.ascontiguousarray(image_elem).ravel(),
                                   self._kernel_params())
        return lat.reshape(line.shape), lon.reshape(line.shape)

    def to_lin_ele(self, lat, lon):
        """Convert lat/lon degrees (east positive, any shape, broadcast together) to ``(line, element)``.

        The result is in AREA (file) coordinates. NaN where the point is not visible from the satellite.
        """
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        image_line, image_elem = _gvar_to_image(np.ascontiguousarray(lat).ravel(), np.ascontiguousarray(lon).ravel(),
                                                self._kernel_params())
        line, elem = self._image_to_area(image_line, image_elem)
        return line.reshape(lat.shape), elem.reshape(lat.shape)

    def toLatLon(self, linele=None):
        """Convert ``[[elements], [lines]]`` (file coordinates) to ``[[lats], [lons]]``."""
        lat, lon = self.to_latlon(linele[self.indexLine], linele[self.indexEle])
        return [lat, lon]

    def toLinEle(self, latlon=None):
        """Convert ``[[lats], [lons]]`` to ``[[elements], [lines]]`` in file coordinates."""
        line, elem = self.to_lin_ele(latlon[self.indexLat], latlon[self.indexLon])
        out = [None, None]
        out[self.indexLine], out[self.indexEle] = line, elem
        return out

    def toLatLon2(self, lines=None, cols=None):
        """Array version of :meth:`toLatLon`: ``lines`` and ``cols`` (broadcast together) to ``(lat, lon)``."""
        return self.to_latlon(lines, cols)

    def toLinEle2(self, lats=None, lons=None):
        """Array version of :meth:`toLinEle`: ``lats`` and ``lons`` to ``(lines, cols)``, NaN where not visible."""
        return self.to_lin_ele(lats, lons)

    def getSubpoint(self):

        return self.sublat * self.DEG, self.sublon * self.DEG


class LALONav(AreaNav):
    def __init__(self, navblock=None, auxblock=None, **kwargs):
        pass


# the nav block of a "GOES" file starts with the 4 ASCII bytes of the nav type code AreaNav already defines
GOES_MAGIC = AreaNav.GOES.encode("ascii")

_RDPDG = np.pi / 180.0

_EQUATORIAL_RADIUS_KM = 6378.137


def _packed_to_double(value):
    """Convert a packed decimal integer to double degrees (or hours).

    Mirrors McIDASUtil.mcPackedIntegerToDouble: SIGN DDD MM SS (or HH MM SS).
    """
    val = abs(int(value))
    dvalue = (val // 10000) + ((val // 100) % 100) / 60.0 + (val % 100) / 3600.0
    return -dvalue if value < 0 else dvalue


def _icon1(yymmdd):
    """YYMMDD -> YYDDD (year/day-of-year), per nvxgoes ICON1."""
    num = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    year = (yymmdd // 10000) % 100
    month = (yymmdd // 100) % 100
    day = yymmdd % 100
    if month < 1 or month > 12:
        month = 1
    julday = day + num[month - 1]
    if year % 4 == 0 and month > 2:
        julday += 1
    return 1000 * year + julday


def _leapyr(iy):
    return 366 - ((iy % 4) + 3) // 4


def _mc_daytime_to_secs(yrday, hms):
    """Convert a YYDDD day and HHMMSS time to seconds since the epoch.

    Matches McIDASUtil.mcDayTimeToSecs closely enough for the *difference*
    used in timdif() below (which is all this port needs -- it's only ever
    used as raha's time delta).
    """
    year = yrday // 1000
    year = 1900 + year if year >= 30 else 2000 + year
    day = yrday % 1000
    # days since a fixed reference (Jan 1 of `year`), matching Java's
    # Calendar-based day-of-year approach for the purpose of subtraction
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day - 1)
    seconds = _packed_to_double(hms) * 3600.0
    epoch = datetime.datetime(1970, 1, 1)
    return (dt - epoch).total_seconds() + seconds


def _timdif(yrday1, hms1, yrday2, hms2):
    return (_mc_daytime_to_secs(yrday2, hms2) - _mc_daytime_to_secs(yrday1, hms1)) / 60.0


def _check_magic(iarr, endian):
    expected = int.from_bytes(GOES_MAGIC, "big" if endian == ">" else "little")
    got = int(iarr[0]) & 0xFFFFFFFF
    if got != expected:
        raise ValueError(
            f"Not a GOES nav block (word 1 = {iarr[0]!r}, expected "
            f"{expected!r} for endian={endian!r}; if you're sure the "
            f"raw bytes are b'GOES', this is probably an endian "
            f"mismatch -- pass the same endian PregvarAreaFile detected)"
        )


# Scalar constants handed to _to_latlon_kernel() as one immutable bundle
# (numba treats a namedtuple of floats as a plain struct), instead of a
# ~45-argument call. Built by GOESNav._kernel_params().
_KernelParams = collections.namedtuple(
    "_KernelParams",
    [
        # scan / image geometry
        "numsen", "tmpscl", "pictim", "piclin", "radlin", "radele",
        "picele", "gamma", "gamdot",
        # spin-axis rotation matrix (b..) and attitude matrix (rotm..)
        "b11", "b12", "b13", "b21", "b22", "b23", "b31", "b32", "b33",
        "rotm11", "rotm13", "rotm21", "rotm23", "rotm31", "rotm33",
        # earth model and earth-rotation terms
        "bsq_over_asq", "asq", "bsq", "emega", "xref", "rdpdg",
        # Kepler orbit (see GOESNav._satvec_init())
        "tdife", "xmmc", "oeccen", "srome2", "epsiln",
        "px", "py", "pz", "qx", "qy", "qz",
    ],
)


def _double_to_packed(dvalue):
    """Convert double degrees (or hours) to a packed decimal integer."""
    dval = abs(dvalue)
    j = int(3600.0 * dval + 0.5)
    value = 10000 * (j // 3600) + 100 * ((j // 60) % 60) + j % 60
    return -value if dvalue < 0.0 else value


def _anomaly_hours(semima, oeccen, xmeana):
    """Return the time in hours that the orbit takes to reach the mean anomaly ``xmeana`` (degrees)."""
    re = 6378.388
    gracon = 0.07436574
    axmmc = gracon * (np.sqrt(re / semima)) ** 3
    xmanom = (np.pi / 180.0) * xmeana
    return (xmanom - oeccen * np.sin(xmanom)) / (60.0 * axmmc)


def _day_rollover(time):
    """Wrap a time in hours into [0, 24) and return it with the whole-day shift that was applied."""
    if time > 48.0:
        return time - 48.0, 2
    if time > 24.0:
        return time - 24.0, 1
    if time < -24.0:
        return time + 48.0, -2
    if time < 0.0:
        return time + 24.0, -1
    return time, 0


def _add_days(ietimy, iday):
    """Shift a YYDDD epoch by ``iday`` days, following nvxgoes' year rollover."""
    jyear = (ietimy // 1000) % 100 + 1000
    jday = ietimy % 1000 + iday
    if jday < 1:
        jyear -= 1
        jday += _leapyr(jyear)
    elif jday > _leapyr(jyear):
        jday += _leapyr(jyear)
        jyear += 1
    return 1000 * (jyear % 100) + jday


def _epoch(ietimy, ietimh, anomaly_hours):
    """Move the orbit epoch (YYDDD, HHMMSS) back to perigee, ``anomaly_hours`` earlier."""
    time, iday = _day_rollover(_packed_to_double(ietimh) - anomaly_hours)
    if iday != 0:
        ietimy = _add_days(ietimy, iday)
    return ietimy, _double_to_packed(time)


def _validate_nav_words(iarr):
    """Raise ValueError if the nav block lacks usable orbit, pointing or spin parameters."""
    problems = (
        ((iarr[6:12] <= 0).all(),
         "Invalid orbital parameters (all <= 0) -- this file's nav block has no real orbital elements."),
        (iarr[4] == 0, "Invalid orbit type"),
        (not iarr[12:15].any(), "Invalid ascension/declination parameters"),
        (iarr[15] == 0, "Invalid spin period"),
    )
    for failed, message in problems:
        if failed:
            raise ValueError(message)


def _has_beta_scan_times(iss):
    """Tell whether satellite source ``iss`` records the first/last scan line times in its nav block."""
    return iss > 25 or iss == 12


@_njit
def _kepler_position(samtim, p):
    """Return the satellite position (satx, saty, satz) at time ``samtim``.

    Kepler's equation is solved by fixed-point iteration (see GOESNav._satvec()).
    """
    xmanom = p.xmmc * (p.tdife + samtim * 60.0)
    ecanm1 = xmanom
    ecanom = ecanm1
    for _ in range(20):
        ecanom = xmanom + p.oeccen * math.sin(ecanm1)
        if abs(ecanom - ecanm1) < p.epsiln:
            break
        ecanm1 = ecanom
    xomega = math.cos(ecanom) - p.oeccen
    yomega = p.srome2 * math.sin(ecanom)
    return (
        xomega * p.px + yomega * p.qx,
        xomega * p.py + yomega * p.qy,
        xomega * p.pz + yomega * p.qz,
    )


@_njit
def _scan_line_state(parlin, p):
    """Return everything that depends only on the scan-line group ``parlin``.

    That is the sample time, the satellite position, the attitude rotation
    angle and the cos/sin of the earth-rotation angle.
    """
    samtim = p.tmpscl * parlin + p.pictim
    sat = _kepler_position(samtim, p)
    xcor = p.b11 * sat[0] + p.b12 * sat[1] + p.b13 * sat[2]
    ycor = p.b21 * sat[0] + p.b22 * sat[1] + p.b23 * sat[2]
    rot = math.atan2(ycor, xcor) + math.pi
    angle = p.emega * samtim + p.xref
    return samtim, sat, rot, math.cos(angle), math.sin(angle)


@_njit
def _line_of_sight(ylin, yele, p):
    """Return the line-of-sight vector (elo, emo, eno) for the scan angles ``ylin`` / ``yele``."""
    coslin, sinlin = math.cos(ylin), math.sin(ylin)
    sinele, cosele = math.sin(yele), math.cos(yele)

    eli = p.rotm11 * coslin - p.rotm13 * sinlin
    emi = p.rotm21 * coslin - p.rotm23 * sinlin
    eni = p.rotm31 * coslin - p.rotm33 * sinlin
    temp = eli
    eli = cosele * eli + sinele * emi
    emi = -sinele * temp + cosele * emi

    return (
        p.b11 * eli + p.b21 * emi + p.b31 * eni,
        p.b12 * eli + p.b22 * emi + p.b32 * eni,
        p.b13 * eli + p.b23 * emi + p.b33 * eni,
    )


@_njit
def _intersect_earth(sat, los, p):
    """Intersect a line of sight with the earth ellipsoid.

    Returns ``(hit, point)``; ``hit`` is False (and ``point`` meaningless)
    where the line of sight misses the earth.
    """
    satx, saty, satz = sat
    elo, emo, eno = los
    onemsq = 1.0 - p.bsq_over_asq
    aq = p.bsq_over_asq + onemsq * eno * eno
    bq = 2.0 * ((elo * satx + emo * saty) * p.bsq_over_asq + eno * satz)
    cq = (satx * satx + saty * saty) * p.bsq_over_asq + satz * satz - p.bsq
    rad = bq * bq - 4.0 * aq * cq
    if rad < 1.0:
        return False, (0.0, 0.0, 0.0)
    s = -(bq + math.sqrt(rad)) / (2.0 * aq)
    return True, (satx + elo * s, saty + emo * s, satz + eno * s)


@_njit
def _geodetic_latlon(point, ct, st, p):
    """Rotate an earth-surface point into the earth-fixed frame and return (lat, lon), East positive."""
    x, y, z = point
    x1 = ct * x + st * y
    y1 = -st * x + ct * y
    if x1 == 0.0 and y1 == 0.0:
        return np.nan, np.nan
    a = math.atan(z / math.sqrt(x1 * x1 + y1 * y1))
    lat = math.atan2(p.asq * math.sin(a), p.bsq * math.cos(a)) / p.rdpdg
    lon = math.atan2(y1, x1) / p.rdpdg
    return lat, lon


@_njit
def _to_latlon_kernel(xlin, xele, p):
    """Convert flattened line/element image coordinates to (lat, lon) in degrees.

    Longitude is East positive.

    The satellite position and the quantities derived only from it
    (earth-rotation angle, attitude rotation) depend on the scan-line
    group (`parlin`) alone, so they are recomputed only when `parlin`
    changes from one pixel to the next. For a (line, element) grid
    flattened in C order -- lines varying slowest -- that is once per
    line group instead of once per pixel; any other ordering is still
    correct, just without the saving.
    """
    n = xlin.shape[0]
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)

    have_line = False
    last_parlin = 0.0
    samtim = rot = ct = st = 0.0
    sat = (0.0, 0.0, 0.0)

    for i in range(n):
        parlin = (np.rint(xlin[i]) - 1.0) // p.numsen + 1.0
        if (not have_line) or parlin != last_parlin:
            have_line = True
            last_parlin = parlin
            samtim, sat, rot, ct, st = _scan_line_state(parlin, p)

        ylin = (xlin[i] - p.piclin) * p.radlin
        yele = (xele[i] - p.picele + p.gamma + p.gamdot * samtim) * p.radele - rot
        hit, point = _intersect_earth(sat, _line_of_sight(ylin, yele, p), p)
        if hit:
            lat[i], lon[i] = _geodetic_latlon(point, ct, st, p)
        else:
            lat[i] = np.nan
            lon[i] = np.nan
    return lat, lon



class GOESNav(AreaNav):
    """Navigation for pre-GVAR SMS/GOES-1..7 VISSR/VAS AREA files (nav type "GOES").

    Port of edu.wisc.ssec.mcidas.GOESnav (itself a port of the McIDAS-X
    nvxgoes codicil), with the per-pixel earth location compiled by numba
    when it is installed. The image-coordinate setup is the usual AreaNav
    one: call :meth:`setImageStart`, :meth:`setRes`, :meth:`setStart` and
    :meth:`setMag` (``AreaFile.navigation`` does this for you).

    ``toLatLon``/``toLinEle`` take and return ``[[elements], [lines]]`` /
    ``[[lats], [lons]]`` lists like the other navigation classes;
    :meth:`to_latlon` and :meth:`to_lin_ele` are the array versions.
    """

    isEastPositive = True

    def __init__(self, navblock=None, auxblock=None, **kwargs):
        """Set up the navigation from the nav block words.

        Requires a nav block with real, populated orbital elements.

        Args:
            navblock: Raw nav-block words as signed 32-bit ints (0-based;
                ``navblock[0]`` is the 4 ASCII bytes ``GOES``).
            auxblock: Unused, kept for the AreaNav interface.
            **kwargs: ``byte_order`` ('>' or '<') and ``int_format`` as
                passed by :class:`AreaNav`; the byte order is needed
                because the ``GOES`` magic word is 4 ASCII bytes
                reinterpreted as an int32 in the file's byte order.
        """
        iarr = np.asarray(navblock, dtype=np.int64)
        _check_magic(iarr, self.byte_order)
        _validate_nav_words(iarr)

        self.navday = int(iarr[1]) % 100000
        self._set_constants()
        self._set_orbit(iarr)
        self._set_scan_geometry(iarr)
        self._set_attitude(iarr)
        self._set_spin_axis(iarr)
        self._set_timing(iarr)
        self._satvec_init()

    def _set_constants(self):
        """Set the earth model and earth-rotation constants."""
        self.emega = 0.26251617
        self.ab = 40546851.22
        self.asq = 40683833.48
        self.bsq = 40410330.18
        self.r = 6371.221
        self.rdpdg = 1.745329252e-02
        self.bsq_over_asq = self.bsq / self.asq
        self.rsq = self.r ** 2

    def _set_orbit(self, iarr):
        """Set the Kepler orbital elements and the orbit epoch from the nav block."""
        ietimy = _icon1(int(iarr[4]))
        ietimh = 100 * (int(iarr[5]) // 100) + round(0.6 * (int(iarr[5]) % 100))
        self.semima = iarr[6] / 100.0
        self.oeccen = iarr[7] / 1000000.0
        self.orbinc = iarr[8] / 1000.0
        xmeana = iarr[9] / 1000.0
        self.perhel = iarr[10] / 1000.0
        self.asnode = iarr[11] / 1000.0
        self.ietimy, self.ietimh = _epoch(ietimy, ietimh, _anomaly_hours(self.semima, self.oeccen, xmeana))

    def _set_scan_geometry(self, iarr):
        """Set the spin-scan geometry: picture center, spin rate and angles per line/element."""
        piclin = float(iarr[14])
        if iarr[14] >= 1000000:
            piclin = piclin / 10000.0
        self.piclin = piclin

        spinra = iarr[15] / 1000.0
        if spinra < 300.0:
            spinra = 60000.0 / spinra
        self.tmpscl = spinra / 3600000.0

        deglin = _packed_to_double(int(iarr[16]))
        lintot = int(iarr[17])
        degele = _packed_to_double(int(iarr[18]))
        ieltot = int(iarr[19])
        self.numsen = max((lintot // 100000) % 100, 1)
        totlin = self.numsen * (lintot % 100000)
        self.radlin = self.rdpdg * deglin / (totlin - 1.0)
        self.radele = self.rdpdg * degele / (ieltot - 1.0)
        self.picele = (1.0 + ieltot) / 2.0

    def _set_attitude(self, iarr):
        """Set the attitude rotation matrix from pitch, yaw, roll and skew."""
        cpitch = self.rdpdg * _packed_to_double(int(iarr[20]))
        cyaw = self.rdpdg * _packed_to_double(int(iarr[21]))
        croll = self.rdpdg * _packed_to_double(int(iarr[22]))
        skew = 0.0 if utils.is_mcmissing(iarr[28]) else iarr[28] / 100000.0

        pskew = np.arctan2(skew, self.radlin / self.radele)
        stp, ctp = np.sin(cpitch), np.cos(cpitch)
        sty, cty = np.sin(cyaw - pskew), np.cos(cyaw - pskew)
        str_, ctr = np.sin(croll), np.cos(croll)
        self.rotm11 = ctr * ctp
        self.rotm13 = sty * str_ * ctp + cty * stp
        self.rotm21 = -str_
        self.rotm23 = sty * ctr
        self.rotm31 = -ctr * stp
        self.rotm33 = cty * ctp - sty * str_ * stp
        self.rfact = self.rotm31 ** 2 + self.rotm33 ** 2
        self.roasin = np.arctan2(self.rotm31, self.rotm33)

    def _set_spin_axis(self, iarr):
        """Set the spin-axis rotation matrix from the declination and right ascension."""
        dec = _packed_to_double(int(iarr[12])) * self.rdpdg
        sindec, cosdec = np.sin(dec), np.cos(dec)
        ras = _packed_to_double(int(iarr[13])) * self.rdpdg
        sinras, cosras = np.sin(ras), np.cos(ras)
        self.b11 = -sinras
        self.b12 = cosras
        self.b13 = 0.0
        self.b21 = -sindec * cosras
        self.b22 = -sindec * sinras
        self.b23 = cosdec
        self.b31 = cosdec * cosras
        self.b32 = cosdec * sinras
        self.b33 = sindec

    def _set_timing(self, iarr):
        """Set the earth-rotation reference, the picture time and the scan line times."""
        raha = _timdif(74001, 0, self.navday, 0) * 1.00273791 / 4.0 + 100.26467
        self.xref = (raha % 360.0) * self.rdpdg
        self.pictim = _packed_to_double(int(iarr[2]))
        self.gamma = float(iarr[38]) / 100.0
        self.gamdot = float(iarr[39]) / 100.0
        self._set_scan_times(iarr)

    def _set_scan_times(self, iarr):
        """Set the first/last scan line numbers and times."""
        if iarr[30] > 0 and _has_beta_scan_times(int(iarr[1]) // 100000):
            self.scan1 = float(iarr[30])
            self.time1 = _packed_to_double(int(iarr[31]))
            self.scan2 = float(iarr[34])
            self.time2 = _packed_to_double(int(iarr[35]))
        else:
            self.scan1 = 1.0
            self.time1 = self.pictim
            self.scan2 = float(int(iarr[17]) % 100000)
            self.time2 = self.time1 + self.scan2 * self.tmpscl

    def _satvec_init(self):
        """Precompute the orbital-plane basis vectors used by _satvec().

        Only called for the normal Kepler-propagated constructor.
        """
        rdpdg = np.pi / 180.0
        re = 6378.388
        gracon = 0.07436574

        o = rdpdg * self.orbinc
        p = rdpdg * self.perhel
        a = rdpdg * self.asnode
        so, co = np.sin(o), np.cos(o)
        sp, cp = np.sin(p) * self.semima, np.cos(p) * self.semima
        sa, ca = np.sin(a), np.cos(a)
        self._px = cp * ca - sp * sa * co
        self._py = cp * sa + sp * ca * co
        self._pz = sp * so
        self._qx = -sp * ca - cp * sa * co
        self._qy = -sp * sa + cp * ca * co
        self._qz = cp * so
        self._srome2 = np.sqrt(1.0 - self.oeccen) * np.sqrt(1.0 + self.oeccen)
        self._xmmc = gracon * re * np.sqrt(re / self.semima) / self.semima

        iey = (self.ietimy // 1000) % 100
        ied = self.ietimy % 1000
        iefac = (iey - 1) // 4 + 1
        de = 365 * (iey - 1) + iefac + ied - 1
        te = 1440.0 * de + 60.0 * _packed_to_double(self.ietimh)

        inavy = (self.navday // 1000) % 100
        inavd = self.navday % 1000
        infac = (inavy - 1) // 4 + 1
        dnav = 365 * (inavy - 1) + infac + inavd - 1

        self._tdife = dnav * 1440.0 - te
        self._epsiln = 1.0e-8

    def _satvec(self, samtim):
        """Return the satellite position (x, y, z) at time ``samtim``.

        ``samtim`` may be a numpy array; the result is ECI-ish arrays
        (Earth-center-to-satellite, in the pre-earth-rotation frame that
        to_latlon()/get_subpoint() then rotate by (emega*samtim+xref)).

        Kepler's-equation solve from real orbital elements (fixed-point
        iteration, matching the Java's plain loop).
        """
        samtim = np.asarray(samtim, dtype=np.float64)

        timsam = samtim * 60.0
        diftim = self._tdife + timsam
        xmanom = self._xmmc * diftim
        ecanm1 = xmanom.copy()
        ecanom = ecanm1
        for _ in range(20):
            ecanom = xmanom + self.oeccen * np.sin(ecanm1)
            if np.all(np.abs(ecanom - ecanm1) < self._epsiln):
                ecanm1 = ecanom
                break
            ecanm1 = ecanom
        xomega = np.cos(ecanom) - self.oeccen
        yomega = self._srome2 * np.sin(ecanom)
        x = xomega * self._px + yomega * self._qx
        y = xomega * self._py + yomega * self._qy
        z = xomega * self._pz + yomega * self._qz
        return x, y, z

    def _nxyzll(self, x, y, z):
        """Convert earth-fixed x, y, z to geodetic (lat, lon) in degrees, with lon positive West."""
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.arctan(z / np.sqrt(x * x + y * y))
            lat = np.arctan2(self.asq * np.sin(a), self.bsq * np.cos(a)) / self.rdpdg
            lon = -np.arctan2(y, x) / self.rdpdg
        bad = (x == 0.0) & (y == 0.0)
        lat = np.where(bad, np.nan, lat)
        lon = np.where(bad, np.nan, lon)
        return lat, lon

    def _area_to_image(self, line, elem):
        """AREAnav.areaCoordToImageCoord, unflipped case."""
        img_line = self.startImageLine + (self.resLine * (line - self.startLine)) / self.magLine
        img_elem = self.startImageElement + (self.resElement * (elem - self.startElement)) / self.magElement
        return img_line, img_elem

    def _image_to_area(self, line, elem):
        """AREAnav.imageCoordToAreaCoord, unflipped case."""
        area_line = self.startLine + (self.magLine * (line - self.startImageLine)) / self.resLine
        area_elem = self.startElement + (self.magElement * (elem - self.startImageElement)) / self.resElement
        return area_line, area_elem

    def _kernel_params(self):
        """Bundle the scalar constants _to_latlon_kernel() needs."""
        return _KernelParams(
            numsen=float(self.numsen), tmpscl=float(self.tmpscl),
            pictim=float(self.pictim), piclin=float(self.piclin),
            radlin=float(self.radlin), radele=float(self.radele),
            picele=float(self.picele), gamma=float(self.gamma),
            gamdot=float(self.gamdot),
            b11=float(self.b11), b12=float(self.b12), b13=float(self.b13),
            b21=float(self.b21), b22=float(self.b22), b23=float(self.b23),
            b31=float(self.b31), b32=float(self.b32), b33=float(self.b33),
            rotm11=float(self.rotm11), rotm13=float(self.rotm13),
            rotm21=float(self.rotm21), rotm23=float(self.rotm23),
            rotm31=float(self.rotm31), rotm33=float(self.rotm33),
            bsq_over_asq=float(self.bsq_over_asq), asq=float(self.asq),
            bsq=float(self.bsq), emega=float(self.emega),
            xref=float(self.xref), rdpdg=float(self.rdpdg),
            tdife=float(self._tdife), xmmc=float(self._xmmc),
            oeccen=float(self.oeccen), srome2=float(self._srome2),
            epsiln=float(self._epsiln),
            px=float(self._px), py=float(self._py), pz=float(self._pz),
            qx=float(self._qx), qy=float(self._qy), qz=float(self._qz),
        )

    def to_latlon(self, line, elem):
        """Convert line/element (AREA/file coordinates) to (lat, lon) in degrees.

        East positive / North positive. NaN where the line of sight misses
        the earth ellipsoid. Accepts any broadcastable pair of arrays and
        returns arrays of the broadcast shape.
        """
        line, elem = np.broadcast_arrays(
            np.asarray(line, dtype=np.float64),
            np.asarray(elem, dtype=np.float64),
        )
        shape = line.shape
        xlin, xele = self._area_to_image(line, elem)
        lat, lon = _to_latlon_kernel(
            np.ascontiguousarray(xlin).ravel(),
            np.ascontiguousarray(xele).ravel(),
            self._kernel_params(),
        )
        return lat.reshape(shape), lon.reshape(shape)

    def get_subpoint(self):
        """Return the satellite nadir (lat, lon) in degrees, east positive.

        Evaluated at time `time1` (nominal image start time, or first-beta
        scan time for VAS-era files). Used by get_nadir_sampling() to center
        the uniform-sampling grid, and a cheap cross-check of the orbital
        solution.
        """
        samtim = self.time1
        satx, saty, satz = self._satvec(np.asarray(samtim))
        ct = np.cos(self.emega * samtim + self.xref)
        st = np.sin(self.emega * samtim + self.xref)
        x1 = ct * satx + st * saty
        y1 = -st * satx + ct * saty
        lat, lon = self._nxyzll(x1, y1, satz)
        return float(lat), float(-lon)  # East positive

    def get_nadir_sampling(self):
        """Return the sub-satellite point, altitude and nadir pixel spacing.

        The spacing is that of the AREA grid this object was built for.

        Returns a dict with ``sub_lon`` / ``sub_lat`` (degrees, east/north
        positive), ``altitude`` (metres above the equatorial radius, at
        the nominal image time) and ``line_step`` / ``elem_step`` (metres
        between neighbouring AREA lines / elements at nadir). The steps
        come from the scan angle per full-resolution line/element
        (``radlin`` / ``radele``) scaled by the AREA directory's
        line/element resolution factors.
        """
        sub_lat, sub_lon = self.get_subpoint()
        satx, saty, satz = self._satvec(np.asarray(self.time1))
        altitude_m = (float(np.sqrt(satx ** 2 + saty ** 2 + satz ** 2)) - _EQUATORIAL_RADIUS_KM) * 1000.0
        return {
            "sub_lon": sub_lon,
            "sub_lat": sub_lat,
            "altitude": altitude_m,
            "line_step": altitude_m * self.radlin * self.resLine / self.magLine,
            "elem_step": altitude_m * self.radele * self.resElement / self.magElement,
        }

    def getSubpoint(self):
        """Return the satellite nadir ``(lat, lon)`` in degrees (AreaNav naming)."""
        return self.get_subpoint()

    def toLatLon(self, linele=None):
        """Convert ``[[elements], [lines]]`` (file coordinates) to ``[[lats], [lons]]``."""
        lat, lon = self.to_latlon(np.asarray(linele[self.indexLine], dtype=float),
                                  np.asarray(linele[self.indexEle], dtype=float))
        return [lat, lon]

    def _nllxyz(self, lat, lon):
        """Lat/lon in degrees (north and west positive) to earth-centred x, y, z in km (Daly, 1978)."""
        ylat1 = np.arctan2(self.bsq * np.sin(self.rdpdg * lat), self.asq * np.cos(self.rdpdg * lat))
        ylon = -self.rdpdg * lon
        snlt, cslt = np.sin(ylat1), np.cos(ylat1)
        tnlt = (snlt / cslt) ** 2
        r = self.ab * np.sqrt((1.0 + tnlt) / (self.bsq + self.asq * tnlt))
        return r * cslt * np.cos(ylon), r * cslt * np.sin(ylon), r * snlt

    def to_lin_ele(self, lat, lon):
        """Convert lat/lon (degrees, east and north positive) to ``(line, element)`` file coordinates.

        The line is always returned; the element is NaN where the point
        is not visible from the satellite, as in the Java original.
        """
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
        valid = np.abs(lat) <= 90.0
        x1, y1, z = self._nllxyz(np.where(valid, lat, 0.0), -np.where(valid, lon, 0.0))

        samtim = self.time1
        oldlin = None
        for i in range(2):
            xsat, ysat, zsat = self._satvec(samtim)
            xht = np.sqrt(xsat ** 2 + ysat ** 2 + zsat ** 2)
            ct, st = np.cos(self.emega * samtim + self.xref), np.sin(self.emega * samtim + self.xref)
            x = ct * x1 - st * y1
            y = st * x1 + ct * y1
            vcste1, vcste2, vcste3 = x - xsat, y - ysat, z - zsat
            vcses3 = self.b31 * vcste1 + self.b32 * vcste2 + self.b33 * vcste3
            znorm = np.sqrt(vcste1 ** 2 + vcste2 ** 2 + vcste3 ** 2)
            x3 = vcses3 / znorm
            umv = np.arctan2(x3, np.sqrt(self.rfact - x3 ** 2)) - self.roasin
            xlin = self.piclin - umv / self.radlin
            if i == 0:
                samtim = self.time2
                oldlin = xlin

        scnnum = ((oldlin + xlin) / 2.0 - 1.0) / self.numsen
        scnfrc = (scnnum - self.scan1) / (self.scan2 - self.scan1)
        xlin = oldlin + scnfrc * (xlin - oldlin)
        samtim = self.time1 + self.tmpscl * (scnnum - self.scan1)
        xsat, ysat, zsat = self._satvec(samtim)
        cosa = x * xsat + y * ysat + z * zsat
        ctst = 0.0001 * self.r * xht + self.rsq

        xsats1 = self.b11 * xsat + self.b12 * ysat + self.b13 * zsat
        ysats2 = self.b21 * xsat + self.b22 * ysat + self.b23 * zsat
        ct, st = np.cos(self.emega * samtim + self.xref), np.sin(self.emega * samtim + self.xref)
        x = ct * x1 - st * y1
        y = st * x1 + ct * y1
        vcste1, vcste2, vcste3 = x - xsat, y - ysat, z - zsat
        vcses1 = self.b11 * vcste1 + self.b12 * vcste2 + self.b13 * vcste3
        vcses2 = self.b21 * vcste1 + self.b22 * vcste2 + self.b23 * vcste3
        vcses3 = self.b31 * vcste1 + self.b32 * vcste2 + self.b33 * vcste3
        xnorm = np.sqrt(vcste1 ** 2 + vcste2 ** 2 + vcste3 ** 2 - vcses3 ** 2)
        ynorm = np.sqrt(xsats1 ** 2 + ysats2 ** 2)
        znorm = np.sqrt(vcste1 ** 2 + vcste2 ** 2 + vcste3 ** 2)
        x3 = vcses3 / znorm
        umv = np.arctan2(x3, np.sqrt(self.rfact - x3 ** 2)) - self.roasin
        u = self.rotm11 * np.cos(umv) + self.rotm13 * np.sin(umv)
        v = self.rotm21 * np.cos(umv) + self.rotm23 * np.sin(umv)
        with np.errstate(invalid="ignore", divide="ignore"):
            xele = self.picele + np.arcsin((xsats1 * vcses2 - ysats2 * vcses1) / (xnorm * ynorm)) / self.radele
        xele = xele + np.arctan2(v, u) / self.radele - self.gamma - self.gamdot * samtim

        xlin = np.where(valid, xlin, np.nan)
        xele = np.where(valid & (cosa >= ctst), xele, np.nan)
        return self._image_to_area(xlin, xele)

    def toLinEle(self, latlon=None):
        """Convert ``[[lats], [lons]]`` to ``[[elements], [lines]]`` file coordinates."""
        line, elem = self.to_lin_ele(np.asarray(latlon[self.indexLat], dtype=float),
                                     np.asarray(latlon[self.indexLon], dtype=float))
        out = [None, None]
        out[self.indexLine], out[self.indexEle] = line, elem
        return out


class RECTNav(AreaNav):
    labels =  { 	1:'test',
                    2: 'a particular image row number',
                    3: 'latitude corresponding to word 2, degrees x10000',
                    4: 'a particular image column number',
                    5: 'longitude corresponding to word 4, degrees x10000',
                    6:'latitude degrees/image line, degrees x10000',
                    7:'longitude degrees/image line, degrees x10000',
                    8:'radius of the planet, meters',
                    9:'eccentricity of the planet, x1000000',
                    10:'coordinate type, >= 0 planetodetic, < 0 planetocentric',
                    11:'longitude convention, >= 0 west positive, < 0 west negative'
                }

    def __init__(self, navblock=None, auxblock=None, **kwargs):
        inavblock = navblock
        #for n, v in self.labels.items():
        #	print n, v, inavblock[n-1]
        self.itype = 2
        self.iwest = 1 if inavblock[10] >= 0 else -1

        self.xrow = inavblock[1]
        self.xcol = inavblock[3]
        ipowlat = inavblock[11]
        if ipowlat == 0:
            ipowlat = 4
        self.zslat = inavblock[2] / (10. ** ipowlat)
        ipowlon = inavblock[12]
        if ipowlon == 0:
            ipowlon == 4
        self.zslon = inavblock[4] / (10. ** ipowlon)

        ipowdlin = inavblock[13]
        if ipowdlin == 0:
            ipowdlin == 4
        self.zdlat = float(inavblock[5]) / (10. ** ipowdlin)
        ipowdele = inavblock[14]
        if ipowdele == 0:
            ipowdele == 4
        self.zdlon = float(inavblock[6]) / (10 ** ipowdele)
        if self.xcol == 1:
            self.zslon = self.zslon - 180.0 * self.iwest

    def toLinEle(self, latlon=None):

        number = len(latlon[0])
        linele = [[nan for i in range(number)], [nan for i in range(number)]]
        for point in range(number):
            xlat = latlon[self.indexLat][point]
            xlon = -latlon[self.indexLon][point] if self.iwest == 1 else latlon[self.indexLon][point]

            if xlon > (self.zslon+180):
                xlon = xlon - 360
            elif xlon < (self.zslon-180):
                xlon = xlon + 360

            xlin = self.xrow - (xlat - self.zslat)/self.zdlat
            if self.xcol == 1:
                xele = self.xcol - (xlon - self.zslon-180*self.iwest) / (self.zdlon*self.iwest)
            else:
                xele = self.xcol - (xlon - self.zslon)/(self.zdlon*self.iwest)
            linele[self.indexLine][point] = float(xlin)
            linele[self.indexEle][point] = float(xele)
        #return linele

        return self.imageCoordToAreaCoord(linele=linele)

    def toLatLon(self, linele=None, lon_circular=True ):

        number = len(linele[0])
        latlon = [[nan for i in range(number)], [nan for i in range(number)]]


        # Convert array to Image coordinatesfor computations

        imglinele = self.areaCoordtoImageCoord(linele=linele)

        for point in range(number):
            xlin = imglinele[self.indexLine][point]
            xele = imglinele[self.indexEle][point]

            xldif = self.xrow - xlin
            if self.xcol == 1:
                xedif = self.iwest * (xele-self.xcol)
                xlon = self.zslon + 180 * self.iwest - xedif*self.zdlon
            else:
                xedif = self.iwest * (self.xcol - xele)
                xlon = self.zslon + xedif * self.zdlon

            xlat = self.zslat + xldif * self.zdlat
            if  xlat > 90 or xlat < -90:
                xlat = nan

            if xlon > (self.zslon + 180) or xlon <  (self.zslon -180):
                xlon = nan
            if not math.isnan(xlon):
                if lon_circular:
                    if xlon < -180:
                        xlon = xlon + 360
                    if xlon > 180:
                        xlon = xlon - 360

            if math.isnan(xlon) or math.isnan(xlat):
                latlon[self.indexLat][point] = nan
                latlon[self.indexLon][point] = nan
            else:
                latlon[self.indexLat][point] = float(xlat)
                latlon[self.indexLon][point] = -xlon if self.iwest == 1 else xlon

        return latlon








    def __str__(self):
        s = ''
        s+= 'itype %i \n' % self.itype
        s+= 'xrow %i \n' % self.xrow
        s+= 'xcol %i \n' % self.xcol
        s+= 'iwest %i \n' % self.iwest
        s+= 'zslat %.6f \n' % self.zslat
        s+= 'zslon %.6f \n' % self.zslon
        s+= 'zdlat %.6f \n' % self.zdlat
        s+= 'zdlon %.6f \n' % self.zdlon
        return s



class GMSXNav(AreaNav):



    def __init__(self, navblock=None, auxblock=None, **kwargs):
        self.bParms = np.zeros((3200), dtype=np.int8)
        self.subLon = nan
        self.subLat = nan
        self.resLin = np.zeros((4,), dtype=np.float32)
        self.resEle = np.zeros((4,), dtype=np.float32)
        self.rlic = np.zeros((4,), dtype=np.float32)
        self.relmfc = np.zeros((4,), dtype=np.float32)
        self.senssu = np.zeros((4,), dtype=np.float32)
        self.rline = np.zeros((4,), dtype=np.float32)
        self.relem = np.zeros((4,), dtype=np.float32)
        self.vmis = np.zeros((3,), dtype=np.float32)
        self.elmis = np.zeros((3, 3), dtype=np.float32)
        self.dtims = 0.
        self.dspin = 0.
        self.sitagt = 0.
        self.sunalp = 0.
        self.sundel = 0.
        self.sat = [nan, nan, nan]
        self.sp = [nan, nan, nan]
        self.ss = [nan, nan, nan]
        self.orbt1 = np.zeros((35, 8), dtype=np.float64)
        self.atit = np.zeros((10, 10), dtype=np.float64)

        self.cdr = math.pi / 180.
        self.crd = 180. / math.pi
        self.hpai = math.pi / 2.
        self. dpai = math.pi / 2.
        self.ea = 6378136.0
        self.ef = 1 / 298.257




        navtype = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=4, format_str='s'),
                      struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=1, format_str=self.int_format),navblock[self.STTYPE]))[0]
        if navtype != self.GMSX:
            raise Exception(f'Invalid navigation type {navtype}')




        #I will use a completely different approach, numpy based
        #convert the string MORE to a integer
        #excluded_value = struct.unpack('>1i',struct.pack('>4s', 'MORE'))[0]
        excluded_value = struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=1, format_str=self.int_format),struct.pack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=4, format_str='s'), 'MORE'))[0]

        #convert initial nav block to numpy array
        navblock_array = np.array(navblock, dtype=np.int64)
        #compute mask by excluding elements whose value == 'MORE'
        navblock_mask = navblock_array!=excluded_value

        #filter the initial block end exclude first value (GMSX), then convert to list
        cleaned_navblock = navblock_array[navblock_mask][1:].tolist()



        #convert back to byte string
        fmt = '{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=len(cleaned_navblock), format_str=self.int_format)
        self.strbParms = struct.pack(fmt, *cleaned_navblock)
        fmt1 = '{byte_order:s}{size:d}{format_str:s}'.format(byte_order=self.byte_order, size=len(cleaned_navblock)*4, format_str='b')
        #convert to  bytes
        self.bParms = struct.unpack(fmt1, self.strbParms)

        self.decOABlock(self.bParms, self.strbParms, 0)
        #int to 4 bytes intToBytes
        #print struct.unpack('>4b', struct.pack('>1i', 123277))

    @staticmethod
    def sv01000(byte_str_array=None, mult_pow=None, byte_order=None):
        """
        Converts a byte string array of a certain length to a float number as per
        http://www.ssec.wisc.edu/mcidas/doc/prog_man/current/formats-11.html#25566
        I have not followed a close to Java implementation because it would have been slow
        Rather than that i used numpy and struct
        The original Java code can deal with 4 or 6 bytes, this code can deal with arbitrary lenghts
        The idea ois that given a byte string of a certain length it is converted to a float number
        by anding &  the most significant byte with 127 '01111111' which is ignoring the most significand bit for BIG endian (it seems the GMSX data is in BIG endian)
        na then multypling with  2  at the power corresponding to the index of each byte expressed as the maximum number of bits (see pows var)
        and the summbin this multiplication
        This implementation is byte order specific which is bad but as the data are just 1 byte  it holds!
        "The Type column in the tables below shows scaled integers in the format R*M.N. The R indicates real numbers, M is the number of bytes and N is the exponent."


        :param byte_str_array: a
        :param mult_pow:
        :param byte_order:
        :return:
        """
        size = len(byte_str_array)

        #get unsigned and signed versions

        unsigned_byte_array = np.array(struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=byte_order, size=size,format_str='B'), byte_str_array))
        signed_byte_array = np.array(struct.unpack('{byte_order:s}{size:d}{format_str:s}'.format(byte_order=byte_order, size=size,format_str='b'), byte_str_array))

        #patch the MSB of the unsigned
        unsigned_byte_array[0] = unsigned_byte_array[0]&127
        #print signed_byte_array
        negative = True if signed_byte_array[0] < 0 else False
        #the power
        pows = 2. ** (np.arange(size)[::-1] * 8)

        sm = sum(unsigned_byte_array*pows)/(10**mult_pow)
        if negative:
            sm= -sm
        return np.asscalar(sm)

    @staticmethod
    def normalize_lon(lon=None, lon_range=180):
        rtn = lon % lon_range
        if rtn <=0:
            rtn+=lon_range
        if rtn> lon_range/2.:
            rtn-=lon_range
        return rtn

    def decOABlock(self, byte_array, byte_string_array, form):
        """
        decode Orbit and Attitude data block
        :param byte_array:
        :param form:
        :return:
        """

        self.dtims = self.sv01000(byte_string_array[:6], 8, self.byte_order)
        #print self.dtims
        self.dspin = self.sv01000(byte_string_array[240:246], 8, self.byte_order)
        #print self.dspin
        self.resLin[0] = self.sv01000(byte_string_array[6:10],8, self.byte_order)
        self.resLin[1] = self.sv01000(byte_string_array[10:14],8, self.byte_order)
        self.resLin[2] = self.sv01000(byte_string_array[10:14],8, self.byte_order)
        self.resLin[3] = self.sv01000(byte_string_array[10:14],8, self.byte_order)
        #print self.resLin
        self.resEle[0] = self.sv01000(byte_string_array[14:18], 10, self.byte_order)
        self.resEle[1] = self.sv01000(byte_string_array[18:22], 10, self.byte_order)
        self.resEle[2] = self.sv01000(byte_string_array[18:22], 10, self.byte_order)
        self.resEle[3] = self.sv01000(byte_string_array[18:22], 10, self.byte_order)
        #print self.resEle
        self.rlic[0] = self.sv01000(byte_string_array[22:26], 4, self.byte_order)
        self.rlic[1] = self.sv01000(byte_string_array[26:30], 4, self.byte_order)
        self.rlic[2] = self.sv01000(byte_string_array[110:114], 4, self.byte_order)
        self.rlic[3] = self.sv01000(byte_string_array[114:118], 4, self.byte_order)
        #print self.rlic
        self.relmfc[0] = self.sv01000(byte_string_array[30:34], 4, self.byte_order)
        self.relmfc[1] = self.sv01000(byte_string_array[34:38], 4, self.byte_order)
        self.relmfc[2] = self.sv01000(byte_string_array[118:122], 4, self.byte_order)
        self.relmfc[3] = self.sv01000(byte_string_array[122:126], 4, self.byte_order)
        #print self.relmfc
        self.senssu[0] = self.sv01000(byte_string_array[38:42], 0, self.byte_order)
        self.senssu[1] = self.sv01000(byte_string_array[42:46], 0, self.byte_order)
        self.senssu[2] = self.sv01000(byte_string_array[42:46], 0, self.byte_order)
        self.senssu[3] = self.sv01000(byte_string_array[42:46], 0, self.byte_order)
        #print self.senssu
        self.rline[0] = self.sv01000(byte_string_array[46:50], 0, self.byte_order)
        self.rline[1] = self.sv01000(byte_string_array[50:54], 0, self.byte_order)
        self.rline[2] = self.sv01000(byte_string_array[50:54], 0, self.byte_order)
        self.rline[3] = self.sv01000(byte_string_array[50:54], 0, self.byte_order)
        #print self.rline
        self.relem[0] = self.sv01000(byte_string_array[54:58], 0, self.byte_order)
        self.relem[1] = self.sv01000(byte_string_array[58:62], 0, self.byte_order)
        self.relem[2] = self.sv01000(byte_string_array[58:62], 0, self.byte_order)
        self.relem[3] = self.sv01000(byte_string_array[58:62], 0, self.byte_order)
        #print self.relem
        self.vmis[0] = self.sv01000(byte_string_array[62:66], 10, self.byte_order)
        self.vmis[1] = self.sv01000(byte_string_array[66:70], 10, self.byte_order)
        self.vmis[2] = self.sv01000(byte_string_array[70:74], 10, self.byte_order)
        #print self.vmis
        self.elmis[0][0] = self.sv01000(byte_string_array[74:78], 7, self.byte_order)
        self.elmis[1][0] = self.sv01000(byte_string_array[78:82], 10, self.byte_order)
        self.elmis[2][0] = self.sv01000(byte_string_array[82:86], 10, self.byte_order)
        self.elmis[0][1] = self.sv01000(byte_string_array[86:90], 10, self.byte_order)
        self.elmis[1][1] = self.sv01000(byte_string_array[90:94], 7, self.byte_order)
        self.elmis[2][1] = self.sv01000(byte_string_array[94:98], 10, self.byte_order)
        self.elmis[0][2] = self.sv01000(byte_string_array[98:102], 10, self.byte_order)
        self.elmis[1][2] = self.sv01000(byte_string_array[102:106], 10, self.byte_order)
        self.elmis[2][2] = self.sv01000(byte_string_array[106:110], 7, self.byte_order)
        #for i in range(3):
        #    print self.elmis[i]
        self.subLon = self.sv01000(byte_string_array[198:204], 6, self.byte_order)
        #print self.subLon
        self.subLat = self.sv01000(byte_string_array[204:210], 6, self.byte_order)
        #print self.subLat
        for i in range(10):
            if form == 1:
                j = (i - 1) * 64 + 256
            if form == 0:
                j = (i - 1) * 48 + 256
            self.atit[0][i] = self.sv01000(byte_string_array[j:j+6], 8, self.byte_order)
            self.atit[2][i] = self.sv01000(byte_string_array[j+12:j+12+6], 8, self.byte_order)
            self.atit[3][i] = self.sv01000(byte_string_array[j+18:j+18+6], 11, self.byte_order)
            self.atit[4][i] = self.sv01000(byte_string_array[j+24:j+24+6], 8, self.byte_order)
            self.atit[5][i] = self.sv01000(byte_string_array[j+30:j+30+6], 8, self.byte_order)

            # if i == 0:
            #     print self.atit[0][i]
            #     print self.atit[2][i]
            #     print self.atit[3][i]
            #     print self.atit[4][i]
            #     print self.atit[5][i]
            #
        for i in range(8):
            if form == 1:
                j = (i-1)*256 + 896
            if form == 0:
                j = (i-1) * 200 + 736
            self.orbt1[0][i] = self.sv01000(byte_string_array[j:j+6], 8, self.byte_order)
            self.orbt1[8][i] = self.sv01000(byte_string_array[j+48:j+48+6], 6, self.byte_order)
            self.orbt1[9][i] = self.sv01000(byte_string_array[j+54:j+54+6], 6, self.byte_order)
            self.orbt1[10][i] = self.sv01000(byte_string_array[j+60:j+60+6], 6, self.byte_order)
            self.orbt1[14][i] = self.sv01000(byte_string_array[j+84:j+84+6], 8, self.byte_order)
            self.orbt1[17][i] = self.sv01000(byte_string_array[j+102:j+102+6], 8, self.byte_order)
            self.orbt1[18][i] = self.sv01000(byte_string_array[j+108:j+108+6], 8, self.byte_order)
            self.orbt1[19][i] = self.sv01000(byte_string_array[j+128:j+128+6], 12, self.byte_order)
            self.orbt1[20][i] = self.sv01000(byte_string_array[j+134:j+134+6], 14, self.byte_order)
            self.orbt1[21][i] = self.sv01000(byte_string_array[j+140:j+140+6], 14, self.byte_order)
            self.orbt1[22][i] = self.sv01000(byte_string_array[j+146:j+146+6], 14, self.byte_order)
            self.orbt1[23][i] = self.sv01000(byte_string_array[j+152:j+152+6], 12, self.byte_order)
            self.orbt1[24][i] = self.sv01000(byte_string_array[j+158:j+158+6], 16, self.byte_order)
            self.orbt1[25][i] = self.sv01000(byte_string_array[j+164:j+164+6], 12, self.byte_order)
            self.orbt1[26][i] = self.sv01000(byte_string_array[j+170:j+170+6], 16, self.byte_order)
            self.orbt1[27][i] = self.sv01000(byte_string_array[j+176:j+176+6], 12, self.byte_order)


            # print '-'*50
            # for mmm in (0, 8, 9, 10, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27):
            #     print 'orbt1[%d][%d] %.5f %s ' % (mmm, i, self.orbt1[mmm][i], type(self.orbt1[mmm][i]))
            #

    def linele_scantime(self, linele=None):
        mode = -1
        imglinele = self.areaCoordtoImageCoord_np(linele=linele)

        rPix = imglinele[1]
        rLin = imglinele[0]

        shp = rPix.shape
        # cmask = np.ones(lat.shape).astype(np.bool)
        rPix = rPix.ravel()
        rLin = rLin.ravel()
        # vissr frame
        lMode = abs(mode) - 1

        rsamp = self.resEle[lMode]

        sens = self.senssu[lMode]

        rtim = ((self.rint_np(rLin - 1) / sens) + (rPix * rsamp) / self.dpai) / (self.dspin * 1440.) + self.dtims
        return rtim.reshape(shp)




    def latlon_scantime(self, latlon=None):
        mode = 1
        lon = ((latlon[self.indexLon] + 180.0) % 360.0) - 180.0
        lat = latlon[self.indexLat]
        sz = lat.size
        shp = lon.shape


        lon = lon.ravel()
        lat = lat.ravel()

        # now allocate matrices for the computataion dependant attributes cecessary in array mode
        # these attributes are scalars in point mode but they become array in array mode
        self.sat = np.zeros((3, sz))
        self.sp = np.zeros((3, sz))
        self.ss = np.zeros((3, sz))
        self.sitagt = np.zeros((sz))
        self.sunalp = np.zeros((sz))
        self.sundel = np.zeros((sz))
        tri = np.zeros((sz))
        trj = np.zeros((sz))

        # cmask = np.ones(lat.shape).astype(np.bool)

        stn1 = [nan, nan, nan]


        m = (lat > 90) | (lat < -90)
        if m[m].size > 0:
            return
        # vissr frame
        lMode = abs(mode) - 1
        rstep = self.resLin[lMode]
        rfcl = self.rlic[lMode]
        sens = self.senssu[lMode]
        dLat = lat * self.cdr
        dLon = lon * self.cdr
        ee = 2. * self.ef - (self.ef * self.ef)

        sin_lat = np.sin(dLat)
        cos_lat = np.cos(dLat)
        sin_lon = np.sin(dLon)
        cos_lon = np.cos(dLon)
        en = (self.ea / np.sqrt(1. - ee * sin_lat * sin_lat))

        stn1[0] = en * cos_lat * cos_lon
        stn1[1] = en * cos_lat * sin_lon
        stn1[2] = en * (1. - ee) * sin_lat
        rio = rfcl - np.arctan(sin_lat / (6.610689 - cos_lat)) / rstep  # i guess 6.1 ms is the scan ytime for a swath, it makes sens if I look to this line of code

        rtim = self.dtims + (rio / sens / 1440.) / self.dspin
        return rtim.reshape(shp)

    def compute_orbital_zones(self, rscantime=None):
        _shp = rscantime.shape
        rscantime = rscantime.ravel()
        orbt_values = self.orbt1[0]
        orbt_zones = np.digitize(rscantime, orbt_values) - 1  # ?? should i extract one
        return orbt_zones.reshape(_shp)
        # unique_orb_zones = np.unique(orbt_zones)
        # for orbt_zone in unique_orb_zones:
        #     m = orbt_zones == orbt_zone

    def toLinEle2(self, latlon=None):
        #print 'indexLat', self.indexLat
        #print 'indexLoin', self.indexLon

        mode = 1
        lon = ((latlon[self.indexLon] + 180.0) % 360.0) - 180.0
        lat = latlon[self.indexLat]
        sz = lat.size
        shp = lon.shape



        lon = lon.ravel()
        lat = lat.ravel()


        #now allocate matrices for the computataion dependant attributes cecessary in array mode
        #these attributes are scalars in point mode but they become array in array mode
        self.sat = np.zeros((3, sz))
        self.sp = np.zeros((3, sz))
        self.ss = np.zeros((3, sz))
        self.sitagt = np.zeros((sz))
        self.sunalp = np.zeros((sz))
        self.sundel = np.zeros((sz))
        tri = np.zeros((sz))
        trj = np.zeros((sz))


        #cmask = np.ones(lat.shape).astype(np.bool)

        stn1 = np.zeros((3, sz))

        slv = np.zeros((3, sz))
        sw3 = np.zeros((3, sz))
        m = (lat > 90) | (lat <-90)
        if m[m].size> 0:
            return None, None



        # vissr frame
        lMode = abs(mode) - 1

        rstep = self.resLin[lMode]
        rsamp = self.resEle[lMode]
        rfcl = self.rlic[lMode]

        rfcp = self.relmfc[lMode]

        sens = self.senssu[lMode]
        rftl = self.rline[lMode] + .5
        rftp = self.relem[lMode] + .5

        dLat = lat * self.cdr
        dLon = lon * self.cdr
        ee = 2. * self.ef - (self.ef * self.ef)

        sin_lat = np.sin(dLat)
        cos_lat = np.cos(dLat)
        sin_lon = np.sin(dLon)
        cos_lon = np.cos(dLon)
        en = (self.ea / np.sqrt(1. - ee * sin_lat * sin_lat))

        stn1[0] = en * cos_lat * cos_lon
        stn1[1] = en * cos_lat * sin_lon
        stn1[2] = en * (1. - ee) * sin_lat
        rio = rfcl - np.arctan(sin_lat / (6.610689 - cos_lat)) / rstep  # i guess 6.1 ms is the scan ytime for a swath, it makes sens if I look to this line of code

        rtim = self.dtims + (rio / sens / 1440.) / self.dspin



        gg = 0
        while True:
            #print 'iter', gg, 'rio size', rio.size
            # if gg == 0:
			#
            #     print 'sp', self.sp
            #     print 'ss', self.ss
            #     print 'sat', self.sat
            beta = self.mg1100_np(rtim)





            # if gg == 0:
            #     print 'beta', beta, beta.dtype
            #     print 'sp', self.sp
            #     print 'ss', self.ss
            #     print 'sat', self.sat

            sw1 = self.mg1220_np(self.sp, self.ss)

            sw2 = self.mg1220_np(sw1, self.sp)

            bc = np.cos(beta)
            bs = np.sin(beta)
            sw3[0] = (sw1[0] * bs) + (sw2[0] * bc)
            sw3[1] = (sw1[1] * bs) + (sw2[1] * bc)
            sw3[2] = (sw1[2] * bs) + (sw2[2] * bc)
            sx = self.mg1200_np(sw3)
            sy = self.mg1220_np(self.sp, sx)

            # try:
			#
            #     slv[0] = stn1[0][cmask] - self.sat[0][cmask]
            #     slv[1] = stn1[1][cmask] - self.sat[1][cmask]
            #     slv[2] = stn1[2][cmask] - self.sat[2][cmask]
            # except UnboundLocalError:
            #     slv[0] = stn1[0] - self.sat[0]
            #     slv[1] = stn1[1] - self.sat[1]
            #     slv[2] = stn1[2] - self.sat[2]
            slv[0] = stn1[0] - self.sat[0]
            slv[1] = stn1[1] - self.sat[1]
            slv[2] = stn1[2] - self.sat[2]
            sl = self.mg1200_np(slv)

            sw2 = self.mg1210(self.sp, sl)
            sw3 = self.mg1210(sy, sw2)
            tp = self.mg1230_np(sy, sw2)
            tf = (self.sp[0] * sw3[0]) + (self.sp[1] * sw3[1]) + (self.sp[2] * sw3[2])

            if np.all(tf < 0):
                tp = -tp

            tl = self.mg1230_np(self.sp, sl)




            ri = (self.hpai - tl) / rstep + rfcl - (self.vmis[1] / rstep)


            rj = tp / rsamp + rfcp + (self.vmis[2] / rsamp) - (self.hpai - tl) * np.tan(self.vmis[0]) / rsamp


            #now this is a bit tricky as I ned to update the cmask in order to correctly emulate the original while loop
            gg += 1
            diff = np.abs(ri-rio)


            cmask = diff >=1
            # from pylab import show, imshow, title, colorbar
            # imshow(cmask.reshape((shp)))
            # title('cmask')
            # colorbar()
            # show()
            # imshow(rio.reshape((shp)))
            # title('rio')
            # colorbar()
            # show()
            #print 'diff', diff.mean()


            if cmask[cmask].size > 0:
                print ('masking %s for %s in percent %s' % (cmask[cmask].size, lat.size, cmask[cmask].size/float(lat.size) * 100))

                # from pylab import show, imshow, title, colorbar
                # imshow(cmask.reshape((3900, 2700)))
                # colorbar()
                # title('cmask%s' % gg)
                # show()
                #if abs(ri - rio) >= 1.:
                #mask the rtim adn rio, i am not sure if this is enough thoufh
                #rtim = ((self.rint_np((ri - 1) / sens) + (rj * rsamp) / self.dpai) / (self.dspin * 1440.0) + self.dtims)[cmask]
                rtim1 = (self.rint_np((ri - 1) / sens) + rj * rsamp / self.dpai) / (self.dspin * 1440.0) + self.dtims

                rtim[cmask] = rtim1[cmask]

                rio[cmask] = ri[cmask]
                #rio = ri[cmask]
                continue

            break

        #print 'end rio size', rio.size
        #sanity check for bounds
        if np.any((ri < 0) | (ri > rftl)):
            rc = 4
            print ('inv line')
            return None, None

        if np.any((rj < 0) | (rj > rftp)):
            rc = 5
            print ('inv ele')
            return None, None


        rri = ri.reshape(shp)
        rrj = rj.reshape(shp)

        return self.imageCoordToAreaCoord_np(linele=[rri, rrj])

    def toLinEle(self, latlon=None):
        mode = 1
        count = len(latlon[0])
        linele = [[nan for i in range(count)], [nan for i in range(count)]]

        for point in range(count):
            #init as not navigable
            linele[self.indexLine][point] = nan
            linele[self.indexEle][point] = nan
            if latlon[self.indexLat][point] > 90:
                continue
            #normalize
            lon =  ((latlon[self.indexLon][point] + 180.0) % 360.0) - 180.0
            lat = latlon[self.indexLat][point]

            rtnPoint = self.msivsr(mode,0,0,lat, lon)

            linele[self.indexLine][point] = rtnPoint[0]
            linele[self.indexEle][point] = rtnPoint[1]

        return self.imageCoordToAreaCoord(linele=linele)

    def toLatLon(self, linele=None):
        mode = -1
        count = len(linele[0])
        latlon = [[nan ], [nan]]
        #transform file to image coordinates
        imgLinEle = self.areaCoordtoImageCoord(linele=linele)

        #print imgLinEle
        for point in range(count):
            rtnPoint = self.msivsr(mode, imgLinEle[self.indexEle][point], imgLinEle[self.indexLine][point], 0,0,)
            #print rtnPoint
            lon = rtnPoint[1]
            lon = ((lon + 180.0) % 360.0) - 180.0
            if abs(lon) > 180 or (-lon > 90-self.subLon and -lon < 270 - self.subLon):
                return [nan, nan]

            latlon[self.indexLat][point] = rtnPoint[0]
            latlon[self.indexLon][point] = lon
        return latlon

    def toLatLon2(self, linele=None, ):

        mode = -1
        imglinele = self.areaCoordtoImageCoord_np(linele=linele)

        rPix = imglinele[1]
        rLin = imglinele[0]


        shp = rPix.shape
        # cmask = np.ones(lat.shape).astype(np.bool)
        rPix = rPix.ravel()
        rLin = rLin.ravel()

        stn1 = [nan, nan, nan]
        sw3 = [nan, nan, nan]


        # vissr frame
        lMode = abs(mode) - 1
        rstep = self.resLin[lMode]
        rsamp = self.resEle[lMode]
        rfcl = self.rlic[lMode]
        rfcp = self.relmfc[lMode]
        sens = self.senssu[lMode]
        #print rLin, sens, rPix, rsamp, self.dpai, self.dspin, self.dtims

        rtim = ((self.rint_np(rLin - 1) / sens) + (rPix * rsamp) / self.dpai) / (self.dspin * 1440.) + self.dtims
        #print 'rtim', rtim
        beta = self.mg1100_np(rtim)
        # print 'beta: ', beta
        # print 'sp:', self.sp
        # print 'ss:', self.ss
        sw1 = self.mg1220_np(self.sp, self.ss)
        sw2 = self.mg1220_np(sw1, self.sp)

        bc = np.cos(beta)
        bs = np.sin(beta)
        sw3[0] = (sw1[0] * bs) + (sw2[0] * bc)
        sw3[1] = (sw1[1] * bs) + (sw2[1] * bc)
        sw3[2] = (sw1[2] * bs) + (sw2[2] * bc)
        # print 'sw1:', sw1
        # print 'sw2:', sw2
        # print 'sw3', sw3
        sx = self.mg1200_np(sw3)
        sy = self.mg1220_np(self.sp, sx)
        # print 'sx:', sx
        # print 'sy:', sy
        pc = np.cos(rstep * (rLin - rfcl))
        ps = np.sin(rstep * (rLin - rfcl))
        qc = np.cos(rsamp * (rPix - rfcp))
        qs = np.sin(rsamp * (rPix - rfcp))
        sw1[0] = (self.elmis[0][0] * pc) + (self.elmis[0][2] * ps)
        sw1[1] = (self.elmis[1][0] * pc) + (self.elmis[1][2] * ps)
        sw1[2] = (self.elmis[2][0] * pc) + (self.elmis[2][2] * ps)
        # print sw1
        sw2[0] = (qc * sw1[0]) - (qs * sw1[1])
        sw2[1] = (qs * sw1[0]) + (qc * sw1[1])
        sw2[2] = sw1[2]
        # print sw2
        sw3[0] = (sx[0] * sw2[0]) + (sy[0] * sw2[1]) + (self.sp[0] * sw2[2])
        sw3[1] = (sx[1] * sw2[0]) + (sy[1] * sw2[1]) + (self.sp[1] * sw2[2])
        sw3[2] = (sx[2] * sw2[0]) + (sy[2] * sw2[1]) + (self.sp[2] * sw2[2])

        # print 'sw1:', sw1
        # print 'sw2:', sw2
        # print 'sw3', sw3

        # print sw3
        sl = self.mg1200_np(sw3)
        # print 'sl:', sl
        # print 'sat %.5f, %.5f, %.5f:' % (self.sat[0], self.sat[1], self.sat[2])
        def1 = (1. - self.ef) * (1 - self.ef)
        # print "def %.9f" % def1
        dda = def1 * ((sl[0] * sl[0]) + (sl[1] * sl[1])) + (sl[2] * sl[2])
        ddb = def1 * ((self.sat[0] * sl[0]) + (self.sat[1] * sl[1])) + (self.sat[2] * sl[2])
        # print def1, self.sat[0], self.sat[1], self.sat[2], self.ea
        ddc = def1 * ((self.sat[0] * self.sat[0]) + (self.sat[1] * self.sat[1]) - (self.ea * self.ea)) + (self.sat[2] * self.sat[2])
        dd = (ddb * ddb) - (dda * ddc)

        # print 'dda %.5f' % dda
        # print 'ddb %.5f' % ddb
        # print 'ddc %.5f' % ddc
        # print 'dd %.5f' % dd

        # if dd >= 0 and dda != 0:
        #     dk1 = (-ddb + np.sqrt(dd)) / dda
        #     dk2 = (- ddb - np.sqrt(dd)) / dda
        # else:
        #     rc = 6
        #     return (None, None)

        #this code is equivalent to the above  commented python code
        ddm = (dd >= 0) & (dda != 0)
        # now we need to be careful here as we will return invalid data
        dk1 = np.where(ddm, (-ddb + np.sqrt(dd)) / dda, nan)
        dk2 = np.where(ddm, (-ddb - np.sqrt(dd)) / dda, nan)


        # if abs(dk1) < abs(dk2):
        #     dk = dk1
        # else:
        #     dk = dk2
        # print 'dk %.5f' % dk
        abscondm = np.abs(dk1) < np.abs(dk2)
        dk = np.where(abscondm, dk1, dk2)
        stn1[0] = self.sat[0] + (dk * sl[0])
        stn1[1] = self.sat[1] + (dk * sl[1])
        stn1[2] = self.sat[2] + (dk * sl[2])
        # print 'stn1', stn1
        dLat = np.arctan(stn1[2] / (def1 * np.sqrt((stn1[0] * stn1[0]) + (stn1[1] * stn1[1]))))
        # print dLat




        # if stn1[0] != 0:
        #     dLon = np.arctan(stn1[1] / stn1[0])
        #
        #     if stn1[0] < 0 and stn1[1] >= 0:
        #         dLon = dLon + self.PI
        #     if stn1[0] < 0 and stn1[1] < 0:
        #         dLon = dLon - self.PI
        # else:
        #     if stn1[1] > 0:
        #         dLon = self.hpai
        #     else:
        #         dLon = -self.hpai

        #this code is equivalent to above python code
        stn1_0_m = stn1[0] != 0
        stn1_1_m = stn1[1] > 0
        dLon = np.arctan(stn1[1] / stn1[0])
        acon = (stn1[0] < 0) & (stn1[1] >= 0)[stn1_0_m]
        bcon = (stn1[0] < 0) & (stn1[1] < 0)[stn1_0_m]
        if np.any(acon):
            dLon[acon] += self.PI
        if np.any(bcon):
            dLon[bcon] -= self.PI
        if np.any(~stn1_0_m):
            dLon[~stn1_0_m] = np.where(stn1_1_m, self.hpai, -self.hpai)




        lat = dLat * self.crd
        lon = dLon * self.crd

        #normalize
        lon = ((lon + 180.0) % 360.0) - 180.0
        return [lat.reshape(shp), lon.reshape(shp)]


    def msivsr(self, iMode, rPix, rLin, rLat, rLon):
        """
        actual conversion to/from lat/lon or line/elem
        @param iMode - conversion mode, to lat/lon or to line/elem
        @param rPix - float pixel or element value
        @param rLin - float line value
        @param rLat - float latitude value
        @param rLon - float longitude value

        @return array of two floating point values, lat/lon or line/elem pair
        """
        name = 'msivsr'
        point = [nan, nan]
        stn1 = [nan, nan, nan]
        slv = [nan, nan, nan]
        sw3  = [nan, nan, nan]
        #sanity check
        if iMode > 4:
            return point
        if np.any(np.abs(rLat)>90) and iMode > 0:
            return point
        #vissr frame
        lMode = abs(iMode) -1
        rstep = self.resLin[lMode]
        rsamp = self.resEle[lMode]
        rfcl = self.rlic[lMode]
        rfcp = self.relmfc[lMode]
        sens = self.senssu[lMode]
        rftl = self.rline[lMode] + .5
        rftp = self.relem[lMode] + .5



        if iMode > 0: #transformation, geographical -> VISSR
            dLat = rLat * self.cdr
            dLon = rLon * self.cdr
            ee = 2.*self.ef - (self.ef * self.ef)
            sin_lat = np.sin(dLat)
            cos_lat = np.cos(dLat)
            sin_lon = np.sin(dLon)
            cos_lon = np.cos(dLon)
            en = self.ea / np.sqrt(1. - ee*sin_lat*sin_lat)
            stn1[0] = en * cos_lat * cos_lon
            stn1[1] = en * cos_lat * sin_lon
            stn1[2] = en * (1. - ee) * sin_lat
            rio = rfcl-np.arctan(sin_lat / (6.610689 - cos_lat)) / rstep # i guess 6.1 ms is the scan ytime for a swath, it makes sens if I look to this line of code
            rtim = self.dtims + (rio/sens/1440.)/ self.dspin

            gg = 0
            while True:
                #print 'mg1100'
                #print 'classic iter ', gg
                # if gg == 0:
                #     print 'sp', self.sp
                #     print 'ss', self.ss
                #     print 'sat', self.sat

                beta = self.mg1100(rtim)
                # if gg == 0:

                #     print 'sp', self.sp
                #     print 'ss', self.ss
                #     print 'sat', self.sat

                sw1 = self.mg1220(self.sp, self.ss)

                sw2 = self.mg1220(sw1, self.sp)

                bc = np.cos(beta)
                bs = np.sin(beta)
                sw3[0] = (sw1[0] * bs) + (sw2[0] * bc)
                sw3[1] = (sw1[1] * bs) + (sw2[1] * bc)
                sw3[2] = (sw1[2] * bs) + (sw2[2] * bc)
                sx = self.mg1200(sw3)
                sy = self.mg1220(self.sp, sx)

                slv[0] = stn1[0] - self.sat[0]
                slv[1] = stn1[1] - self.sat[1]
                slv[2] = stn1[2] - self.sat[2]

                sl = self.mg1200(slv)

                sw2 = self.mg1210(self.sp, sl)
                sw3 = self.mg1210(sy, sw2)
                tp = self.mg1230(sy, sw2)
                tf = (self.sp[0] * sw3[0]) + (self.sp[1] * sw3[1]) + (self.sp[2] * sw3[2])

                if tf < 0:
                    tp = -tp


                tl = self.mg1230(self.sp, sl)

                ri = float(self.hpai - tl) / rstep + rfcl - (self.vmis[1] / rstep)
                rj = float(tp / rsamp + rfcp + (self.vmis[2] / rsamp) - (self.hpai - tl) * np.tan(self.vmis[0]) / rsamp)

                gg+=1
                if abs(ri - rio) >= 1.:
                    rtim = (self.rint((ri - 1) / sens) + (rj * rsamp) / self.dpai) / (self.dspin * 1440.0) + self.dtims
                    rio = ri
                    continue
                break

            rLin = ri
            rPix = rj

            if rLin > 0 and rLin < rftl:
                if rPix > 0 and rPix < rftp:
                    point[0] = rLin
                    point[1] = rPix

                else:
                    rc = 5
            else:
                rc = 4

        if iMode < 0: #transformation, VISSR -> geographical
            #print rLin, sens, rPix, rsamp, self.dpai, self.dspin, self.dtims
            rtim = ((self.rint(rLin-1) / sens) + (rPix*rsamp) / self.dpai) / (self.dspin*1440.) + self.dtims
            #print 'rtim: ', rtim
            beta = self.mg1100(rtim)
            #print 'beta: ', beta
            #print 'sp:', self.sp
            #print 'ss:', self.ss
            sw1 = self.mg1220(self.sp, self.ss)
            sw2 = self.mg1220(sw1, self.sp)

            bc = np.cos(beta)
            bs = np.sin(beta)
            sw3[0] = (sw1[0] * bs) + (sw2[0] * bc)
            sw3[1] = (sw1[1] * bs) + (sw2[1] * bc)
            sw3[2] = (sw1[2] * bs) + (sw2[2] * bc)
            #print 'sw1:', sw1
            #print 'sw2:', sw2
            #print 'sw3', sw3
            sx = self.mg1200(sw3)
            sy = self.mg1220(self.sp, sx)
            #print 'sx:', sx
            #print 'sy:', sy
            pc = np.cos(rstep * (rLin-rfcl))
            ps = np.sin(rstep * (rLin-rfcl))
            qc = np.cos(rsamp * (rPix-rfcp))
            qs = np.sin(rsamp * (rPix-rfcp))
            sw1[0] = (self.elmis[0][0] * pc) + (self.elmis[0][2] * ps)
            sw1[1] = (self.elmis[1][0] * pc) + (self.elmis[1][2] * ps)
            sw1[2] = (self.elmis[2][0] * pc) + (self.elmis[2][2] * ps)
            #print sw1
            sw2[0] = (qc * sw1[0]) - (qs * sw1[1])
            sw2[1] = (qs * sw1[0]) + (qc * sw1[1])
            sw2[2] = sw1[2]
            #print sw2
            sw3[0] = (sx[0] * sw2[0]) + (sy[0] * sw2[1]) + (self.sp[0] * sw2[2])
            sw3[1] = (sx[1] * sw2[0]) + (sy[1] * sw2[1]) + (self.sp[1] * sw2[2])
            sw3[2] = (sx[2] * sw2[0]) + (sy[2] * sw2[1]) + (self.sp[2] * sw2[2])

            # print 'sw1:', sw1
            # print 'sw2:', sw2
            # print 'sw3', sw3

            #print sw3
            sl = self.mg1200(sw3)
            #print 'sl:', sl
            #print 'sat %.5f, %.5f, %.5f:' % (self.sat[0], self.sat[1], self.sat[2])
            def1 = (1.- self.ef) * (1-self.ef)
            #print "def %.9f" % def1
            dda = def1 * ((sl[0] * sl[0]) + (sl[1] * sl[1])) + (sl[2] * sl[2])
            ddb = def1 * ((self.sat[0] * sl[0]) + (self.sat[1] * sl[1])) + (self.sat[2] * sl[2])
            #print def1, self.sat[0], self.sat[1], self.sat[2], self.ea
            ddc = def1 * ((self.sat[0] * self.sat[0]) + (self.sat[1] * self.sat[1]) - (self.ea * self.ea)) + (self.sat[2] * self.sat[2])
            dd = (ddb * ddb) - (dda*ddc)

            # print 'dda %.5f' % dda
            # print 'ddb %.5f' % ddb
            # print 'ddc %.5f' % ddc
            #print 'dd %.5f' % dd

            if dd >= 0 and dda != 0:
                dk1 = (-ddb + np.sqrt(dd))/ dda
                dk2 =  (- ddb - np.sqrt(dd)) / dda
            else:
                rc= 6
                return point

            if abs(dk1) < abs(dk2):
                dk = dk1
            else:
                dk = dk2
            #print 'dk %.5f' % dk
            stn1[0] = self.sat[0] + (dk * sl[0])
            stn1[1] = self.sat[1] + (dk * sl[1])
            stn1[2] = self.sat[2] + (dk * sl[2])
            #print 'stn1', stn1
            dLat = np.arctan(stn1[2] / (def1 * np.sqrt((stn1[0]* stn1[0]) + (stn1[1]*stn1[1])) ))
            #print dLat
            if stn1[0] != 0:
                dLon = math.atan(stn1[1]/stn1[0])
                if stn1[0] < 0 and stn1[1] >= 0:
                    dLon = dLon + self.PI
                if stn1[0] < 0 and stn1[1] < 0:
                    dLon = dLon - self.PI
            else:
                if stn1[1] > 0:
                    dLon = self.hpai
                else:
                    dLon = -self.hpai

            point[0]= dLat * self.crd
            point[1]= dLon * self.crd

        return point

    @staticmethod
    def rint(num):
        """Rounds toward the even number if equidistant"""
        return round(num + (num % 2 - 1 if (num % 1 == 0.5) else 0))

    @staticmethod
    def rint_np(num):
        """Rounds toward the even number if equidistant"""

        t = np.where(num % 1 == 0.5, num % 2 - 1 , 0)
        return np.round(num + t)

    def mg1110_np(self, i, rtim, orbta, mask):

        npa = np.zeros((3, 3))

        #print i, rtim, (rtim-orbta[0][0:6]) / ( orbta[0][1:7]-orbta[0][0:6]).tolist()

        if i != 7:

            # print 'i sat %d' % i
            # print 'rtim %.5f' % rtim
            # print 'orbta[0][%d] %.8f , orbta[0][%d+1] %.8f' % ( i, orbta[0][i], i, orbta[0][i+1])

            delt = (rtim - orbta[0][i]) / (orbta[0][i + 1] - orbta[0][i])


            # delt = np.where(mask, delt, 0)

            # print 'delt', delt

            self.sat[0] = orbta[8][i] + (orbta[8][i + 1] - orbta[8][i]) * delt
            self.sat[1] = orbta[9][i] + (orbta[9][i + 1] - orbta[9][i]) * delt
            self.sat[2] = orbta[10][i] + (orbta[10][i + 1] - orbta[10][i]) * delt


            # print 'setting sat %.5f, %.5f, %.5f:' %  (self.sat[0], self.sat[1], self.sat[2])
            self.sitagt = (orbta[14][i] + (orbta[14][i + 1] - orbta[14][i]) * delt) * self.cdr
            if orbta[14][i + 1] - orbta[14][i] < 0:
                self.sitagt = (orbta[14][i] + (orbta[14][i + 1] - orbta[14][i] + 360.0) * delt) * self.cdr

            self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i]) * delt) * self.cdr

            if orbta[17][i + 1] - orbta[17][i] > 0:
                self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i] - 360.0) * delt) * self.cdr

            self.sundel = (orbta[18][i] + (orbta[18][i + 1] - orbta[18][i]) * delt) * self.cdr

            # compute npa for every point/coodinate?

            npa[0][0] = orbta[19][i]
            npa[1][0] = orbta[20][i]
            npa[2][0] = orbta[21][i]
            npa[0][1] = orbta[22][i]
            npa[1][1] = orbta[23][i]
            npa[2][1] = orbta[24][i]
            npa[0][2] = orbta[25][i]
            npa[1][2] = orbta[26][i]
            npa[2][2] = orbta[27][i]
        return npa

    def mg1100_np(self, rtim):
        """
        I have no clue , but it seem to do some conversion. Im mcidas check the file gms5_nav.for
        for further details.
        :param rtim:
        :return:
        """
        sz = rtim.size

        #beta = delta = attalp = attdel = wkcos = wksin = 0.
        beta = np.zeros((sz))
        delta_orbt = np.zeros((sz))
        delta_atit = np.zeros((sz))
        attalp = np.zeros((sz))
        attdel = np.zeros((sz))
        #wkcos = np.zeros((sz))
        #wksin = np.zeros((sz))

        #att1 = [nan, nan, nan]
        att1 = np.zeros((3, sz))
        #att2 = [nan, nan, nan]
        att2 = np.zeros((3, sz))
        #att3 = [nan, nan, nan]
        att3 = np.zeros((3,sz))
        # npa = [[nan for _i in range(3)] for _ii in range(3)]
        #npa = np.zeros((3, 3))


        """
            This is where the array based code is pretty much different from original code. The backbone of the navigation are the orbital
            and attitude coefficients. One has to be aware that as the scan pattern (swaths) plays a central role in determining the correct navigation

            While in single point mode the computation happens in a loop that exits when the orbit prediction sub-block for the specific pixel is determined.

            There are 8 orbit prediction sub blocks.
        """



        # step 2 determine and calcultae attitude corrections
        atit_values = self.atit[0]
        atit_zones = np.digitize(rtim, atit_values) - 1  # ?? should i extract one
        unique_atit_zones = np.unique(atit_zones)

        for atit_zone in unique_atit_zones:
            m_ = atit_zones == atit_zone
            delta_atit[m_] = (rtim[m_] - atit_values[atit_zone]) / (atit_values[atit_zone + 1] - atit_values[atit_zone])

            attalp[m_] = self.atit[2][atit_zone] + (self.atit[2][atit_zone + 1] - self.atit[2][atit_zone]) * delta_atit[m_]
            attdel[m_] = self.atit[3][atit_zone] + (self.atit[3][atit_zone + 1] - self.atit[3][atit_zone]) * delta_atit[m_]
            beta[m_] = self.atit[4][atit_zone] + (self.atit[4][atit_zone + 1] - self.atit[4][atit_zone]) * delta_atit[m_]
            if self.atit[4][atit_zone + 1] - self.atit[4][atit_zone] > 0:
                beta[m_] = self.atit[4][atit_zone] + (self.atit[4][atit_zone + 1] - self.atit[4][atit_zone] - 360.0 * self.cdr) * delta_atit[m_]

        wkcos = np.cos(attdel)
        att1[0] = np.sin(attdel)
        att1[1] = wkcos * -np.sin(attalp)
        att1[2] = wkcos * np.cos(attalp)
        # from pylab import show, imshow, title, colorbar
        # imshow(attdel.reshape((4800, 2400)))
        # colorbar()
        # title('attdel')
        # show()
        # print wkcos, att1
        '''
		for i in range(9):
			# print 'atit', i, rtim, self.atit[0][i], self.atit[0][i + 1]
			m__ = (rtim > self.atit[0][i]) & (rtim < self.atit[0][i + 1])
			mrtim = rtim[m__]

			if np.all(m__) :

				if mrtim.size != rtim.size:  # what happens here?? the npa seems to be som
					print 'Inconsistent1 results are possible! mrtim.size %s rtim.size %s' % (mrtim.size, rtim.size)

				delt_orbt = (rtim - self.atit[0][i]) / (self.atit[0][i + 1] - self.atit[0][i])
				# delt = np.where(~m__, delt,0)
				attalp = self.atit[2][i] + (self.atit[2][i + 1] - self.atit[2][i]) * delt_orbt
				attdel = self.atit[3][i] + (self.atit[3][i + 1] - self.atit[3][i]) * delt_orbt
				beta = self.atit[4][i] + (self.atit[4][i + 1] - self.atit[4][i]) * delt_orbt
				if self.atit[4][i + 1] - self.atit[4][i] > 0:
					beta = self.atit[4][i] + (self.atit[4][i + 1] - self.atit[4][i] - 360.0 * self.cdr) * delt_orbt
				break
		'''

        # step 1 determine the orbit subblock fir each element in he rtims
        orbt_values = self.orbt1[0]
        orbt_zones = np.digitize(rtim, orbt_values) - 1  # minus one because digitize gives you zones +1
        unique_orb_zones = np.unique(orbt_zones)

        for orbt_zone in unique_orb_zones:
            m = orbt_zones == orbt_zone
            delta_orbt[m] = (rtim[m] - orbt_values[orbt_zone]) / (orbt_values[orbt_zone + 1] - orbt_values[orbt_zone])

            # sate positions
            self.sat[0][m] = self.orbt1[8][orbt_zone] + (self.orbt1[8][orbt_zone + 1] - self.orbt1[8][orbt_zone]) * delta_orbt[m]
            self.sat[1][m] = self.orbt1[9][orbt_zone] + (self.orbt1[9][orbt_zone + 1] - self.orbt1[9][orbt_zone]) * delta_orbt[m]
            self.sat[2][m] = self.orbt1[10][orbt_zone] + (self.orbt1[10][orbt_zone + 1] - self.orbt1[10][orbt_zone]) * delta_orbt[m]

            self.sitagt[m] = (self.orbt1[14][orbt_zone] + (self.orbt1[14][orbt_zone + 1] - self.orbt1[14][orbt_zone]) * delta_orbt[m]) * self.cdr
            if self.orbt1[14][orbt_zone + 1] - self.orbt1[14][orbt_zone] < 0:
                self.sitagt[m] = (self.orbt1[14][orbt_zone] + (self.orbt1[14][orbt_zone + 1] - self.orbt1[14][orbt_zone] + 360.0) * delta_orbt[m]) * self.cdr

            self.sunalp[m] = (self.orbt1[17][orbt_zone] + (self.orbt1[17][orbt_zone + 1] - self.orbt1[17][orbt_zone]) * delta_orbt[m]) * self.cdr

            if self.orbt1[17][orbt_zone + 1] - self.orbt1[17][orbt_zone] > 0:
                self.sunalp[m] = (self.orbt1[17][orbt_zone] + (self.orbt1[17][orbt_zone + 1] - self.orbt1[17][orbt_zone] - 360.0) * delta_orbt[m]) * self.cdr

            self.sundel[m] = (self.orbt1[18][orbt_zone] + (self.orbt1[18][orbt_zone + 1] - self.orbt1[18][orbt_zone]) *delta_orbt[m]) * self.cdr
            '''
			npa[0][0] = self.orbt1[19][orbt_zone]
			npa[1][0] = self.orbt1[20][orbt_zone]
			npa[2][0] = self.orbt1[21][orbt_zone]
			npa[0][1] = self.orbt1[22][orbt_zone]
			npa[1][1] = self.orbt1[23][orbt_zone]
			npa[2][1] = self.orbt1[24][orbt_zone]
			npa[0][2] = self.orbt1[25][orbt_zone]
			npa[1][2] = self.orbt1[26][orbt_zone]
			npa[2][2] = self.orbt1[27][orbt_zone]
			'''
            # just replace directly, not reason to use npa. for this the atit was moved before the orbt computation, as opposed to the java code to avoid looping again through orbtila zones
            att2[0][m] = (self.orbt1[19][orbt_zone] * att1[0][m]) + (self.orbt1[22][orbt_zone] * att1[1][m]) + (self.orbt1[25][orbt_zone] * att1[2][m])
            att2[1][m] = (self.orbt1[20][orbt_zone] * att1[0][m]) + (self.orbt1[23][orbt_zone] * att1[1][m]) + (self.orbt1[26][orbt_zone] * att1[2][m])
            att2[2][m] = (self.orbt1[21][orbt_zone] * att1[0][m]) + (self.orbt1[24][orbt_zone] * att1[1][m]) + (self.orbt1[27][orbt_zone] * att1[2][m])




            # from pylab import show, imshow, title, colorbar
        # imshow(rtim.reshape((4800, 4200)))
        # colorbar()
        # title('sitagt')
        # show()


        '''
        for i in range(7):
            m_ = (rtim > self.orbt1[0][i]) & (rtim < self.orbt1[0][i + 1])
            # print i, rtim,  self.orbt1[0][i],  self.orbt1[0][i+1], m_
            # from pylab import show, imshow, title, colorbar
            # imshow(m_.reshape((3900, 4200)))
            # colorbar()
            # title('m_%s' % i)
            # show()
            mrtim = rtim[m_]
            npa = self.mg1110_np(i, rtim, self.orbt1, m_)
            print i, mrtim.size>0, npa.mean()
            if np.all(m_):
                if mrtim.size != rtim.size:  # what happens here?? the npa seems to be som
                    print 'Inconsistent results are possible! mrtim.size %s rtim.size %s, orbt1[0] size %s for i %s' % (mrtim.size, rtim.size, self.orbt1[0].size, i)

                npa = self.mg1110_np(i, rtim, self.orbt1, m_)
                print i,  npa
                # print m_npa.shape
                break
        '''




        # print att1
        # print npa[0][0], npa[0][1], npa[0][2]
        wksin = np.sin(self.sitagt)
        wkcos = np.cos(self.sitagt)

        att3[0] = (wkcos * att2[0]) + (wksin * att2[1])
        att3[1] = (-wksin * att2[0]) + (wkcos * att2[1])
        att3[2] = att2[2]

        self.sp = self.mg1200_np(att3)

        wkcos = np.cos(self.sundel)
        self.ss[0] = wkcos * np.cos(self.sunalp)
        self.ss[1] = wkcos * np.sin(self.sunalp)
        self.ss[2] = np.sin(self.sundel)

        return beta


    def mg1200_np(self, vec):

        vecu = [nan, nan, nan]

        rv1 = (vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2])
        if np.all(rv1 == 0):
            print ('issue')
            return vecu
        rv2 = np.sqrt(rv1)
        vecu[0] = vec[0] / rv2
        vecu[1] = vec[1] / rv2
        vecu[2] = vec[2] / rv2
        return vecu

    def mg1220_np(self, va, vb):

        vc = [nan, nan, nan]

        vc[0] = (va[1] * vb[2]) - (va[2] * vb[1])
        vc[1] = (va[2] * vb[0]) - (va[0] * vb[2])
        vc[2] = (va[0] * vb[1]) - (va[1] * vb[0])

        vd = self.mg1200_np(vc)

        return vd

    def mg1230_np(self, va, vb):

        as1 = (va[0] * vb[0]) + (va[1] * vb[1]) + (va[2] * vb[2])
        as2 = ((va[0] * va[0]) + (va[1] * va[1]) + (va[2] * va[2])) * (
            (vb[0] * vb[0]) + (vb[1] * vb[1]) + (vb[2] * vb[2]))

        if np.all(as2 == 0):
            return as2
        temp = as1 / np.sqrt(as2)
        tmask = (temp > 1) & (temp - 1. < 0.000001)
        temp = np.where(tmask, 1, temp)

        return np.arccos(temp)

    def mg1110(self, i, rtim, orbta):
        """

        :param i:
        :param rtim:
        :param orbta:
        :return: 3 by 3 array
        """

        npa  = [[nan for _i in range(3)] for _i in range(3)]
        aa = []
        for j in range(6):
            aa.append( (rtim - orbta[0][j]) / (orbta[0][j+1] - orbta[0][j]))
        if i != 7:
            #print 'i sat %d' % i
            #print 'rtim %.5f' % rtim
            #print 'orbta[0][%d] %.8f , orbta[0][%d+1] %.8f' % ( i, orbta[0][i], i, orbta[0][i+1])
            delt = (rtim - orbta[0][i]) / (orbta[0][i+1] - orbta[0][i])
            #print 'classic delt ', delt
            self.sat[0] = orbta[8][i] + (orbta[8][i+1] - orbta[8][i]) * delt
            self.sat[1] = orbta[9][i] + (orbta[9][i+1] - orbta[9][i]) * delt
            self.sat[2] = orbta[10][i] + (orbta[10][i+1] - orbta[10][i]) * delt
            #print 'setting sat %.5f, %.5f, %.5f:' %  (self.sat[0], self.sat[1], self.sat[2])
            self.sitagt = (orbta[14][i] + (orbta[14][i+1] - orbta[14][i]) * delt) * self.cdr
            if orbta[14][i+1] - orbta[14][i] < 0:
                self.sitagt = (orbta[14][i] + (orbta[14][i + 1] - orbta[14][i] + 360.0) * delt) * self.cdr

            self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i]) * delt) * self.cdr

            if orbta[17][i + 1] - orbta[17][i] > 0:
                self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i] - 360.0) * delt) * self.cdr

            self.sundel = (orbta[18][i] + (orbta[18][i+1] - orbta[18][i]) * delt) * self.cdr

            npa[0][0] = orbta[19][i]
            npa[1][0] = orbta[20][i]
            npa[2][0] = orbta[21][i]
            npa[0][1] = orbta[22][i]
            npa[1][1] = orbta[23][i]
            npa[2][1] = orbta[24][i]
            npa[0][2] = orbta[25][i]
            npa[1][2] = orbta[26][i]
            npa[2][2] = orbta[27][i]

        return npa

    def mg1200(self, vec):

        vecu = [nan, nan, nan]
        rv1 = (vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2])
        if rv1 == 0:
            return vecu
        rv2 = np.sqrt(rv1)
        vecu[0] = vec[0] / rv2
        vecu[1] = vec[1] / rv2
        vecu[2] = vec[2] / rv2
        return vecu

    def mg1100(self, rtim):
        """
        I have no clue , but it seem to do some conversion. Im mcidas check the file gms5_nav.for
        for further details.
        :param rtim:
        :return:
        """
        beta = delta = attalp = attdel = wkcos = wksin = 0.
        att1 = [nan, nan, nan]
        att2 = [nan, nan, nan]
        att3 = [nan, nan, nan]
        npa  = [[nan for _i in range(3)] for _ii in range(3)]

        #print rtim
        for i in range(7):
            #print i, rtim,  self.orbt1[0][i],  self.orbt1[0][i+1], rtim > self.orbt1[0][i] and rtim < self.orbt1[0][i+1]

            if rtim > self.orbt1[0][i] and rtim < self.orbt1[0][i+1]:
                #print 'computing npa', i
                npa = self.mg1110(i, rtim, self.orbt1)
                break
        #print 'npa classic', npa
        for i in range(9):
            #print 'atit', i, rtim, self.atit[0][i], self.atit[0][i + 1]
            if rtim >= self.atit[0][i] and rtim < self.atit[0][i+1]:
                delt = (rtim - self.atit[0][i]) / (self.atit[0][i + 1] - self.atit[0][i])
                attalp = self.atit[2][i] + (self.atit[2][i + 1] - self.atit[2][i]) * delt
                attdel = self.atit[3][i] + (self.atit[3][i + 1] - self.atit[3][i]) * delt
                beta = self.atit[4][i] + (self.atit[4][i + 1] - self.atit[4][i]) * delt
                if self.atit[4][i + 1] - self.atit[4][i] > 0:
                    beta = self.atit[4][i] + (self.atit[4][i + 1] - self.atit[4][i] - 360.0 * self.cdr) * delt
                break
        #print beta
        wkcos = np.cos(attdel)
        att1[0] = np.sin(attdel)
        att1[1] = wkcos * -np.sin(attalp)
        att1[2] =  wkcos * np.cos(attalp)
        #print wkcos, att1

        att2[0] = (npa[0][0] * att1[0]) + (npa[0][1] * att1[1]) + (npa[0][2] * att1[2])
        att2[1] = (npa[1][0] * att1[0]) + (npa[1][1] * att1[1]) + (npa[1][2] * att1[2])
        att2[2] = (npa[2][0] * att1[0]) + (npa[2][1] * att1[1]) + (npa[2][2] * att1[2])
        #print att1
        #print npa[0][0], npa[0][1], npa[0][2]
        wksin = np.sin(self.sitagt)
        wkcos = np.cos(self.sitagt)

        att3[0] = (wkcos * att2[0]) + (wksin * att2[1])
        att3[1] = (-wksin * att2[0]) + (wkcos * att2[1])
        att3[2] = att2[2]
        self.sp = self.mg1200(att3)

        wkcos = np.cos(self.sundel)
        self.ss[0] = wkcos * np.cos(self.sunalp)
        self.ss[1] = wkcos * np.sin(self.sunalp)
        self.ss[2] = np.sin(self.sundel)

        return beta


    def mg1220(self, va, vb):
        vc = [nan, nan, nan]

        vc[0] = (va[1] * vb[2]) - (va[2] * vb[1])
        vc[1] = (va[2] * vb[0]) - (va[0] * vb[2])
        vc[2] = (va[0] * vb[1]) - (va[1] * vb[0])

        vd = self.mg1200(vc)

        return vd


    def mg1210(self, va, vb):

        vc = [nan, nan, nan]
        vc[0] = (va[1] * vb[2]) - (va[2] * vb[1])
        vc[1] = (va[2] * vb[0]) - (va[0] * vb[2])
        vc[2] = (va[0] * vb[1]) - (va[1] * vb[0])
        return vc

    def mg1230(self, va, vb):

        as1 = (va[0] * vb[0]) + (va[1] * vb[1]) + (va[2] * vb[2])
        as2 = ((va[0] * va[0]) + (va[1] * va[1]) + (va[2] * va[2])) * ((vb[0] * vb[0]) + (vb[1] * vb[1]) + (vb[2] * vb[2]))

        if as2 ==0:
            return as2
        temp = as1/np.sqrt(as2)
        if temp > 1 and temp-1. < 0.000001:
            temp = 1
        return np.arccos(temp)



class ABINNav(AreaNav):

    def __init__(self, navblock=None, auxblock=None, **kwargs):
        self.loff = navblock[1] / 100000000.0;
        self.coff = navblock[2] / 100000000.0;
        self.lfac = navblock[3] / 100000000.0;
        self.cfac = navblock[4] / 100000000.0;
        self.plon = navblock[5] / 10.0;
        self.bres = navblock[6];




