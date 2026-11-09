import numpy as np

from pyarea.navigation import hrit_navigation as hnav
from pyarea.navigation import proj4_navigation as pnav


def test_chunk_intersection(af=None, points=None, lon0=None, gt=None):
    """
    test if a chunk of image represented by four corners intersects an AreaFile
    :param af: str, path tothe pyarea file
    :param points: iter of 4 points (line, column) in range 0-number of lines in the image, 0-number of columns in the image
    :param lon0: SSp longitude of the image (geos)
    :param gt: geotransform
    :return:
    """


    nv = 1e30
    nav = af.navigation
    top, bottom, left, right = af.areabox
    yres, xres = af.resolution
    top*=yres
    bottom*=yres
    left*=xres
    right*=xres
    for e in points:
        l, c = e
        lat, lon = pnav.lc2ll(l=l, c=c, ssp=lon0, gt=gt)
        if lat == nv or lon == nv:
            continue
        else:
            area_cs, area_ls  = nav.toLinEle(latlon=[[lat], [lon]])
            area_l = area_ls[0]
            area_c = area_cs[0]

            if (top <= area_l < bottom) and (left <=area_c < right):
                return True
    return False





def reproject_area2geos(af=None, out_resolution=2000, out_ssp=None, use_proj_navigation=False, reproject_in_chunks=False, return_gdalinfo=False):
    """
    Reproject an McIDAS pyarea file featuring GVAR navigation to geostationary projection
    :param in_area_file: pyarea file obj from pyarea lib
    :param out_resolution: int, the resoution of the geostationary reprojected data (meters)
    :param out_ssp: number (usually float), the SSP of the output geostationary file, if not supplied it is taken and rounded from the Area file
    :param use_proj_navigation: bool, if true, proj4 based navigation will be used
    :param reproject_in_chunks: bool, if true a ckunk based approach will be employed to avoid memory issues
    :param return_gdalinfo, bool, if true then geotransform nad projection string are returned as a tuple
    :return: 2D numpy array representing the reprojected data

    Notes:
        The reprojection  is performed  by creating  index grids on Y and X axis in geostationary
        projection and then reprojecting these grids to lat/lon index grids. These can be done using proj4 or HRIT based navigation
        The geographical index grids are then transformed into GVAR index grids (Area file coordinates not Image coordinates)
        For a detailed account consult https://www.ssec.wisc.edu/mcidas/doc/learn_guide/2015/sat-1.html
        The pyarea file index grids are the used to fetch the pixels from input pyarea file and reproject
        using nearest neighbour to geostationary projection

    """
    try:



        navigation = af.navigation  # GVAR
        data = af.data.squeeze()
        # imshow(data)
        # show()
        anl, anc = data.shape  # dimensions of input pyarea file data
        # the dimensions of the dataset
        offset = hnav.compute_offset(out_resolution)
        scale_fact = hnav.compute_scale_factor(out_resolution)

        nl = nc = 2 * offset  # the output image has same size in each dimension
        # allocate output
        out_geos_data = np.zeros((nl, nc), dtype=data.dtype)


        # SS longitude
        lon0 = navigation.subpoint[1] if out_ssp is None else out_ssp
        lon0 = int(round(lon0))
        # compute GeoTranform
        half_width = offset * out_resolution
        geotr = -half_width, out_resolution, 0, half_width, 0, -out_resolution
        nsegs = 5 # split the full diska space into 5X5
        seg_len = nl / nsegs

        if use_proj_navigation:
            if not reproject_in_chunks:
                l, c = np.mgrid[0:nl, 0:nc]
                lats, lons = pnav.lc2ll(l=l, c=c, ssp=lon0, gt=geotr)

            else:
                for i in range(nsegs):
                    sl = i * seg_len
                    el = sl + seg_len
                    for j in range(nsegs):
                        sc = j * seg_len
                        ec = sc + seg_len
                        corners = ((sl + 1, sc + 1), (sl + 1, ec + 1), (el + 1, sc + 1), (el + 1, ec + 1))
                        should_continue = test_chunk_intersection(af=af, points=corners, lon0=lon0, gt=geotr)
                        if should_continue:
                            l, c = np.mgrid[sl:el, sc:ec]
                            lats, lons = pnav.lc2ll(l=l, c=c, ssp=lon0, gt=geotr)
                            navigation = af.navigation  # GVAR
                            area_l, area_c = navigation.toLinEle2(lats=lats, lons=lons) # toLineEle2 is fatser than toLineEle
                            # mask the invalid indices
                            lmask = (area_l >= 0) & (area_l < anl)
                            cmask = (area_c >= 0) & (area_c < anc)
                            m = lmask & cmask
                            # reproject
                            marea_l = np.floor(area_l[m]).astype(np.int32)
                            marea_c = np.floor(area_c[m]).astype(np.int32)
                            out_geos_data[sl:el, sc:ec][m] = data[marea_l, marea_c]

                if return_gdalinfo:
                    proj4_str = '+proj=geos +a=6378169.0 +b=6356583.8 +h=35785831.0 +lon_0=%.2f +units=m' % lon0
                    gdal_info = geotr, proj4_str
                    return out_geos_data, gdal_info
                return out_geos_data

        else:  # HRIT based reprojection
            if not reproject_in_chunks:
                lats, lons = hnav.lc2ll(c=np.arange(1, nl + 1), l=np.arange(1, nc + 1), lon0=lon0, cfac=scale_fact,
                                        lfac=scale_fact, coff=offset, loff=offset)
            else:
                for i in range(nsegs):
                    sl = i * seg_len
                    el = sl + seg_len
                    for j in range(nsegs):
                        sc = j * seg_len
                        ec = sc + seg_len
                        corners = ((sl + 1, sc + 1), (sl + 1, ec + 1), (el + 1, sc + 1), (el + 1, ec + 1))

                        should_continue = test_chunk_intersection(af=af, points=corners, lon0=lon0, gt=geotr)
                        if should_continue:
                            l, c = np.ogrid[sl + 1:el + 1, sc + 1:ec + 1]
                            lats, lons = hnav.lc2ll(c=c.squeeze(), l=l.squeeze(), lon0=lon0, cfac=scale_fact,
                                                    lfac=scale_fact, coff=offset, loff=offset)
                            navigation = af.navigation  # GVAR
                            area_l, area_c = navigation.toLinEle2(lats=lats, lons=lons)
                            # mask the invalid indices
                            lmask = (area_l >= 0) & (area_l < anl)
                            cmask = (area_c >= 0) & (area_c < anc)
                            m = lmask & cmask
                            # reproject
                            marea_l = np.floor(area_l[m]).astype(np.int32)
                            marea_c = np.floor(area_c[m]).astype(np.int32)
                            out_geos_data[sl:el, sc:ec][m] = data[marea_l, marea_c]

                if return_gdalinfo:
                    proj4_str = '+proj=geos +a=6378169.0 +b=6356583.8 +h=35785831.0 +lon_0=%.2f +units=m' % lon0
                    gdal_info = geotr, proj4_str
                    return out_geos_data, gdal_info
                return out_geos_data


        # continue as now we have the lats and lons and the code is the same
        area_l, area_c = navigation.toLinEle2(lats=lats, lons=lons)



        # mask the invalid indices
        lmask = (area_l >= 0) & (area_l < anl)
        cmask = (area_c >= 0) & (area_c < anc)

        m = lmask & cmask

        # reproject
        marea_l = np.floor(area_l[m]).astype(np.int32)
        marea_c = np.floor(area_c[m]).astype(np.int32)

        out_geos_data[m] = data[marea_l, marea_c]
        if return_gdalinfo:
            proj4_str = '+proj=geos +a=6378169.0 +b=6356583.8 +h=35785831.0 +lon_0=%.2f +units=m' % lon0
            gdal_info = geotr, proj4_str
            return out_geos_data, gdal_info
        return out_geos_data
    finally:
        af.close() # the close functin does nothing in case the pyarea file is not opened form a file, perhaps it should clear the memory. However python  does not guarantee that "del" operator does the job of dealocationg when invoked
