from io import BytesIO
from pyarea.directory import area_directory
from pyarea.nav import AreaNav
from pyarea.utils import INT32SIZE
import struct
import numpy as np
import os
import logging
_, name = os.path.split(__file__)
logger = logging.getLogger(name)

class AreaFile(BytesIO):
    INTEGER_FORMAT = 'i'

    def __init__(self, source=None, lazy=True):

        if not isinstance(source, bytes):
            self.is_remote = False
            self.source = source
            with open(source, 'rb') as src:
                d = src.read()
                super().__init__(d)

        else:
            self.is_remote = True
            self.source = None
            super().__init__(source)

        self.sz  = len(self.getbuffer())

        self.directory = area_directory(self.read(256))
        self.adir_bytes = self.directory.dir_bytes


        #as per http://www.ssec.wisc.edu/mcidas/doc/prog_man/current/formats-1.html#34971
        #after the directory bytes the block are, in order
        #nav, cal, aux, data, comments

        self.navigation_offset = self.calibration_offset = self.aux_offset = None
        self.navigation_offset = self.directory.nav_block_offset
        self.calibration_offset = self.directory.cal_block_offset
        self.aux_offset = self.directory.aux_block_offset
        self.data_offset = self.directory.data_block_offset




        self.line_prefix_len = self.directory.prefix_doc_length + self.directory.prefix_cal_length + self.directory.prefix_band_length
        if self.directory.validity_code != 0:
            self.line_prefix_len +=4
        # print(self.line_prefix_len != self.directory.line_prefix_length)
        if self.line_prefix_len != self.directory.line_prefix_length:
            raise Exception('Invalid line prefix length in AREA file.')


        #start from the back of the file
        if self.data_offset>0:
            self.nav_size = self.data_offset - self.navigation_offset
            self.cal_size = self.data_offset - self.calibration_offset
            self.aux_size = self.data_offset - self.aux_offset


        if self.aux_offset>0 and self.directory.aux_block_length>0:
            self.nav_size = self.aux_offset - self.nav_size
            self.cal_size = self.aux_offset - self.calibration_offset


        if self.calibration_offset>0:
            self.nav_size = self.calibration_offset - self.navigation_offset


        #read
        self.nav_bytes = self.cal_bytes = self.aux_bytes = None
        self.has_navigation = self.has_calibration = self.has_auxiliary = False
        if self.navigation_offset>0 and self.nav_size > 0:
            self.nav_bytes = self.read(self.nav_size)
            self.nav = struct.unpack(f'{self.directory.byte_order}{self.nav_size//INT32SIZE}{self.INTEGER_FORMAT}', self.nav_bytes)
            self.has_navigation = True
        if self.calibration_offset>0 and self.cal_size>0:
            self.cal_bytes = self.read(self.cal_size)
            self.has_calibration = True
            self.cal = struct.unpack(f'{self.directory.byte_order}{self.cal_size//INT32SIZE}{self.INTEGER_FORMAT}', self.cal_bytes)
        if self.aux_offset> 0 and self.aux_size>0:
            self.aux_bytes = self.read(self.aux_size)
            self.has_auxiliary = True
            self.aux = struct.unpack(f'{self.directory.byte_order}{self.aux_size//INT32SIZE}{self.INTEGER_FORMAT}', self.aux_bytes)



        logger.debug(f'Has navigation: {self.has_navigation},  has calibration: {self.has_calibration}, Has auxiliary: {self.has_auxiliary}')
        self.areaNav = None

        #setup some info that will be accessed while reading data etc

        self.datawidth = self.directory.bytes_per_element
        self.num_bands = self.directory.spectral_band_count
        self.lines = self.directory.lines
        self.elements = self.directory.elements

        self.line_data_len = self.num_bands * self.elements * self.datawidth
        self.line_len = self.line_prefix_len + self.line_data_len
        self.data_block_len =self.lines * self.line_len

        self.prefix_len = self.line_prefix_len//self.datawidth
        self.data_len = self.data_block_len//self.datawidth



        if lazy:
            self.data = None
        else:

            # read the data
            self.data = self.__readdata__()



    def __datamask__(self):
        """
        Although an pyarea file can have multiple bands they have to have the same resolution, therefore have same number of lines and
        columns. Becaus eof this the mask, is actually the same for all bands.
        As a result the resulted mask is 2D not 3D
        :return:
        """

        mask = np.ones(( self.lines, self.num_bands*self.elements + self.prefix_len), dtype=bool)
        #all at once, no loops
        mask[ :,:self.prefix_len] = False
        # pylab.imshow(mask)
        # pylab.show()
        return mask

    def __readdata__(self):

        mask = self.__datamask__()
        dtp = f'{self.directory.byte_order}u{self.datawidth}'
        darray = np.zeros(shape=(self.num_bands, self.lines, self.elements), dtype=dtp)
        self.seek(0) #because np.buffer has an offset arg
        self.data_block = np.frombuffer(self.read(), dtype=dtp, count=self.data_len, offset=self.data_offset)


        temp_data_block = self.data_block.reshape(self.lines, (self.num_bands*self.elements)+self.prefix_len)
        temp_masked_data_block = temp_data_block[mask].reshape(self.lines, (self.num_bands*self.elements))

        for bi in range(self.num_bands):
            darray[bi,:] = temp_masked_data_block[:,bi::self.num_bands]

        return darray



    def __getattribute__(self, name):
        if name == 'data':
            value = object.__getattribute__(self, name)
            if value is None:
                self.data = self.__readdata__()
                return self.data
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)

    @property
    def istim(self):
        """
        Computes an approximation of scan time in miliseconds of a swath if the image was produced by
        GOES 8-15 instruments
        :return:
        """
        sens = self.areaDirectory.sensor
        if sens.startswith('GOES'):

            try:
                instr = sens[-2:]
                iinstr = int(instr)
                if iinstr >= 8:
                    return int(round((self.origNumElements * self.resolution[1]) * 954) / 20834.)
            except Exception as e:
                pass

    @property
    def imgbox(self):
        """
            Property
            Computes and returns the bounding box as a tuple (stline, endline, stelem, endelem)
        """
        start_line = self.directory.line_ul
        end_line = start_line + self.lines
        start_elem = self.directory.element_ul
        end_elem = start_elem + self.elements
        return start_line, end_line, start_elem, end_elem

    @property
    def resolution(self):
        return self.directory.line_res, self.directory.element_res

    @property
    def size(self):
        return self.lines, self.elements

    @property
    def npdtype(self):
        return f'{self.directory.byte_order}u{self.datawidth}'


    @property
    def file_name(self):
        """
            Property
            returns the file name of the AREA file
        """
        if self.source:
            _, name = os.path.split(self.source)
            return name


    @property
    def navigation(self):
        if self.areaNav is None:
            self.areaNav = AreaNav(navblock=self.nav, byte_order=self.directory.byte_order, int_format=self.INTEGER_FORMAT)
            self.areaNav.setImageStart(self.directory.line_ul, self.directory.element_ul)
            self.areaNav.setRes(*self.resolution)
            self.areaNav.setMag(1,1)
            self.areaNav.setStart(0,0)


        return self.areaNav

    def __str__(self):
        if self.source:
            ads = f'AREA File: {self.source} \n'
        else:
            ads = '\n'
        ads += str(self.directory)
        return ads

    @property
    def nominal_datetime(self):
        return self.directory.nominal_time
    @property
    def scan_datetime(self):
        return self.directory.start_time
    @property
    def creation_datetime(self):
        return self.directory.creation_time

    def save(self, to_file_path=None, compress_level=None):
        """
        Saves the AreFile to disk. Optionally supply  a sring representing the coverage of  the AreaFile to be incorporated in the name
        Optiaonally compress to gzip
        :param folder: str, full path to folder where data will be saved
        :param coverage: str, the coverage ex SH, NH for goes images
        :param compress_level: int, if provided  the image will be compressed using gzip and specified compresslevel
        :return: str, the full path of the saved file
        """

        assert os.path.isabs(to_file_path), 'to_file_path arg %s is not an absolute path' % to_file_path
        try:
            import gzip
            with open(to_file_path, 'wb') as aft:
                aft.seek(0)
                aft.write(self.adir_bytes)
                if self.navigation_offset>0 and self.nav_size>0:
                    aft.seek(self.navigation_offset)
                    aft.write(self.nav_bytes)
                if self.calibration_offset > 0 and self.cal_size > 0:
                    aft.seek(self.calibration_offset)
                    aft.write(self.cal_bytes)
                if self.aux_offset > 0 and self.aux_size > 0:
                    aft.seek(self.aux_offset)
                    aft.write(self.aux_bytes)
                #if theer is not datLoc there si not AreaFile
                aft.seek(self.data_offset)
                #TODO line prefix bytes

                for bi in range(self.num_bands):
                    aft.write(self.data[bi,:,:].tobytes())


            if compress_level is not None:

                afpz = to_file_path + '.z'
                with open(to_file_path, 'rb') as f_in, gzip.open(afpz, 'wb', compresslevel=compress_level) as f_out:
                    for l in f_in:
                        f_out.write(l)
                os.remove(to_file_path)
                return afpz

            return  to_file_path
        except Exception as e:

            raise

    @property
    def imc_flip(self):
        """
        Computes

        Line Prefix COntent
              Validity Code: 4 bytes
              Documentation: 76 bytes. The documentation region
            consists of the following:
            - Block Header CRC: 2 bytes. Last three bits only;
            bit is set if Block Header copy is good. This data
            is usually 00,07.
            - Scan Status: 4 bytes (OGE Table 3-6)
            - Year, Day, Time from Block 0: 8 bytes
            - Block Header: 30 bytes (OGE Table 3-5)
              Bytes 17-24 now contain the time the block was disseminated from
              the ground station.
            - Additional Line Documentation: 32 bytes. 16 10-
            bit fields, right-justified. (OGE Table 3-7)
            The 6 bit field on the left hand side is not zero filled.
            To get the 10 bit field a logical AND against 03FF (hex) must be used.

        :return:
        """
        if self.directory.source_type.decode() in ['GVAR', 'VISR']:

            if self.directory.line_prefix_length == 0:

                raise Exception('Cannot extract imc anf flip flags  because the image lines have no  documentation ' )
        else:
            raise  Exception('The Area file was not produced from a GVAR stream. The IMC anf FLIP flags do not apply on this case.')

        import ctypes as C

        if self.directory.byte_order == '>':
            klass = C.BigEndianStructure
        else:
            klass = C.LittleEndianStructure

        class ISCAN(klass):

            """
                ' 0 Frame Start',
                ' 1 Frame End',
                ' 2 Frame break ( lines lost )',
                ' 3 Line break ( Pixels lost )',
                ' 4 Priority 1 frame data ',
                ' 5 Priority 2 frame data ',
                ' 6 East-to-west scan',
                ' 7 South-to-north scan',
                ' 8 IMC active',
                ' 9 Lost header block',
                '10 Lost trailer block',
                '11 Lost telemetry data',
                '12 (Star sense) time break',
                '13 Side 2 (secondary) active',
                '14 Visible normalization active',
                '15 IR calibration active',
                '16 yaw-flipped mode (GOES-10)',
                '17 IR detector 1 data not valid',
                '18 IR detector 2 data not valid',
                '19 IR detector 3 data not valid',
                '20 IR detector 4 data not valid',
                '21 IR detector 5 data not valid',
                '22 IR detector 6 data not valid',
                '23 IR detector 7 data not valid',
                '24 Visible detector 1 data not valid',
                '25 Visible detector 2 data not valid',
                '26 Visible detector 3 data not valid',
                '27 Visible detector 4 data not valid',
                '28 Visible detector 5 data not valid',
                '29 Visible detector 6 data not valid',
                '30 Visible detector 7 data not valid',
                '31 Visible detector 8 data not valid'
            """
            _pack_ = 1

            _fields_ = (('frame_start', C.c_uint8, 1),  # 0
                        ('frame_end', C.c_uint8, 1),  # 1
                        ('frame_break', C.c_uint8, 1),  # 2
                        ('line_break', C.c_uint8, 1),  # 3
                        ('priority1', C.c_uint8, 1),  # 4
                        ('priority2', C.c_uint8, 1),  # 5
                        ('west_to_east', C.c_uint8, 1),  # 6
                        ('north_to_south', C.c_uint8, 1),  # 7
                        ('imc_active', C.c_uint8, 1),  # 8
                        ('header_lost', C.c_uint8, 1),  # 9
                        ('trailer_lost', C.c_uint8, 1),  # 10
                        ('telemetry_lost', C.c_uint8, 1),  # 11
                        ('time_break', C.c_uint8, 1),  # 12
                        ('side2', C.c_uint8, 1),  # 13
                        ('vis_normalization', C.c_uint8, 1),  # 14
                        ('ir_calibration', C.c_uint8, 1),  # 15
                        ('yaw_flip', C.c_uint8, 1),  # 16
                        ('irdet1_invalid', C.c_uint8, 1), ('irdet2_invalid', C.c_uint8, 1),
                        ('irdet3_invalid', C.c_uint8, 1), ('irdet4_invalid', C.c_uint8, 1),
                        ('irdet5_invalid', C.c_uint8, 1), ('irdet6_invalid', C.c_uint8, 1),
                        ('irdet7_invalid', C.c_uint8, 1), ('visdet1_invalid', C.c_uint8, 1),
                        ('visdet2_invalid', C.c_uint8, 1), ('visdet3_invalid', C.c_uint8, 1),
                        ('visdet4_invalid', C.c_uint8, 1), ('visdet5_invalid', C.c_uint8, 1),
                        ('visdet6_invalid', C.c_uint8, 1), ('visdet7_invalid', C.c_uint8, 1),
                        ('visdet8_invalid', C.c_uint8, 1),)

            def bstring(self):
                return ''.join([str(getattr(self, e[0])) for e in self._fields_])


        '''
        line prefix length = doc + cal + band + 4 (if valcode is present)
        line data length = nbands*nele*nbytes
        line length = line prefix length + line data length
        data block length = nlines * line length
        '''

        try:


            mask = self.__datamask__().ravel()
            data = self.data


            # #
            # for b in range(self.num_bands):

            # t he line prefix is 4 bytes, so we view it as 1 byte
            prfx1 = self.data_block[~mask].reshape(self.lines, self.prefix_len).view('u1')
            # some story

            # valid = prfx1.view('%si4' % self.BYTE_ORDER)

            if self.directory.validity_code:  # this is the validity code, if its value is present in yhe header
                # one vcould also use byte position 23 or 24 , to chcek valid lines using block0 header
                # valid_lines_mask = valid[:,0] == self.valcode
                # valid_lines = prfx1[:,INT32SIZE:][valid_lines_mask]
                lines = prfx1[:, INT32SIZE:]
            else:
                lines = prfx1
            valid_lines_mask = lines[:, 22].view(bool)
            valid_lines = lines[valid_lines_mask]
            if not lines.size > 0:
                raise Exception('The Area files dodo not have any valid lines as per block0 data form the line prefix')
            aline = valid_lines[0]
            astatus = ISCAN.from_buffer_copy(aline[2:6])
            return astatus.imc_active, astatus.yaw_flip  # print aline  # BO = self.BYTE_ORDER  # HEADER = [('BlockID', 'B'), ('WordSize', 'B'), ('WordCount', 'H'), ('ProductID', 'H'),  #           ('RepeatFlag', 'B'), ('VersionNumber', 'B'), ('DataValid', 'B'), ('AsciiBin', 'B'),  #           ('SPS_ID', 'B'), ('RangeWord', 'B'), ('BlockCount', 'H'), ('Spares_1', '2B'),  #           ('SPS_Time', '8B'), ('Spares_2', '4B'), ('CRC', 'H')]  #  # HEADER = [(e[0], BO + e[1]) for e in HEADER]  # for i in range(10):  #     # h = np.frombuffer(valid_lines[i,:],dtype=HEADER, count=1, offset=14)  #     # print h['RangeWord'].item()>>4  #     from pygvar.classes import ISCAN  #     b = valid_lines[i, 2:6]  #     a = ISCAN.from_buffer_copy(b)  #     print a.imc_active, a.yaw_flip

            # first_line_scandatetime = bcdtime2datetime(bytes=valid_lines[0, 6:14], byte_order=self.BYTE_ORDER)  # last_line_scandatetime = bcdtime2datetime(bytes=valid_lines[-1, 6:14], byte_order=self.BYTE_ORDER)

        except Exception as e:
                raise Exception(f'Failed to fetch IMC and FLIP flags from {self.__class__.__name__} object, Error is {e}' )
    @property
    def resolution(self):
        return self.directory.line_res, self.directory.element_res

if __name__ == '__main__':

    #import pylab
    logger = logging.getLogger()
    logging.basicConfig()
    logger.setLevel('DEBUG')
    afp = '/media/d/data/satellite_goes_r/adde/incoming/0000000001.OR_ABI-L1b-RadF-M3C07_G16_s2018166103041000000_e2018166104118000000_c2018166104122000000.area'
    afpz = '/media/d/data/satellite_goes_r/adde/incoming/0000000001.OR_ABI-L1b-RadF-M3C13_G16_s2018226120048000000_e2018226121125000000_c2018226121129000000.area.z'
    #afp = '/home/jano/sw/06091408 GOES-13 Imager Complete Thu 09-Jun-2016 1408.VIN'

    with AreaFile(source=afp, lazy=True) as a:
        logger.info(a)
    # import gzip
    # with gzip.open(afpz) as cf:
    #     with AreaFile(source=cf.read(), lazy=True) as a:
    #         logger.info(a)
    #         band = a.directory.bands[0]
    #         for band in range(1,17):
    #             band_chunk_start = 1 + (band - 1) * 18
    #             # band_chunki_start = ninstr_bytes + (band-1)* n_calib_elements*4
    #             band_chunk_end = band_chunk_start + 18
    #
    #             calib_info = a.cal[band_chunk_start:band_chunk_end]
    #             print (dict(zip(range(1,len(calib_info)+1),calib_info)))

            # for b in a.data:
            #     pylab.imshow(b)
            #     pylab.show()









