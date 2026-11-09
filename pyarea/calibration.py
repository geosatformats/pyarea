__author__ = 'jano'

import numpy as np

from pyarea import utils
from pyarea.sensors import get_sensor


class Calibrator(object):

    CAL_NONE = -1
    CAL_MIN = 1
    CAL_RAW = 1
    CAL_RAD = 2
    CAL_ALB = 3
    CAL_TEMP = 4
    CAL_BRIT = 5
    CAL_MAX = 5
    def setCalType(self, calType):
        raise Exception( 'Function selCalType is abstract')

    def calibrate_pixel(self, inputPixel, band, calTypeOut):
        raise Exception('Function calibrate_pixel is abstract')

    def calibrate_array(self, input_array, band, calTypeOut):
        raise Exception( 'Function calibrate_array is abstract')


class DefaultCalibrator(Calibrator):
    curCalType = 0

    def setCalType(self, calType):
        if calType < Calibrator.CAL_MIN or calType > Calibrator.CAL_MAX:
            return -1
        self.curCalType = calType

    def calibrate_pixel(self, inputPixel, band, calTypeOut ):
        """
            Calibrate one pixel
        """
        if self.curCalType in [self.CAL_RAW, self.CAL_RAD, self.CAL_ALB, self.CAL_TEMP, self.CAL_BRIT]:
            return inputPixel
        else:
            return 0.


    def calibrate_array(self, input_array, band, calTypeOut):

        out = list()
        for e in input_array:
            out.append(self.calibrate_pixel(e,band, calTypeOut))
        return  out


# Planck constants of the GOES-8 to 15 imagers (5 bands, band 1 is the visible and unused) and sounders (18 bands),
# keyed by satellite number and then instrument. FK1, FK2, TC1 and TC2 are the McIDAS/VisAD constants
# (edu.wisc.ssec.mcidas.CalibratorGvarG*) for GOES-12 to 15. VisAD has no GOES-11 class, so GOES-11 comes from SSEC's
# McIDAS file sndimg11.pfc (BC1 = TC1, BC2 = TC2; sounder channels 1-18, imager channels 22-25 = bands 2-5).
GVAR_PLANCK_COEFFS = {
    8: {
        "imager": {
            "FK1": (0.0, 199986.2, 38792.39, 9737.93, 6944.64),
            "FK2": (0.0, 3684.27, 2132.72, 1345.37, 1201.99),
            "TC1": (0.0, 0.6357, 0.606, 0.3735, 0.2217),
            "TC2": (0.0, 0.9991, 0.9986, 0.9987, 0.9992),
        },
        "sounder": {
            "FK1": (
                3756.81, 4011.1, 4296.87, 4681.13, 4975.25, 5881.41, 6787.44, 8873.71, 12997.94, 28629.32, 34248.3,
                43114.3, 124235.3, 128123.5, 135148.2, 169167.1, 188235.0, 225794.4
            ),
            "FK2": (
                3765.12, 3981.16, 4281.88, 4678.91, 4962.59, 5860.42, 6770.32, 8958.91, 12965.93, 28398.28,
                34201.34, 42525.14, 124057.4, 128011.4, 134849.7, 167814.2, 188801.2, 225856.5
            ),
            "TC1": (
                0.0123, 0.0133, 0.0186, 0.015, 0.0165, 0.0474, 0.1318, 0.12, 0.0426, 0.1505, 0.2743, 0.1447, 0.0224,
                0.022, 0.0217, 0.0579, 0.0623, 0.3675
            ),
            "TC2": (
                0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9998, 0.9995, 0.9996, 0.9999, 0.9996, 0.9993, 0.9997, 1.0,
                1.0, 1.0, 0.9999, 0.9999, 0.9995
            ),
        },
    },
    9: {
        "imager": {
            "FK1": (0.0, 198807.8, 38732.41, 9717.21, 6899.47),
            "FK2": (0.0, 3677.02, 2131.62, 1344.41, 1199.38),
            "TC1": (0.0, 0.5864, 0.4841, 0.3622, 0.2014),
            "TC2": (0.0, 0.9992, 0.9989, 0.9988, 0.9992),
        },
        "sounder": {
            "FK1": (
                3765.12, 3981.16, 4281.88, 4678.91, 4962.59, 5860.42, 6770.32, 8958.91, 12965.93, 28398.28,
                34201.34, 42525.14, 124057.4, 128011.4, 134849.7, 167814.2, 188801.2, 225856.5
            ),
            "FK2": (
                980.12, 998.52, 1023.05, 1053.74, 1074.62, 1135.87, 1191.85, 1308.49, 1480.08, 1922.13, 2045.03,
                2199.04, 3142.14, 3175.18, 3230.74, 3475.05, 3614.26, 3836.74
            ),
            "TC1": (
                0.0099, 0.0119, 0.0122, 0.0119, 0.0135, 0.044, 0.1345, 0.1193, 0.0407, 0.1438, 0.2762, 0.137,
                0.0189, 0.0198, 0.0191, 0.0531, 0.0612, 0.3042
            ),
            "TC2": (
                1.0, 0.9999, 0.9999, 0.9999, 0.9999, 0.9998, 0.9995, 0.9996, 0.9999, 0.9996, 0.9993, 0.9997, 1.0,
                1.0, 1.0, 0.9999, 0.9999, 0.9996
            ),
        },
    },
    10: {
        "imager": {
            "FK1": (0.0, 198410.0, 39086.0, 9774.4, 6828.6),
            "FK2": (0.0, 3674.5, 2138.1, 1347.0, 1195.3),
            "TC1": (0.0, 0.62226, 0.61438, 0.27791, 0.21145),
            "TC2": (0.0, 0.99912, 0.99857, 0.99905, 0.99919),
        },
        "sounder": {
            "FK1": (
                3730.5, 4003.9, 4312.4, 4661.6, 4973.4, 5869.8, 6816.1, 8940.4, 12973.0, 28708.0, 34401.0, 43086.0,
                124680.0, 128820.0, 135320.0, 168530.0, 188620.0, 224870.0
            ),
            "FK2": (
                977.1, 1000.4, 1025.5, 1052.4, 1075.4, 1136.5, 1194.5, 1307.6, 1480.4, 1929.1, 2049.0, 2208.7,
                3147.4, 3181.8, 3234.5, 3480.0, 3613.1, 3831.1
            ),
            "TC1": (
                0.00988, 0.01196, 0.01245, 0.01245, 0.01366, 0.04311, 0.13973, 0.11707, 0.03979, 0.14968, 0.27603,
                0.13049, 0.02008, 0.01834, 0.02017, 0.05292, 0.0533, 0.28683
            ),
            "TC2": (
                0.99995, 0.99994, 0.99994, 0.99995, 0.99994, 0.99983, 0.99947, 0.99959, 0.99988, 0.99962, 0.99933,
                0.9997, 0.99997, 0.99997, 0.99997, 0.99992, 0.99992, 0.99961
            ),
        },
    },
    11: {
        "imager": {
            "FK1": (0.0, 200180.0, 38789.0, 9653.4, 6877.8),
            "FK2": (0.0, 3685.4, 2132.7, 1341.5, 1198.1),
            "TC1": (0.0, 0.62864, 0.59339, 0.38284, 0.20258),
            "TC2": (0.0, 0.99911, 0.99861, 0.99869, 0.99923),
        },
        "sounder": {
            "FK1": (
                3765.6, 3992.8, 4303.9, 4680.6, 4956.3, 5858.9, 6866.8, 8939.5, 13027.0, 28611.0, 34527.0, 43237.0,
                124990.0, 128850.0, 134490.0, 169500.0, 188730.0, 225760.0
            ),
            "FK2": (
                980.16, 999.49, 1024.8, 1053.9, 1074.2, 1135.8, 1197.5, 1307.5, 1482.4, 1926.9, 2051.5, 2211.2,
                3150.0, 3182.1, 3227.8, 3486.7, 3613.8, 3836.2
            ),
            "TC1": (
                0.00955, 0.01206, 0.0123, 0.01189, 0.01317, 0.04492, 0.1299, 0.118, 0.03855, 0.15092, 0.27383,
                0.13448, 0.019, 0.01939, 0.02005, 0.04879, 0.05423, 0.29165
            ),
            "TC2": (
                0.99996, 0.99994, 0.99994, 0.99995, 0.99994, 0.99982, 0.99951, 0.99959, 0.99988, 0.99962, 0.99934,
                0.99969, 0.99997, 0.99997, 0.99997, 0.99993, 0.99992, 0.9996
            ),
        },
    },
    12: {
        "imager": {
            "FK1": (0.0, 200960.0, 43702.0, 9685.9, 5047.1),
            "FK2": (0.0, 3690.2, 2219.1, 1343.0, 1080.7),
            "TC1": (0.0, 0.69703, 5.08315, 0.37554, 0.09537),
            "TC2": (0.0, 0.99902, 0.98872, 0.99872, 0.9996),
        },
        "sounder": {
            "FK1": (
                3777.8, 4008.6, 4308.5, 4704.1, 5013.4, 5864.5, 6907.1, 9038.8, 12972.0, 28931.0, 34531.0, 43340.0,
                124920.0, 128220.0, 135350.0, 169810.0, 189540.0, 225380.0
            ),
            "FK2": (
                981.21, 1000.8, 1025.2, 1055.6, 1078.3, 1136.1, 1199.8, 1312.4, 1480.3, 1934.0, 2051.6, 2213.0,
                3149.4, 3176.9, 3234.7, 3488.8, 3618.9, 3834.0
            ),
            "TC1": (
                0.0101, 0.01252, 0.01229, 0.01189, 0.01264, 0.04189, 0.13474, 0.12341, 0.03844, 0.15764, 0.2742,
                0.13683, 0.02124, 0.0178, 0.02037, 0.04933, 0.05386, 0.28872
            ),
            "TC2": (
                0.99995, 0.99994, 0.99994, 0.99995, 0.99995, 0.99983, 0.99949, 0.99957, 0.99988, 0.9996, 0.99934,
                0.99969, 0.99996, 0.99997, 0.99997, 0.99993, 0.99992, 0.99961
            ),
        },
    },
    13: {
        "imager": {
            "FK1": (0.0, 200750.0, 42508.0, 9805.0, 5017.3),
            "FK2": (0.0, 3689.0, 2198.7, 1348.4, 1078.6),
            "TC1": (0.0, 1.4795, 3.96964, 0.3635, 0.09502),
            "TC2": (0.0, 0.99794, 0.99112, 0.99876, 0.9996),
        },
        "sounder": {
            "FK1": (
                3733.0, 4035.1, 4266.1, 4692.0, 4995.3, 5802.1, 6833.8, 8942.0, 13042.0, 28772.0, 34368.0, 42901.0,
                125130.0, 128120.0, 134820.0, 168920.0, 189380.0, 226080.0
            ),
            "FK2": (
                977.32, 1003.0, 1021.8, 1054.7, 1077.0, 1132.1, 1195.6, 1307.7, 1483.0, 1930.5, 2048.3, 2205.5,
                3151.2, 3176.1, 3230.5, 3482.6, 3617.9, 3838.0
            ),
            "TC1": (
                0.00944, 0.01022, 0.01011, 0.01291, 0.01353, 0.04272, 0.12493, 0.12033, 0.03838, 0.15609, 0.28,
                0.18057, 0.01799, 0.01809, 0.02012, 0.05092, 0.0574, 0.29874
            ),
            "TC2": (
                0.99996, 0.99995, 0.99995, 0.99994, 0.99994, 0.99983, 0.99952, 0.99958, 0.99988, 0.99961, 0.99932,
                0.99959, 0.99997, 0.99997, 0.99997, 0.99992, 0.99992, 0.99959
            ),
        },
    },
    14: {
        "imager": {
            "FK1": (0.0, 204320.0, 42187.0, 9721.1, 5070.1),
            "FK2": (0.0, 3710.7, 2192.9, 1344.6, 1082.3),
            "TC1": (0.0, 1.56149, 3.75637, 0.3797, 0.08482),
            "TC2": (0.0, 0.99783, 0.99154, 0.99869, 0.99964),
        },
        "sounder": {
            "FK1": (
                3748.9, 4008.3, 4269.6, 4695.2, 4978.0, 5807.4, 6839.8, 8972.5, 13085.0, 28920.0, 34362.0, 42901.0,
                125040.0, 128340.0, 135210.0, 168900.0, 189000.0, 227200.0
            ),
            "FK2": (
                978.71, 1000.8, 1022.1, 1055.0, 1075.7, 1132.4, 1195.9, 1309.2, 1484.6, 1933.8, 2048.2, 2205.5,
                3150.4, 3177.9, 3233.6, 3482.5, 3615.5, 3844.3
            ),
            "TC1": (
                0.00949, 0.01039, 0.01018, 0.01299, 0.01335, 0.04179, 0.12271, 0.11528, 0.0387, 0.14343, 0.27532,
                0.18229, 0.01783, 0.01744, 0.01906, 0.05207, 0.05324, 0.2968
            ),
            "TC2": (
                0.99996, 0.99995, 0.99995, 0.99994, 0.99994, 0.99983, 0.99953, 0.9996, 0.99988, 0.99964, 0.99933,
                0.99958, 0.99997, 0.99997, 0.99997, 0.99992, 0.99992, 0.9996
            ),
        },
    },
    15: {
        "imager": {
            "FK1": (0.0, 200750.0, 42330.0, 9766.8, 5088.2),
            "FK2": (0.0, 3689.0, 2195.7, 1346.6, 1083.6),
            "TC1": (0.0, 1.59091, 3.75399, 0.38176, 0.08671),
            "TC2": (0.0, 0.99777, 0.99156, 0.9987, 0.99963),
        },
        "sounder": {
            "FK1": (
                3731.8, 4002.2, 4265.6, 4684.9, 4981.7, 5833.2, 6816.8, 8993.7, 13075.0, 28886.0, 34356.0, 42758.0,
                124880.0, 128320.0, 135200.0, 169620.0, 188730.0, 227910.0
            ),
            "FK2": (
                977.22, 1000.3, 1021.8, 1054.2, 1076.0, 1134.1, 1194.6, 1310.2, 1484.2, 1933.1, 2048.1, 2203.0,
                3149.1, 3177.7, 3233.5, 3487.5, 3613.8, 3848.3
            ),
            "TC1": (
                0.00909, 0.00998, 0.01021, 0.01327, 0.01318, 0.04219, 0.11811, 0.11003, 0.0389, 0.13744, 0.27731,
                0.17079, 0.01809, 0.01782, 0.01976, 0.05217, 0.05275, 0.29972
            ),
            "TC2": (
                0.99996, 0.99995, 0.99995, 0.99994, 0.99994, 0.99983, 0.99955, 0.99962, 0.99988, 0.99965, 0.99933,
                0.99961, 0.99997, 0.99997, 0.99997, 0.99992, 0.99992, 0.99959
            ),
        },
    },
}

def gvar_sensor(sensor_number):
    """``(satellite number, "imager" or "sounder")`` of a GVAR sensor source number, e.g. ``(13, "imager")``.

    Taken from :mod:`pyarea.sensors`, so it follows the same sensor table as the directory.

    Raises:
        ValueError: if the number is not a GOES imager or sounder.
    """
    sensor = get_sensor(sensor_number)
    if (sensor is None or not sensor.platform.startswith("GOES-")
            or sensor.instrument not in ("imager", "sounder")):
        raise ValueError(f"Sensor {sensor_number} is not a GOES imager or sounder.")
    return int(sensor.platform.split("-")[1]), sensor.instrument


_UNITS = {"RAW": Calibrator.CAL_RAW, "RAD": Calibrator.CAL_RAD, "ALB": Calibrator.CAL_ALB,
          "TEMP": Calibrator.CAL_TEMP, "BRIT": Calibrator.CAL_BRIT}


def cal_type_from_name(name):
    """The ``Calibrator.CAL_*`` number for ``RAW``, ``RAD``, ``ALB``, ``TEMP`` or ``BRIT``."""
    try:
        return _UNITS[name.strip().upper()]
    except KeyError:
        raise ValueError(f"Unknown calibration type {name!r}; use one of {', '.join(_UNITS)}.") from None


def calibrated_unit(cal_type):
    """The unit of a calibrated value, or None where there isn't one (counts and brightness)."""
    return {Calibrator.CAL_RAD: "mW/m^2/sr/cm-1", Calibrator.CAL_ALB: "%", Calibrator.CAL_TEMP: "K"}.get(cal_type)


def brit_to_temperature(brit):
    """Brightness (0-255) to temperature in K, the inverse of the IR brightness formula."""
    brit = np.asarray(brit, dtype=np.float32)
    return np.where(brit > 176, 418.0 - brit, (660.0 - brit) / 2.0)


def temperature_to_brit(temperature):
    """Temperature in K to brightness: ``660 - 2T`` from 242 K up, ``418 - T`` below; NaN stays NaN."""
    t = np.asarray(temperature, dtype=np.float64)
    return np.where(t >= 242.0, np.maximum(660.0 - np.trunc(t * 2.0), 0.0), np.minimum(418.0 - np.trunc(t), 255.0))


class GVARCalibrator(Calibrator):
    """Calibrate GVAR (GOES-8 onward) counts from the coefficients in the AREA calibration block.

    A numpy port of ``edu.wisc.ssec.mcidas.CalibratorGvar`` (VisAD): bias and gain coefficients are read
    from the block in Gould floating point format; the detector coefficients are averaged for the visible,
    and the first detector's are used for the IR bands.

    Args:
        sensorId: The sensor source number; its satellite and instrument come from :mod:`pyarea.sensors`.
        calBlock: The calibration block as 32-bit words, or None for data that is already calibrated.
    """

    NUM_BANDS_IMAGER = 5
    NUM_BANDS_SOUNDER = 18

    NUM_VIS_DETECTORS = 8
    NUM_IR_DETECTORS = 2
    NUM_IR_BANDS = 4

    def __init__(self, sensorId=None, calBlock=None):
        self.sid = sensorId
        self.satellite, instrument = gvar_sensor(sensorId)
        self.imager = instrument == "imager"
        self.curCalType = self.CAL_RAW
        self.isPreCalibrated = calBlock is None
        self.scale = 32 if self.imager else 2
        if calBlock is not None:
            self._read_cal_block([utils.gould2native(int(word) & 0xFFFFFFFF) for word in calBlock])

    def _read_cal_block(self, values):
        """Split the block's coefficients like ``CalibratorGvar.initGvar``."""
        n_vis = self.NUM_VIS_DETECTORS if self.imager else self.NUM_VIS_DETECTORS // 2
        need = 3 * n_vis + 1 + (2 * self.NUM_IR_DETECTORS * self.NUM_IR_BANDS if self.imager
                                else 2 * self.NUM_BANDS_SOUNDER)
        if len(values) < need:
            raise ValueError(f"The GVAR calibration block has {len(values)} words, but {need} are needed.")
        values = np.asarray(values, dtype=np.float64)
        self.visBiasCoef = values[0:n_vis]
        self.visGain1Coef = values[n_vis:2 * n_vis]
        self.visGain2Coef = values[2 * n_vis:3 * n_vis]
        self.visBiasCoefAVG = self.visBiasCoef.mean()
        self.visGain1CoefAVG = self.visGain1Coef.mean()
        self.visGain2CoefAVG = self.visGain2Coef.mean()
        self.visRadToAlb = values[3 * n_vis]
        rest = values[3 * n_vis + 1:]
        if self.imager:
            # the IR bands are stored rotated by two for satellites before GOES-12
            ir_offset = 0 if self.satellite >= 12 else 2
            order = [(i + ir_offset) % self.NUM_IR_BANDS for i in range(self.NUM_IR_BANDS)]
            ir = rest[:4 * self.NUM_IR_BANDS].reshape(4, self.NUM_IR_BANDS)
            self.irBiasCoef = np.empty((self.NUM_IR_DETECTORS, self.NUM_IR_BANDS))
            self.irGainCoef = np.empty((self.NUM_IR_DETECTORS, self.NUM_IR_BANDS))
            for det in range(self.NUM_IR_DETECTORS):
                self.irBiasCoef[det, order] = ir[det]
                self.irGainCoef[det, order] = ir[self.NUM_IR_DETECTORS + det]
        else:
            self.sBiasCoef = rest[:self.NUM_BANDS_SOUNDER]
            self.sGainCoef = rest[self.NUM_BANDS_SOUNDER:2 * self.NUM_BANDS_SOUNDER]

    def setCalType(self, calType):
        """Set the calibration type of the input data; anything but RAW is passed through unchanged."""
        if calType < Calibrator.CAL_MIN or calType > Calibrator.CAL_MAX:
            return -1
        self.curCalType = calType
        return 0

    def _gain_bias(self, band):
        if self.imager:
            if band == 6:  # the 13 micron band of GOES-12 onward uses band 5's coefficients
                band = 5
            return self.irGainCoef[0, band - 2], self.irBiasCoef[0, band - 2]
        return self.sGainCoef[band - 1], self.sBiasCoef[band - 1]

    def _planck(self, band):
        satellite = self.satellite
        if satellite not in GVAR_PLANCK_COEFFS:
            raise NotImplementedError(f"No Planck constants for GOES-{satellite}; only GOES-8 to 15 are available.")
        instrument = "imager" if self.imager else "sounder"
        if instrument not in GVAR_PLANCK_COEFFS[satellite]:
            raise NotImplementedError(f"No {instrument} Planck constants for GOES-{satellite}.")
        table = GVAR_PLANCK_COEFFS[satellite][instrument]
        index = (5 if band == 6 else band) - 1 if self.imager else band - 1
        return tuple(table[name][index] for name in ("FK1", "FK2", "TC1", "TC2"))

    def radToTemp(self, inval=None, band=None, sId=None):
        """Radiance to temperature in K: ``T = (FK2 / ln(FK1 / R + 1) - TC1) / TC2``."""
        fk1, fk2, tc1, tc2 = self._planck(band)
        radiance = np.asarray(inval, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            expn = fk1 / radiance + 1.0
            temperature = np.where(expn > 0, (fk2 / np.log(expn) - tc1) / tc2, np.nan)
        return temperature

    def _is_visible(self, band):
        return band == (1 if self.imager else 19)

    def _albedo(self, counts):
        """Visible albedo as a fraction, from counts."""
        f = counts / self.scale
        g2 = f ** self.visGain2CoefAVG if abs(self.visGain2CoefAVG) > 0.0001 else 1.0
        return np.maximum((g2 + f * self.visGain1CoefAVG + self.visBiasCoefAVG) * self.visRadToAlb, 0.0)

    def calibrate_array(self, input_array, band, calTypeOut):
        """Convert counts to ``calTypeOut`` (a ``Calibrator.CAL_*`` number) for ``band``; float32 array.

        Visible bands can give RAW, ALB (percent) or BRIT; IR bands RAW, RAD, TEMP (K) or BRIT.
        Input that is not RAW is passed through, as is everything when no calibration block is present.
        """
        counts = np.asarray(input_array, dtype=np.float64)
        if self.curCalType != self.CAL_RAW or self.isPreCalibrated or calTypeOut == self.CAL_RAW:
            return counts.astype(np.float32)
        if self._is_visible(band):
            if calTypeOut not in (self.CAL_ALB, self.CAL_BRIT):
                raise ValueError(f"Band {band} is visible; it can be calibrated to ALB or BRIT, not {calTypeOut}.")
            albedo = self._albedo(counts)
            if calTypeOut == self.CAL_ALB:
                return (albedo * 100.0).astype(np.float32)
            return np.minimum(np.floor(np.sqrt(100.0 * albedo) * 25.5 + 0.5), 255.0).astype(np.float32)
        if calTypeOut == self.CAL_ALB:
            raise ValueError(f"Band {band} is infrared; it can be calibrated to RAD, TEMP or BRIT, not ALB.")
        gain, bias = self._gain_bias(band)
        radiance = ((counts / self.scale if self.imager else counts) - bias) / gain
        if calTypeOut == self.CAL_RAD:
            return radiance.astype(np.float32)
        temperature = self.radToTemp(radiance, band, self.sid)
        if calTypeOut == self.CAL_TEMP:
            return temperature.astype(np.float32)
        return temperature_to_brit(temperature).astype(np.float32)

    def calibrate_pixel(self, inputPixel, band, calTypeOut):
        """:meth:`calibrate_array` for a single value."""
        return float(self.calibrate_array(np.asarray([inputPixel]), band, calTypeOut)[0])

    def calibratedList(self, band, isPreCal=None):
        """The calibration types available for ``band``."""
        isPreCal = self.isPreCalibrated if isPreCal is None else isPreCal
        visible = band in (1, 12)
        if isPreCal:
            return [self.CAL_RAW, self.CAL_BRIT] if visible else [self.CAL_RAW, self.CAL_TEMP, self.CAL_BRIT]
        if visible:
            return [self.CAL_RAW, self.CAL_ALB, self.CAL_BRIT]
        return [self.CAL_RAW, self.CAL_TEMP, self.CAL_RAD, self.CAL_BRIT]


class G8GVARCalibrator(GVARCalibrator):
    """GOES-8 (sensors 70 and 71). The constants for every GVAR satellite are in ``GVAR_PLANCK_COEFFS``."""


class G9GVARCalibrator(GVARCalibrator):
    """GOES-9 (sensors 72 and 73)."""


class G10GVARCalibrator(GVARCalibrator):
    """GOES-10 (sensors 74 and 75)."""


# ---------------------------------------------------------------------------
# Pre-GVAR (SMS/GOES-1..7 VISSR/VAS) calibration.

class GOESCalibrator(Calibrator):
    """Calibration of the pre-GVAR GOES (SMS/GOES-1 to 7) VISSR and VAS data, like :class:`GVARCalibrator` for GVAR.

    Three kinds of file need three calibrations, and each needs different parts of the file, so unlike
    :class:`GVARCalibrator` this class has no per-file state: the calibrations are static methods that take the
    counts and whatever coefficients the file holds (``AreaFile.read_*`` supplies them).

    * Mode A (source type VISR): 8-bit counts, brightness temperature from the documented BRIT formula
      (:meth:`mode_a_ir_temperature`) and a relative visible reflectance (:meth:`vis_unnormalized_reflectance`).
    * Mode AA (VAS, GOES-4/5/6, 1980-86): per-line offset and gain entries in the line prefix, Planck-inverted
      with the McIDAS vastbl constants (:meth:`mode_aa_temperature`).
    * Mode AAA (GOES-6/7, 1987-94): per-channel coefficients in the CAL block (:meth:`aaa_radiance`,
      :meth:`aaa_temperature`).

    Constants courtesy of SSEC via NOAA CLASS.
    """

    # channel code (1-38) -> IR band (1-12); index 0 is code 1
    MODE_AA_CHANNEL_BANDS = tuple(range(1, 13)) * 2 + (3, 4, 5, 7, 8, 9, 10) * 2
    # deltaF prefix code (0-15) -> actual deltaF (-5..+5)
    MODE_AA_DELTA_F = (0, -1, -2, -3, -4, -5, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0)
    # band -> (F, FK1, FK2, TC1, TC2)
    MODE_AA_COEFFS = {
        1: (7, 3723, 976.5, -0.0003905001, 1.0),
        2: (7, 3910, 992.5, 0.002225, 1.0),
        3: (7, 4097, 1008, 0.00236, 1.0),
        4: (7, 4352, 1029, 0.004170999, 0.9999),
        5: (7, 5049, 1081, 0.004202001, 0.9999),
        6: (11, 128100, 3176, 0.06761003, 0.9998),
        7: (7, 5874, 1137, 0.004214, 0.9999),
        8: (7, 8512, 1286, 0.3212, 0.9974),
        9: (8, 30940, 1978, 0.06927001, 0.9997),
        10: (8, 38880, 2134, 0.7513, 0.9973),
        11: (11, 136000, 3240, 0.05738, 0.9999),
        12: (13, 195900, 3659, 0.4237, 0.9991),
    }

    # Mode AAA channel -> (FK1, FK2, TC1, TC2). Only channels 1-12 are confirmed;
    # channel 13 borrows channel 8's constants.
    PLANCK_COEFFS_GOES6 = {
        1: (3716.0, 975.8, -0.0002859, 1.0),
        2: (3936.0, 994.8, 0.002472, 1.0),
        3: (4089.0, 1007.0, 0.002334, 1.0),
        4: (4336.0, 1027.0, 0.003948, 0.9999),
        5: (5019.0, 1079.0, 0.004031, 0.9999),
        6: (128000.0, 3175.0, 0.06857, 0.9998),
        7: (5855.0, 1136.0, 0.004256, 0.9999),
        8: (8438.0, 1283.0, 0.3263, 0.9973),
        9: (31030.0, 1980.0, 0.06822, 0.9997),
        10: (38790.0, 2133.0, 0.736, 0.9973),
        11: (136100.0, 3241.0, 0.05787, 0.9999),
        12: (195000.0, 3653.0, 0.4213, 0.9991),
    }
    PLANCK_COEFFS_GOES7 = {
        1: (3767.0, 980.3, -0.001267, 1.0),
        2: (3940.0, 995.1, 0.00318, 0.9999),
        3: (4167.0, 1014.0, 0.002383, 1.0),
        4: (4364.0, 1030.0, 0.002739, 0.9999),
        5: (5025.0, 1079.0, 0.00536, 0.9999),
        6: (129100.0, 3185.0, 0.07353, 0.9998),
        7: (5851.0, 1135.0, 0.004447, 0.9999),
        8: (8525.0, 1287.0, 0.3408, 0.9973),
        9: (31070.0, 1981.0, 0.06478, 0.9998),
        10: (39200.0, 2140.0, 0.6448, 0.9977),
        11: (132500.0, 3212.0, 0.0583, 0.9999),
        12: (193200.0, 3642.0, 0.415, 0.9991),
    }
    PLANCK_COEFFS_BY_SATELLITE = {
        "GOES-6": PLANCK_COEFFS_GOES6,
        "GOES-7": PLANCK_COEFFS_GOES7,
    }

    @classmethod
    def vas_planck_coeffs(cls, satellite, channel):
        """Return ``(FK1, FK2, TC1, TC2)`` for a Mode AAA channel of ``satellite``."""
        table = cls.PLANCK_COEFFS_BY_SATELLITE.get(satellite)
        if table is None:
            raise NotImplementedError(
                f"No verified VAS Planck-inversion constants for satellite {satellite!r} "
                f"(only GOES-6 and GOES-7 are confirmed)."
            )
        planck_channel = 8 if channel == 13 else channel
        if planck_channel not in table:
            raise NotImplementedError(
                f"No verified Planck-inversion constants for channel {channel} on {satellite} "
                f"(only channels 1-12 have been checked against real data)."
            )
        return table[planck_channel]

    @staticmethod
    def mode_a_ir_temperature(counts):
        """Mode A (VISR) IR counts to brightness temperature in K: ``418 - B`` for B >= 176, else ``330 - B/2``."""
        counts = np.asarray(counts, dtype=np.float32)
        return np.where(counts >= 176, 418.0 - counts, 330.0 - counts / 2.0)

    @staticmethod
    def vis_unnormalized_reflectance(counts):
        """VIS counts to a relative reflectance in percent: ``(counts / 255) ** 2 * 100``, clipped to 0-100.

        The visible counts are 6 bits shifted left by two, so they run from 0 to 252 on the 0-255 brightness scale
        of McIDAS. The VAS visible channel does not output counts proportional to radiance but to its square root,
        so that the photomultiplier noise, which grows with the square root of the radiance, is about the same in
        counts at every level (Kidder and Vonder Haar, 1995, *Satellite Meteorology: An Introduction*, chapter 4).
        Squaring the counts undoes that, in the form of the GOES-R fire algorithm's "visible brightness"
        (``int(sqrt(reflectivity) * 255)``, its ATBD): the largest count, 252, gives 97.7 %. This has not been
        checked on real data, and the VISSR visible channel of the satellites before the VAS may be encoded
        differently.

        The result is relative, not an absolute reflectance: no calibration to albedo exists for these satellites
        and the solar zenith angle is not corrected for.
        """
        fraction = np.clip(np.asarray(counts, dtype=np.float32) / 255.0, 0.0, 1.0)
        return (fraction ** 2 * 100.0).astype(np.float32)

    @classmethod
    def mode_aa_code_to_band(cls, codes):
        """Map Mode AA channel codes (1-38) to IR bands (1-12); anything else (e.g. unused 0) maps to 0."""
        codes = np.asarray(codes)
        lookup = np.array((0,) + cls.MODE_AA_CHANNEL_BANDS)
        out = np.zeros_like(codes)
        valid = (codes >= 1) & (codes <= 38)
        out[valid] = lookup[codes[valid]]
        return out

    @classmethod
    def mode_aa_temperature(cls, raw, delta_f, ysubz, band):
        """Mode AA brightness temperature (K) for IR ``band`` from raw counts and per-line offset/gain.

        Args:
            raw: Raw counts, shape (lines, elements).
            delta_f: Per-line deltaF, shape (lines,).
            ysubz: Per-line zero offset, shape (lines,).
            band: IR band 1-12.

        Returns:
            float32 array, NaN where the raw count is 0 or the corrected count rounds down to 0.
        """
        f, fk1, fk2, tc1, tc2 = cls.MODE_AA_COEFFS[band]
        corrected = np.maximum(0.0, raw - ysubz[:, None])
        corrected = 8.0 * np.floor(corrected / 8.0)
        radiance = corrected * (2.0 ** (delta_f[:, None] - f))
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (fk2 / np.log(fk1 / radiance + 1.0) - tc1) / tc2
        return np.where((raw == 0) | (corrected == 0), np.nan, t).astype(np.float32)

    @staticmethod
    def aaa_radiance(counts, ab1, ab2, ifab):
        """Mode AAA counts to radiance using the CAL block's AB1, AB2 and IFAB for the channel."""
        return ((ab2 * np.asarray(counts, dtype=np.float64) / 32.0 - ab1) / 2.0 ** (15 - ifab)).astype(np.float32)

    @staticmethod
    def aaa_temperature(radiance, coeffs):
        """Planck-invert Mode AAA radiance with ``(FK1, FK2, TC1, TC2)``; NaN where the inversion is undefined."""
        fk1, fk2, tc1, tc2 = coeffs
        r = np.asarray(radiance, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            expn = fk1 / r + 1.0
            t = fk2 / np.log(expn)
        return np.where((r > -fk1) & (expn > 0), (t - tc1) / tc2, np.nan).astype(np.float32)
