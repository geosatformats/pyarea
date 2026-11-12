"""Independent scalar Python port of VisAD's GMSXnav.java (LGPL), used as a reference in tests.

It follows the Java statement by statement, including the float casts, the (i - 1) in the sub-block offsets and
the state (sat, sp, ss, sitagt, ...) that mg1100 leaves behind, so it is slow and only meant for comparing.
"""

import math

import numpy as np

f32 = np.float32
CDR = math.pi / 180.0
CRD = 180.0 / math.pi
HPAI = math.pi / 2.0
DPAI = math.pi * 2.0
EA = 6378136.0
EF = 1.0 / 298.257


def int_to_bytes(v):
    return [(v >> (8 * (3 - i))) & 0xFF for i in range(4)]


def signed(b):
    return b - 256 if b > 127 else b


class GMSXRef:
    def __init__(self, navblock):
        b = []
        for i in range(126):
            b += int_to_bytes(int(navblock[i + 1]) & 0xFFFFFFFF)
        src = 128
        for _ in range(4):
            for i in range(127):
                b += int_to_bytes(int(navblock[i + src]) & 0xFFFFFFFF)
            src += 128
        self.b = [signed(x) for x in b] + [0] * (3200 - len(b))
        self.resLin = [0.0] * 4
        self.resEle = [0.0] * 4
        self.rlic = [0.0] * 4
        self.relmfc = [0.0] * 4
        self.senssu = [0.0] * 4
        self.rline = [0.0] * 4
        self.relem = [0.0] * 4
        self.vmis = [0.0] * 3
        self.elmis = [[0.0] * 3 for _ in range(3)]
        self.orbt1 = [[0.0] * 8 for _ in range(35)]
        self.atit = [[0.0] * 10 for _ in range(10)]
        self.sat = [0.0] * 3
        self.sp = [0.0] * 3
        self.ss = [0.0] * 3
        self.sitagt = self.sunalp = self.sundel = 0.0
        self.dec_oa_block(0)

    def sv0100(self, iword, ipos, off):
        b = self.b
        negative = b[off] < 0
        tmp = [0] * 6
        r = 0.0
        if iword == 4:
            for i in range(1, 4):
                tmp[i] = (b[i + off] & 0x7F) + 128 if b[i + off] < 0 else b[i + off]
            r = (b[off] & 0x7F) * 16777216.0 + tmp[1] * 65536.0 + tmp[2] * 256.0 + tmp[3]
        if iword == 6:
            for i in range(1, 6):
                tmp[i] = (b[i + off] & 0x7F) + 128 if b[i + off] < 0 else b[i + off]
            r = (
                (b[off] & 0x7F) * 2.0**40
                + tmp[1] * 2.0**32
                + tmp[2] * 16777216.0
                + tmp[3] * 65536.0
                + tmp[4] * 256.0
                + tmp[5]
            )
        r = r / 10.0**ipos
        return -r if negative else r

    def dec_oa_block(self, form):
        s, F = self.sv0100, lambda x: float(f32(x))
        self.dtims = s(6, 8, 0)
        self.dspin = s(6, 8, 240)
        self.resLin[0] = F(s(4, 8, 6))
        self.resLin[1] = self.resLin[2] = self.resLin[3] = F(s(4, 8, 10))
        self.resEle[0] = F(s(4, 10, 14))
        self.resEle[1] = self.resEle[2] = self.resEle[3] = F(s(4, 10, 18))
        self.rlic[0], self.rlic[1], self.rlic[2], self.rlic[3] = (F(s(4, 4, o)) for o in (22, 26, 110, 114))
        self.relmfc[0], self.relmfc[1], self.relmfc[2], self.relmfc[3] = (F(s(4, 4, o)) for o in (30, 34, 118, 122))
        self.senssu[0] = F(s(4, 0, 38))
        self.senssu[1] = self.senssu[2] = self.senssu[3] = F(s(4, 0, 42))
        self.rline[0] = F(s(4, 0, 46))
        self.rline[1] = self.rline[2] = self.rline[3] = F(s(4, 0, 50))
        self.relem[0] = F(s(4, 0, 54))
        self.relem[1] = self.relem[2] = self.relem[3] = F(s(4, 0, 58))
        self.vmis = [F(s(4, 10, 62)), F(s(4, 10, 66)), F(s(4, 10, 70))]
        e = self.elmis
        e[0][0] = F(s(4, 7, 74))
        e[1][0] = F(s(4, 10, 78))
        e[2][0] = F(s(4, 10, 82))
        e[0][1] = F(s(4, 10, 86))
        e[1][1] = F(s(4, 7, 90))
        e[2][1] = F(s(4, 10, 94))
        e[0][2] = F(s(4, 10, 98))
        e[1][2] = F(s(4, 10, 102))
        e[2][2] = F(s(4, 7, 106))
        self.subLon = F(s(6, 6, 198))
        self.subLat = F(s(6, 6, 204))
        for i in range(10):
            j = (i - 1) * 64 + 256 if form == 1 else (i - 1) * 48 + 256
            self.atit[0][i] = s(6, 8, j)
            self.atit[2][i] = s(6, 8, j + 12)
            self.atit[3][i] = s(6, 11, j + 18)
            self.atit[4][i] = s(6, 8, j + 24)
            self.atit[5][i] = s(6, 8, j + 30)
        for i in range(8):
            j = (i - 1) * 256 + 896 if form == 1 else (i - 1) * 200 + 736
            o = self.orbt1
            o[0][i] = s(6, 8, j)
            o[8][i] = s(6, 6, j + 48)
            o[9][i] = s(6, 6, j + 54)
            o[10][i] = s(6, 6, j + 60)
            o[14][i] = s(6, 8, j + 84)
            o[17][i] = s(6, 8, j + 102)
            o[18][i] = s(6, 8, j + 108)
            for k, (power, off) in zip(
                range(19, 28),
                [(12, 128), (14, 134), (14, 140), (14, 146), (12, 152), (16, 158), (12, 164), (16, 170), (12, 176)],
            ):
                o[k][i] = s(6, power, j + off)

    # vector helpers
    @staticmethod
    def mg1200(v):
        rv1 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if rv1 == 0.0:
            return [0.0, 0.0, 0.0]
        rv2 = math.sqrt(rv1)
        return [v[0] / rv2, v[1] / rv2, v[2] / rv2]

    @staticmethod
    def mg1210(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def mg1220(self, a, b):
        return self.mg1200(self.mg1210(a, b))

    @staticmethod
    def mg1230(a, b):
        as1 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        as2 = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) * (b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        if as2 == 0.0:
            return 0.0
        temp = as1 / math.sqrt(as2)
        if temp > 1.0 and temp - 1.0 < 0.000001:
            temp = 1
        return math.acos(temp)

    def mg1110(self, i, rtim, orbta):
        npa = [[0.0] * 3 for _ in range(3)]
        if i != 7:
            delt = (rtim - orbta[0][i]) / (orbta[0][i + 1] - orbta[0][i])
            self.sat = [orbta[k][i] + (orbta[k][i + 1] - orbta[k][i]) * delt for k in (8, 9, 10)]
            self.sitagt = (orbta[14][i] + (orbta[14][i + 1] - orbta[14][i]) * delt) * CDR
            if orbta[14][i + 1] - orbta[14][i] < 0.0:
                self.sitagt = (orbta[14][i] + (orbta[14][i + 1] - orbta[14][i] + 360.0) * delt) * CDR
            self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i]) * delt) * CDR
            if orbta[17][i + 1] - orbta[17][i] > 0.0:
                self.sunalp = (orbta[17][i] + (orbta[17][i + 1] - orbta[17][i] - 360.0) * delt) * CDR
            self.sundel = (orbta[18][i] + (orbta[18][i + 1] - orbta[18][i]) * delt) * CDR
            for n, (r, c) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]):
                npa[r][c] = orbta[19 + n][i]
        return npa

    def mg1100(self, rtim):
        beta = delt = attalp = attdel = 0.0
        npa = [[0.0] * 3 for _ in range(3)]
        for i in range(7):
            if self.orbt1[0][i] < rtim < self.orbt1[0][i + 1]:
                npa = self.mg1110(i, rtim, self.orbt1)
                break
        atit = self.atit
        for i in range(9):
            if atit[0][i] <= rtim < atit[0][i + 1]:
                delt = (rtim - atit[0][i]) / (atit[0][i + 1] - atit[0][i])
                attalp = atit[2][i] + (atit[2][i + 1] - atit[2][i]) * delt
                attdel = atit[3][i] + (atit[3][i + 1] - atit[3][i]) * delt
                beta = atit[4][i] + (atit[4][i + 1] - atit[4][i]) * delt
                if atit[4][i + 1] - atit[4][i] > 0.0:
                    beta = atit[4][i] + (atit[4][i + 1] - atit[4][i] - 360.0 * CDR) * delt
                break
        wkcos = math.cos(attdel)
        att1 = [math.sin(attdel), wkcos * (-math.sin(attalp)), wkcos * math.cos(attalp)]
        att2 = [sum(npa[r][c] * att1[c] for c in range(3)) for r in range(3)]
        wksin, wkcos = math.sin(self.sitagt), math.cos(self.sitagt)
        att3 = [wkcos * att2[0] + wksin * att2[1], -wksin * att2[0] + wkcos * att2[1], att2[2]]
        self.sp = self.mg1200(att3)
        wkcos = math.cos(self.sundel)
        self.ss = [wkcos * math.cos(self.sunalp), wkcos * math.sin(self.sunalp), math.sin(self.sundel)]
        return beta

    def mgivsr(self, mode, rPix, rLin, rLon, rLat):
        """Returns (a, b) = (line, element) for mode > 0 and (lat, lon) for mode < 0; NaN when not navigable."""
        nan = float("nan")
        point = [nan, nan]
        if abs(mode) > 4:
            return point
        if abs(rLat) > 90 and mode > 0:
            return point
        lmode = abs(mode) - 1
        rstep, rsamp = self.resLin[lmode], self.resEle[lmode]
        rfcl, rfcp, sens = self.rlic[lmode], self.relmfc[lmode], self.senssu[lmode]
        rftl = float(f32(self.rline[lmode] + 0.5))
        rftp = float(f32(self.relem[lmode] + 0.5))
        vmis, elmis = self.vmis, self.elmis
        if mode > 0:
            dLat, dLon = float(f32(rLat)) * CDR, float(f32(rLon)) * CDR
            ee = 2.0 * EF - EF * EF
            en = EA / math.sqrt(1.0 - ee * math.sin(dLat) * math.sin(dLat))
            stn1 = [
                en * math.cos(dLat) * math.cos(dLon),
                en * math.cos(dLat) * math.sin(dLon),
                en * (1.0 - ee) * math.sin(dLat),
            ]
            rio = float(f32(rfcl - math.atan(math.sin(dLat) / (6.610689 - math.cos(dLat))) / rstep))
            rtim = self.dtims + float(f32(rio) / f32(sens)) / 1440.0 / self.dspin
            for _ in range(200):
                beta = self.mg1100(rtim)
                sw1 = self.mg1220(self.sp, self.ss)
                sw2 = self.mg1220(sw1, self.sp)
                bc, bs = math.cos(beta), math.sin(beta)
                sw3 = [sw1[k] * bs + sw2[k] * bc for k in range(3)]
                sx = self.mg1200(sw3)
                sy = self.mg1220(self.sp, sx)
                slv = [stn1[k] - self.sat[k] for k in range(3)]
                sl = self.mg1200(slv)
                sw2 = self.mg1210(self.sp, sl)
                sw3 = self.mg1210(sy, sw2)
                tp = self.mg1230(sy, sw2)
                tf = self.sp[0] * sw3[0] + self.sp[1] * sw3[1] + self.sp[2] * sw3[2]
                if tf < 0.0:
                    tp = -tp
                tl = self.mg1230(self.sp, sl)
                ri = float(
                    f32(HPAI - tl) / f32(rstep) + f32(rfcl) - f32(vmis[1]) / f32(rstep)
                )  # float arithmetic in Java
                rj = float(f32(tp / rsamp + rfcp + vmis[2] / rsamp - (HPAI - tl) * math.tan(vmis[0]) / rsamp))
                if abs(ri - rio) >= 1.0:
                    rtim = (float(np.rint(f32(f32(ri) - f32(1)) / f32(sens))) + float(f32(rj) * f32(rsamp)) / DPAI) / (
                        self.dspin * 1440.0
                    ) + self.dtims
                    rio = ri
                    continue
                break
            else:
                return point
            point = [ri, rj]
            if ri < 0 or ri > rftl or rj < 0 or rj > rftp:
                point = [nan, nan]
        if mode < 0:
            rtim = (float(np.rint(f32(f32(rLin) - f32(1)) / f32(sens))) + float(f32(rPix) * f32(rsamp)) / DPAI) / (
                self.dspin * 1440.0
            ) + self.dtims
            beta = self.mg1100(rtim)
            sw1 = self.mg1220(self.sp, self.ss)
            sw2 = self.mg1220(sw1, self.sp)
            bc, bs = math.cos(beta), math.sin(beta)
            sw3 = [sw1[k] * bs + sw2[k] * bc for k in range(3)]
            sx = self.mg1200(sw3)
            sy = self.mg1220(self.sp, sx)
            pc, ps = math.cos(rstep * (rLin - rfcl)), math.sin(rstep * (rLin - rfcl))
            qc, qs = math.cos(rsamp * (rPix - rfcp)), math.sin(rsamp * (rPix - rfcp))
            sw1 = [elmis[r][0] * pc + elmis[r][2] * ps for r in range(3)]
            sw2 = [qc * sw1[0] - qs * sw1[1], qs * sw1[0] + qc * sw1[1], sw1[2]]
            sw3 = [sx[r] * sw2[0] + sy[r] * sw2[1] + self.sp[r] * sw2[2] for r in range(3)]
            sl = self.mg1200(sw3)
            sat = self.sat
            de = (1.0 - EF) * (1.0 - EF)
            dda = de * (sl[0] ** 2 + sl[1] ** 2) + sl[2] ** 2
            ddb = de * (sat[0] * sl[0] + sat[1] * sl[1]) + sat[2] * sl[2]
            ddc = de * (sat[0] ** 2 + sat[1] ** 2 - EA * EA) + sat[2] ** 2
            dd = ddb * ddb - dda * ddc
            if dd >= 0.0 and dda != 0.0:
                dk1 = (-ddb + math.sqrt(dd)) / dda
                dk2 = (-ddb - math.sqrt(dd)) / dda
            else:
                return point
            dk = dk1 if abs(dk1) < abs(dk2) else dk2
            stn1 = [sat[k] + dk * sl[k] for k in range(3)]
            dLat = math.atan(stn1[2] / (de * math.sqrt(stn1[0] ** 2 + stn1[1] ** 2)))
            if stn1[0] != 0.0:
                dLon = math.atan(stn1[1] / stn1[0])
                if stn1[0] < 0.0 and stn1[1] >= 0.0:
                    dLon += math.pi
                if stn1[0] < 0.0 and stn1[1] < 0.0:
                    dLon -= math.pi
            else:
                dLon = HPAI if stn1[1] > 0.0 else -HPAI
            point = [float(f32(dLat * CRD)), float(f32(dLon * CRD))]
        return point

    def to_latlon(self, line, elem):
        """Image line and element (as AREAnav gives them after areaCoordToImageCoord) to (lat, lon)."""
        lat, lon = self.mgivsr(-1, elem, line, 0.0, 0.0)
        if lon > 180:
            lon -= 360
        if lon < -180:
            lon += 360
        if abs(lon) > 180 or (-lon > 90 - self.subLon and -lon < 270 - self.subLon):
            return float("nan"), float("nan")
        return lat, lon

    def to_linele(self, lat, lon):
        """Latitude and longitude to image (line, element)."""
        nan = float("nan")
        if abs(lat) > 90.0:
            return nan, nan
        if lon > 180.0:
            lon -= 360.0
        if lon < -180.0:
            lon += 360.0
        if -lon > 90 - self.subLon and -lon < 270 - self.subLon:
            return nan, nan
        return tuple(self.mgivsr(1, 0.0, 0.0, lon, lat))
