"""Independent Python port of the constructor and toLatLon of VisAD's GVARnav.java (LGPL), a reference for tests."""

import math
import struct

import numpy as np

f32 = np.float32
RELLST = [
    (4, 10),
    (13, 63),
    (65, 94),
    (98, 100),
    (103, 105),
    (108, 110),
    (113, 115),
    (116, 118),
    (120, 149),
    (153, 155),
    (158, 160),
    (163, 165),
    (168, 170),
    (171, 173),
    (175, 204),
    (208, 210),
    (213, 215),
    (218, 220),
    (223, 225),
    (226, 228),
    (230, 259),
    (263, 265),
    (268, 270),
    (273, 275),
    (278, 283),
    (285, 314),
    (318, 320),
    (323, 325),
    (328, 330),
    (333, 335),
]


def bits2f(i, order):
    return struct.unpack(order + "f", struct.pack(order + "i", i))[0]


def jday(year, day):
    return day + 1461 * (year + 4799) // 4 - 3 * ((year + 4899) // 100) // 4 - 2465022


def bcd_epoch(e0, e1):
    e0 &= 0xFFFFFFFF
    e1 &= 0xFFFFFFFF
    t0 = t1 = 0
    p = 1
    for _ in range(8):
        t0 += (e0 & 0xF) * p
        t1 += (e1 & 0xF) * p
        e0 >>= 4
        e1 >>= 4
        p *= 10
    year = t0 // 10000
    day = int((t0 - year * 10000) * 0.1)
    iaa = t0 - year * 10000
    iab = (iaa - day * 10) * 10
    nbc = t1 // 10000000
    iac = t1 - nbc * 10000000
    d = t1 - iac
    hour = iab + nbc
    minuts = int(iac * 0.00001)
    s = (t1 - (d + minuts * 100000)) * 0.001
    return jday(year, day) * 1440.0 + hour * 60.0 + minuts + s / 60.0


class Ref:
    def __init__(self, w, order="<"):
        w = [int(x) for x in w]
        iparms = [x if x < 2**31 else x - 2**32 for x in [y & 0xFFFFFFFF for y in w]]
        rp = [f32(0)] * 640
        for i in range(640):
            rp[i] = f32(bits2f(iparms[i], order)) if i < len(iparms) else f32(0)
        rp[368] = f32(f32(iparms[368]) / f32(1000.0))
        for lo, hi in RELLST:
            off = 1
            if lo > 115:
                off = 13
            if lo > 225:
                off = 31
            for j in range(lo, hi + 1):
                if j == 13 or j == 60 or ((j - 7) % 55 == 0 and j != 7):
                    rp[j + off] = f32(f32(iparms[j + off]) / f32(100.0))
                else:
                    rp[j + off] = f32(f32(iparms[j + off]) / f32(10000000.0))
        self.rp = rp
        self.ip = iparms
        FER = 1.0 - (6356.7533 / 6378.137)
        AE = 6378.137
        self.AEBE2 = float(f32(1.0 / (1.0 - FER) ** 2))
        self.AEBE3 = float(f32(f32(self.AEBE2) - f32(1.0)))
        self.AEBE4 = float(f32(f32((1.0 - FER) ** 4) - f32(1.0)))
        self.AE = AE
        self.instr = iparms[369]
        self.elvmax = [0.220896, 0.22089375]
        self.scnmax = [0.24544, 0.2454375]
        self.elvinc = [8e-6, 17.5e-6]
        self.scninc = [16e-6, 35e-6]
        self.elvln = [28e-6, 280e-6]
        self.scnpx = [16e-6, 280e-6]
        self.ewnom = [0.24544, 0.2454375]
        self.incmax = [6136, 2805]
        self.elvmax = [float(f32(x)) for x in self.elvmax]
        self.scnmax = [float(f32(x)) for x in self.scnmax]
        self.elvinc = [float(f32(x)) for x in self.elvinc]
        self.scninc = [float(f32(x)) for x in self.scninc]
        self.elvln = [float(f32(x)) for x in self.elvln]
        self.scnpx = [float(f32(x)) for x in self.scnpx]
        self.ewnom = [float(f32(x)) for x in self.ewnom]
        nsc, nsi, ewc, ewi = iparms[379], iparms[381], iparms[380], iparms[382]
        if nsc and nsi and ewc and ewi:
            if self.instr == 1:
                self.elvmax[0] = float(f32((nsc * self.incmax[0] + nsi) * self.elvinc[0]))
            else:
                self.elvmax[1] = float(f32(((9 - nsc) * self.incmax[1] - nsi) * self.elvinc[1]))
            k = self.instr - 1
            self.scnmax[k] = float(f32((ewc * self.incmax[k] + ewi) * self.scninc[k]))
        year = 1900 + iparms[367] // 1000
        day = iparms[367] - iparms[367] // 1000 * 1000
        t = float(rp[368])
        hour = int(t) // 10000
        minuts = int(t) // 100 - hour * 100
        secs = float(f32(rp[368] - f32(100) * minuts - f32(10000) * hour))
        imgtim = jday(year, day) * 1440.0 + hour * 60.0 + minuts + secs / 60.0
        epoch = bcd_epoch(iparms[12], iparms[13])
        imc = 0 if (iparms[2] & (1 << 7)) else 1
        self.iflip = -1 if (iparms[3] & (1 << 15)) else 1
        lam = float(rp[5])
        dr = float(rp[6])
        phi = float(rp[7])
        psi = float(rp[8])
        roll = float(rp[9])
        pitch = float(rp[10])
        yaw = float(rp[11])
        self.rma = 0.0
        self.pma = 0.0
        self.imc = imc
        self.imgtim = imgtim
        self.epoch = epoch
        if imc != 0:
            dr = phi = psi = 0.0
            ts = imgtim - epoch
            self.ts = ts
            w = 0.729115e-4 * 60.0 * ts
            def R(i):
                return float(rp[i])

            sw, cw = math.sin(w), math.cos(w)
            sw1, cw1 = math.sin(0.927 * w), math.cos(0.927 * w)
            s2w, c2w = math.sin(2 * w), math.cos(2 * w)
            sw3, cw3 = math.sin(1.9268 * w), math.cos(1.9268 * w)
            lam = (
                lam
                + R(18)
                + (R(19) + R(20) * w) * w
                + (
                    R(27) * sw1
                    + R(28) * cw1
                    + R(21) * sw
                    + R(22) * cw
                    + R(23) * s2w
                    + R(24) * c2w
                    + R(25) * sw3
                    + R(26) * cw3
                    + w * (R(29) * sw + R(30) * cw)
                )
                * 2.0
            )
            dr = (
                dr
                + R(31)
                + R(32) * cw
                + R(33) * sw
                + R(34) * c2w
                + R(35) * s2w
                + R(36) * cw3
                + R(37) * sw3
                + R(38) * cw1
                + R(39) * sw1
                + w * (R(40) * cw + R(41) * sw)
            )
            dlat = (
                R(42)
                + R(43) * cw
                + R(44) * sw
                + R(45) * c2w
                + R(46) * s2w
                + w * (R(47) * cw + R(48) * sw)
                + R(49) * cw1
                + R(50) * sw1
            )
            phi = phi + dlat * (1.0 + dlat * dlat / 6.0)
            dyaw = (
                R(51)
                + R(52) * sw
                + R(53) * cw
                + R(54) * s2w
                + R(55) * c2w
                + w * (R(56) * sw + R(57) * cw)
                + R(58) * sw1
                + R(59) * cw1
            )
            psi = psi + dyaw * (1.0 + dyaw * dyaw / 6.0)
        slat = math.sin(phi)
        syaw = math.sin(psi)
        sinoi = slat * slat + syaw * syaw
        cosoi = math.sqrt(1.0 - sinoi)
        sinoi = math.sqrt(sinoi)
        u = 0.0 if (slat == 0 and syaw == 0) else math.atan2(slat, syaw)
        sinu, cosu = math.sin(u), math.cos(u)
        asc = lam - u
        sinasc, cosasc = math.sin(asc), math.cos(asc)
        b = np.zeros((3, 3))
        b[0][1] = -sinasc * sinoi
        b[1][1] = cosasc * sinoi
        b[2][1] = -cosoi
        b[0][2] = -cosasc * cosu + sinasc * sinu * cosoi
        b[1][2] = -sinasc * cosu - cosasc * sinu * cosoi
        b[2][2] = -slat
        b[0][0] = -cosasc * sinu - sinasc * cosu * cosoi
        b[1][0] = -sinasc * sinu + cosasc * cosu * cosoi
        b[2][0] = cosu * sinoi
        r = (42164.365 + dr) / self.AE
        self.xs = [-b[0][2] * r, -b[1][2] * r, -b[2][2] * r]
        self.q3 = self.xs[0] ** 2 + self.xs[1] ** 2 + self.AEBE2 * self.xs[2] ** 2 - 1.0
        if imc != 0:
            wa = float(rp[60]) * ts
            te = ts - float(rp[61])
            roll = roll + self.gatt(62, wa, te)
            pitch = pitch + self.gatt(129, wa, te)
            yaw = yaw + self.gatt(184, wa, te)
            self.rma = float(f32(self.gatt(257, wa, te)))
            self.pma = float(f32(self.gatt(312, wa, te)))
            roll += float(rp[15])
            pitch += float(rp[16])
            yaw += float(rp[17])
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        rpy = np.array(
            [
                [1 - 0.5 * (pitch**2 + yaw**2), -yaw, pitch],
                [yaw + pitch * roll, 1 - 0.5 * (yaw**2 + roll**2), -roll],
                [-pitch + roll * yaw, roll + pitch * yaw, 1 - 0.5 * (pitch**2 + roll**2)],
            ]
        )
        self.bt = b @ rpy

    def gatt(self, k, wa, te):
        rp = self.rp
        ip = self.ip
        att = float(rp[k + 2])
        if te >= 0:
            att += float(rp[k]) * math.exp(-te / float(rp[k + 1]))
        for j in range(1, ip[k + 3] + 1):
            att += float(rp[k + 2 * j + 2]) * math.cos(wa * j + float(rp[k + 2 * j + 3]))
        k += 34
        kkk = ip[k]
        for j in range(1, kkk + 1):
            ll = k + 5 * j
            jr = float(ip[ll - 4])
            mr = float(ip[ll - 3])
            att += float(rp[ll - 2]) * math.pow(wa - float(rp[ll]), mr) * math.cos(jr * wa + float(rp[ll - 1]))
        return att

    def to_latlon(self, line, elem):
        """image coords (1-based as in Java 'image' coordinates)."""
        k = self.instr - 1
        rl, rpx = line, elem
        if self.instr == 2:
            rl = (rl + 9.0) / 10.0
            rpx = (rpx + 9.0) / 10.0
        a0 = (
            self.elvmax[0] - (rl - 4.5) * self.elvln[0]
            if self.instr == 1
            else self.elvmax[1] - (rl - 2.5) * self.elvln[1]
        )
        z0 = (rpx - 1.0) * self.scnpx[k] - self.scnmax[k]
        ff = float(self.iflip)
        ff = -ff if self.instr == 2 else ff
        doff = self.scnmax[k] - self.ewnom[k]
        alpha = a0 - a0 * z0 * doff
        zeta = z0 + 0.5 * a0 * a0 * doff
        ca, sa, cz = math.cos(alpha), math.sin(alpha), math.cos(zeta)
        da = alpha - self.pma * sa * (ff / cz + math.tan(zeta)) - self.rma * (1.0 - ca / cz)
        dz = zeta + ff * self.rma * sa
        cz = math.cos(dz)
        g = [math.sin(dz), -cz * math.sin(da), cz * math.cos(da)]
        bt = self.bt
        g1 = [sum(bt[i][j] * g[j] for j in range(3)) for i in range(3)]
        xs = self.xs
        A = self.AEBE2
        q1 = g1[0] ** 2 + g1[1] ** 2 + A * g1[2] ** 2
        q2 = xs[0] * g1[0] + xs[1] * g1[1] + A * xs[2] * g1[2]
        d = q2 * q2 - q1 * self.q3
        if abs(d) < 1e-9:
            d = 0.0
        if d < 0:
            return float("nan"), float("nan")
        d = math.sqrt(d)
        h = -(q2 + d) / q1
        u = [xs[i] + h * g1[i] for i in range(3)]
        d1 = u[2] / math.sqrt(sum(x * x for x in u))
        return math.degrees(math.atan(A * d1 / math.sqrt(1.0 - d1 * d1))), math.degrees(math.atan2(u[1], u[0]))
