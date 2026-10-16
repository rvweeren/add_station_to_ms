#!/usr/bin/env python3
# Copyright (C) 2026 Reinout van Weeren
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Add a new antenna (station) to a Measurement Set.

The new antenna is a copy of an existing "template" antenna (all ANTENNA, FEED,
LOFAR_ANTENNA_FIELD, POINTING, ... properties), placed at a user-given
geodetic position (WGS84, default height 0 m). A new output MS is written with
the extra baselines (i, NEW) inserted at the end of every timeslot.

For the new baselines:
  * UVW          : computed (see --uvw-method)
  * FLAG, WEIGHT_SPECTRUM, WEIGHT, SIGMA, scalar columns
                 : copied from the partner baseline (i, template)

The DATA of ALL cross baselines (existing and new) is replaced by complex
Gaussian noise with the noise properties of the input MS (see below).

Noise: for every data column the weight-normalised noise of every baseline
is measured (clipped rms of time and frequency differences, per channel and
polarisation). A station sensitivity model sigma(p,q) = s_p * s_q * shape(chan,pol)
is fitted to it, so that baselines to e.g. a superstation, international or
remote stations get their own noise level; the new station gets s of the
template. Noise per visibility is sigma0 / sqrt(weight).

The output MS has a single visibility column, DATA (taken from the input
column --datacolumn); other visibility columns of the input (MODEL_DATA,
CORRECTED_DATA, ...) are not written.

The input MS is opened read-only and never modified.

Example:
  add_station_to_ms.py --msin in.ms --msout out.ms --template DE604HBA \
      --name DE614HBA --position N53.04,E8.60
"""

import argparse
import gc
import math
import os
import re
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import casacore.tables as ct
from casacore.measures import measures
from casacore.quanta import quantity

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Storage managers that are built into casacore and can always be created.
BUILTIN_STMANS = ('StandardStMan', 'IncrementalStMan', 'TiledColumnStMan',
                  'TiledShapeStMan', 'TiledCellStMan')
ITEMSIZE = {'bool': 1, 'boolean': 1, 'uchar': 1, 'short': 2, 'int': 4, 'uint': 4,
            'int64': 8, 'float': 4, 'double': 8, 'complex': 8, 'dcomplex': 16}


_T0 = time.time()


def log(msg):
    print('[%7.1fs] %s' % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def parse_position(text):
    """Parse 'N53.04,E8.60', '53.04N,8.60E', 'S12.3,W45.6' or '53.04,8.60'.

    Returns (lat_deg, lon_deg)."""
    parts = [p for p in re.split(r'[,\s]+', text.strip()) if p]
    if len(parts) != 2:
        raise ValueError('position must have two components, e.g. N53.04,E8.60')
    lat = lon = None
    plain = []
    for p in parts:
        m = re.fullmatch(r'([NSEW]?)([+-]?\d+(?:\.\d*)?|[+-]?\.\d+)([NSEW]?)', p, re.I)
        if m is None or (m.group(1) and m.group(3)):
            raise ValueError('cannot parse position component %r' % p)
        letter = (m.group(1) or m.group(3)).upper()
        value = float(m.group(2))
        if letter in ('S', 'W'):
            value = -value
        if letter in ('N', 'S'):
            if lat is not None:
                raise ValueError('latitude given twice in %r' % text)
            lat = value
        elif letter in ('E', 'W'):
            if lon is not None:
                raise ValueError('longitude given twice in %r' % text)
            lon = value
        else:
            plain.append(value)
    for value in plain:  # components without a letter: lat first, then lon
        if lat is None:
            lat = value
        elif lon is None:
            lon = value
    if lat is None or lon is None:
        raise ValueError('need both latitude and longitude in %r' % text)
    if not -90 <= lat <= 90:
        raise ValueError('latitude out of range: %g' % lat)
    if not -180 <= lon <= 360:
        raise ValueError('longitude out of range: %g' % lon)
    return lat, lon


def geodetic_to_itrf(lat_deg, lon_deg, height):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([(n + height) * np.cos(lat) * np.cos(lon),
                     (n + height) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + height) * np.sin(lat)])


def itrf_to_geodetic(xyz):
    """Returns (lat_deg, lon_deg, height_m) on WGS84 (iterative)."""
    x, y, z = xyz
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    height = 0.0
    for _ in range(10):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        height = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
    return np.degrees(lat), np.degrees(lon), height


def enu_matrix(lat_deg, lon_deg):
    """Rows are the local East, North, Up unit vectors in ITRF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([[-np.sin(lon), np.cos(lon), 0.0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def procrustes(x, y):
    """Rotation R minimising |x R^T - y| for row vectors x, y (N x 3)."""
    u, _, vt = np.linalg.svd(y.T @ x)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


class UVWCalculator:
    """Per-antenna UVW (J2000, metres) with casacore measures."""

    def __init__(self, positions, phase_dir, dir_ref='J2000'):
        self.dm = measures()
        self.dm.do_frame(self.dm.position('ITRF', *[quantity(v, 'm') for v in positions[0]]))
        self.dm.do_frame(self.dm.direction(dir_ref, quantity(phase_dir[0], 'rad'),
                                           quantity(phase_dir[1], 'rad')))
        self.pos = self.dm.position('ITRF', quantity(positions[:, 0], 'm'),
                                    quantity(positions[:, 1], 'm'),
                                    quantity(positions[:, 2], 'm'))

    def __call__(self, mjd_sec):
        self.dm.do_frame(self.dm.epoch('UTC', quantity(mjd_sec, 's')))
        uvw = self.dm.to_uvw(self.dm.as_baseline(self.pos))['xyz'].get_value('m')
        return np.array(uvw).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def copy_row(tab, src_row, dst_row):
    for col in tab.colnames():
        if tab.iscelldefined(col, src_row):
            tab.putcell(col, dst_row, tab.getcell(col, src_row))


def rotate_vectors(arr, rot, up=None):
    """Rotate ITRF vectors stored in arr (..., 3) or a 3x3 axes matrix."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (3, 3) and up is not None:
        # Decide whether the axes are stored as rows or as columns: the third
        # axis is (close to) the local normal of the station.
        if abs(arr[2] @ up) >= abs(arr[:, 2] @ up):
            return arr @ rot.T
        return rot @ arr
    if arr.shape[-1] == 3:
        return arr @ rot.T
    if arr.shape[0] == 3:
        return rot @ arr
    raise ValueError('cannot rotate array of shape %s' % (arr.shape,))


def subtable_names(tab):
    names = []
    for key in tab.keywordnames():
        val = tab.getkeyword(key)
        if isinstance(val, str) and val.startswith('Table: '):
            names.append(key)
    return names


def cell_shape_casa(tab, col):
    """Cell shape in casacore (Fortran) order, or None if unknown."""
    desc = tab.getcoldesc(col)
    if 'shape' in desc and len(desc['shape']) > 0:
        return list(desc['shape'])[::-1]
    if tab.nrows() > 0 and tab.iscelldefined(col, 0):
        return list(np.asarray(tab.getcell(col, 0)).shape)[::-1]
    return None


def plain_dminfo(tab, columns):
    """Uncompressed data managers for the given columns."""
    dminfo = {}
    scalars = []
    for col in columns:
        desc = tab.getcoldesc(col)
        if desc.get('ndim', 0) == 0 and 'shape' not in desc:
            scalars.append(col)
            continue
        shape = cell_shape_casa(tab, col)
        key = '*%d' % (len(dminfo) + 1)
        if shape is None:
            dminfo[key] = {'TYPE': 'StandardStMan', 'NAME': 'PlainSSM_' + col,
                           'SPEC': {'BUCKETSIZE': 32768}, 'COLUMNS': [col]}
            continue
        itemsize = ITEMSIZE.get(desc.get('valueType', 'double'), 8)
        nrow_tile = max(1, int(1048576 // (itemsize * int(np.prod(shape)))))
        fixed = 'shape' in desc and (desc.get('option', 0) & 4)
        dminfo[key] = {'TYPE': 'TiledColumnStMan' if fixed else 'TiledShapeStMan',
                       'NAME': 'PlainTiled_' + col,
                       'SPEC': {'DEFAULTTILESHAPE': np.array(shape + [nrow_tile], dtype=np.int32)},
                       'COLUMNS': [col]}
    if scalars:
        dminfo['*%d' % (len(dminfo) + 1)] = {'TYPE': 'StandardStMan', 'NAME': 'PlainSSM',
                                             'SPEC': {'BUCKETSIZE': 32768}, 'COLUMNS': scalars}
    return dminfo


def clean_dminfo(dminfo):
    """Drop row-count dependent parts (hypercube shapes) of tiled specs."""
    out = {}
    for key, dm in dminfo.items():
        dm = dict(dm)
        spec = dict(dm.get('SPEC', {}))
        if dm['TYPE'] == 'TiledDataStMan':
            dm['TYPE'] = 'TiledShapeStMan'
        if dm['TYPE'] == 'AntennaPairStMan':
            # DP3 antenna compression needs a row-interleaved write order
            dm['TYPE'] = 'StandardStMan'
            spec = {'BUCKETSIZE': 32768}
        if dm['TYPE'] == 'UvwStMan':
            # DP3 UVW compression stores per-antenna UVWs and requires
            # antenna-consistent baselines (not true for e.g. superstations):
            # always write UVW uncompressed (like msout.uvwcompression=false).
            dm['TYPE'] = 'TiledShapeStMan'
            spec = {'DEFAULTTILESHAPE': np.array([3, 16384], dtype=np.int32)}
        if dm['TYPE'].startswith('Tiled'):
            spec = {k: v for k, v in spec.items() if k in ('DEFAULTTILESHAPE', 'MAXIMUMCACHESIZE')}
            if 'DEFAULTTILESHAPE' not in spec:
                dm['TYPE'] = 'StandardStMan'
                spec = {'BUCKETSIZE': 32768}
        dm['SPEC'] = spec
        out[key] = dm
    return out


def remove_table_dir(path):
    gc.collect()
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        raise RuntimeError('could not remove %s' % path)


def select_dminfo(dminfo, drop, rename):
    """Remove dropped columns from a dminfo and rename columns."""
    out = {}
    for key, dm in dminfo.items():
        cols = [rename.get(c, c) for c in dm['COLUMNS'] if c not in drop]
        if cols:
            dm = dict(dm)
            dm['COLUMNS'] = cols
            out[key] = dm
    return out


def make_output(tin, msout, storage, nrow, drop=(), rename=None):
    """Create msout with the structure of tin (without the columns in drop,
    with columns renamed according to rename), nrow main rows and full
    copies of all subtables. Returns the open (writable) output table."""
    rename = rename or {}
    desc = tin.getdesc()
    desc.pop('_keywords_', None)
    for col in drop:
        desc.pop(col, None)
    for old, new in rename.items():
        desc[new] = desc.pop(old)
    keep_cols = [c for c in tin.colnames() if c not in drop]
    attempts = []
    if storage == 'same':
        dmi = clean_dminfo(tin.getdminfo())
        attempts.append(('same as input', dmi))
        nonbuiltin = [dm for dm in dmi.values() if dm['TYPE'] not in BUILTIN_STMANS]
        if nonbuiltin:
            # Fallback: replace non-builtin managers (Dysco, LofarStMan, ...) by plain ones.
            keep = {k: dm for k, dm in dmi.items() if dm['TYPE'] in BUILTIN_STMANS}
            cols = [c for dm in nonbuiltin for c in dm['COLUMNS']]
            for k, dm in plain_dminfo(tin, [c for c in cols if c not in drop]).items():
                keep['*P%s' % k[1:]] = dm
            attempts.append(('input, non-builtin managers replaced by plain ones', keep))
    attempts.append(('plain', plain_dminfo(tin, keep_cols)))

    for label, dmi in attempts:
        dmi = select_dminfo(dmi, drop, rename)
        try:
            tout = ct.table(msout, desc, nrow=nrow, dminfo=dmi, ack=False)
            log('Output storage managers: %s' % label)
            break
        except Exception as exc:  # noqa: BLE001
            log('WARNING: creating output with storage "%s" failed: %s' % (label, exc))
            remove_table_dir(msout)
    else:
        raise RuntimeError('could not create output table %s' % msout)

    tout.putinfo(tin.info())
    for key in tin.keywordnames():
        val = tin.getkeyword(key)
        if isinstance(val, str) and val.startswith('Table: '):
            src = val[len('Table: '):]
            dst = os.path.join(msout, key)
            tsrc = ct.table(src, ack=False)
            tsrc.copy(dst, deep=True, valuecopy=True).close()
            tsrc.close()
            tout.putkeyword(key, 'Table: ' + dst)
        else:
            tout.putkeyword(key, val)
    for dm in tout.getdminfo().values():
        log('  %-20s %-28s %s' % (dm['TYPE'], dm['NAME'], ', '.join(dm['COLUMNS'])))
    # Keep the table open: some storage managers (Dysco) cannot reopen a
    # file that was closed before any data was written.
    return tout


# ---------------------------------------------------------------------------
# Main table layout
# ---------------------------------------------------------------------------

class Layout:
    """Regular MS layout: identical list of baselines in every timeslot."""

    def __init__(self, tin):
        nrow = tin.nrows()
        self.a1 = tin.getcol('ANTENNA1')
        self.a2 = tin.getcol('ANTENNA2')
        times = tin.getcol('TIME')
        ddid = tin.getcol('DATA_DESC_ID')
        if len(np.unique(ddid)) != 1:
            raise RuntimeError('only MSs with a single DATA_DESC_ID are supported')
        change = np.flatnonzero(np.diff(times) != 0) + 1
        starts = np.concatenate([[0], change])
        sizes = np.diff(np.concatenate([starts, [nrow]]))
        if len(np.unique(sizes)) != 1:
            raise RuntimeError('timeslots have different numbers of rows; MS is not regular')
        self.nbl = int(sizes[0])
        self.ntime = nrow // self.nbl
        a1 = self.a1.reshape(self.ntime, self.nbl)
        a2 = self.a2.reshape(self.ntime, self.nbl)
        if not ((a1 == a1[0]).all() and (a2 == a2[0]).all()):
            raise RuntimeError('baseline order differs between timeslots; MS is not regular')
        if np.any(np.diff(times[starts]) <= 0):
            raise RuntimeError('MS is not sorted in time')
        self.a1 = a1[0].copy()
        self.a2 = a2[0].copy()
        self.times = times[starts]
        self.bl_index = {(int(p), int(q)): k for k, (p, q) in enumerate(zip(self.a1, self.a2))}

    def find(self, p, q):
        """Return (row in slot, swapped) for baseline p-q, or (None, None)."""
        if (p, q) in self.bl_index:
            return self.bl_index[(p, q)], False
        if (q, p) in self.bl_index:
            return self.bl_index[(q, p)], True
        return None, None


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

CLIP_NSIGMA = 3.5
# rms of a unit Gaussian truncated at +-CLIP_NSIGMA
CLIP_CORR = np.sqrt(1.0 - 2.0 * CLIP_NSIGMA * np.exp(-0.5 * CLIP_NSIGMA ** 2) / np.sqrt(2.0 * np.pi)
                    / math.erf(CLIP_NSIGMA / np.sqrt(2.0)))


def nanmedian(x, axis=0):
    """Median along an axis ignoring NaN (fast replacement of np.nanmedian,
    which uses slow masked arrays for short axes)."""
    x = np.moveaxis(np.asarray(x), axis, -1)
    n = np.count_nonzero(~np.isnan(x), axis=-1)
    xs = np.sort(x, axis=-1)  # NaN sorts to the end
    lo = np.maximum((n - 1) // 2, 0)[..., None]
    hi = np.maximum(n // 2, 0)[..., None]
    med = 0.5 * (np.take_along_axis(xs, lo, -1)[..., 0] + np.take_along_axis(xs, hi, -1)[..., 0])
    return np.where(n > 0, med, np.nan)


def robust_std(x, axis=0, nsigma=CLIP_NSIGMA, niter=3, min_count=8):
    """Clipped rms along an axis, ignoring NaN.

    Starts from the MAD, then iteratively clips at nsigma and recomputes the
    rms (corrected for the truncation of a Gaussian). Returns NaN where fewer
    than min_count samples remain."""
    with np.errstate(all='ignore'):
        x = np.ascontiguousarray(np.moveaxis(x, axis, -1))
        med = nanmedian(x, axis=-1)
        sig = 1.4826 * nanmedian(np.abs(x - med[..., None]), axis=-1)
        for _ in range(niter):
            keep = np.abs(x - med[..., None]) <= (nsigma * sig)[..., None]  # False for NaN
            count = np.count_nonzero(keep, axis=-1)
            med = np.sum(np.where(keep, x, 0), axis=-1, dtype=np.float64) / count
            dev = np.where(keep, x - med[..., None], 0)
            sig = np.sqrt(np.sum(dev * dev, axis=-1, dtype=np.float64) / count) / CLIP_CORR
        return np.where(count >= min_count, sig, np.nan)


def read_weights(tin, weight_col, startrow, nrow, rowincr, cell_shape):
    """Weights broadcast to (nrow, nchan, npol)."""
    if weight_col == 'WEIGHT_SPECTRUM':
        return tin.getcol('WEIGHT_SPECTRUM', startrow, nrow, rowincr).astype(np.float32)
    if weight_col == 'WEIGHT':
        w = tin.getcol('WEIGHT', startrow, nrow, rowincr).astype(np.float32)
        return np.broadcast_to(w[:, None, :], (nrow,) + tuple(cell_shape)).copy()
    return np.ones((nrow,) + tuple(cell_shape), dtype=np.float32)


def estimate_sigma0(data, flags, weights):
    """Weight-normalised noise estimates of visibilities.

    Axis 0 is time, the last two axes are channel and polarisation. The data
    are first normalised by sqrt(weight), so that pure noise has the same
    sigma0 everywhere: sigma(t, ch, pol) = sigma0 / sqrt(w(t, ch, pol)).
    Sky signal is suppressed by differencing; every estimate is a clipped
    rms of the real and imaginary parts. Returns a dict with arrays
    (..., nchan, npol) for 't' (time differences), 'f' (frequency
    differences), 'tf' (time-frequency double differences), and for 4-pol
    data 'x' (..., nchan): double differences of the cross-hand difference
    p1 - p2 (Stokes U for circular, V for linear feeds), in which the sky is
    nearly absent."""
    valid = (~flags) & (weights > 0) & np.isfinite(data)
    est = {}
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = np.where(valid, data * np.sqrt(weights), np.nan).astype(np.complex64)

        def rstd(x):
            return robust_std(np.concatenate([x.real, x.imag], axis=0), axis=0)

        def to_chan(pair):  # estimates per channel pair -> per channel
            out = np.empty(pair.shape[:-2] + (pair.shape[-2] + 1,) + pair.shape[-1:], pair.dtype)
            out[..., 0, :] = pair[..., 0, :]
            out[..., -1, :] = pair[..., -1, :]
            if out.shape[-2] > 2:
                out[..., 1:-1, :] = np.nanmean(np.stack([pair[..., :-1, :], pair[..., 1:, :]]), axis=0)
            return out

        nchan = v.shape[-2]
        est['t'] = rstd((v[1:] - v[:-1]) / np.sqrt(2.0))
        if nchan > 1:
            est['f'] = to_chan(rstd((v[..., 1:, :] - v[..., :-1, :]) / np.sqrt(2.0)))
            est['tf'] = to_chan(rstd((v[1:, ..., 1:, :] - v[:-1, ..., 1:, :]
                                      - v[1:, ..., :-1, :] + v[:-1, ..., :-1, :]) / 2.0))
        if v.shape[-1] == 4:
            x = (v[..., 1:2] - v[..., 2:3]) / np.sqrt(2.0)
            if nchan > 1:
                xd = (x[1:, ..., 1:, :] - x[:-1, ..., 1:, :] - x[1:, ..., :-1, :] + x[:-1, ..., :-1, :]) / 2.0
                est['x'] = to_chan(rstd(xd))[..., 0]
            else:
                est['x'] = rstd((x[1:] - x[:-1]) / np.sqrt(2.0))[..., 0]
    return est


def parallel_pols(npol):
    return [0, 3] if npol == 4 else list(range(npol))


def combine_estimates(est):
    """Combine the estimates of estimate_sigma0 into a noise level per
    baseline (sigma0 scalar, ...) and a per (channel, pol) estimate.

    Per baseline and polarisation the differencing method with the lowest
    median is taken (sky signal only increases the estimates). The scalar
    level is min(parallel hands, cross-hand difference)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        methods = [m for m in ('t', 'f', 'tf') if m in est]
        stack = np.stack([est[m] for m in methods])           # (nm, ..., nchan, npol)
        med = np.nanmedian(stack, axis=-2)                       # (nm, ..., npol)
        med = np.where(np.isfinite(med), med, np.inf)
        best = np.argmin(med, axis=0)                            # (..., npol)
        per_pol = np.take_along_axis(stack, best[None, ..., None, :], axis=0)[0]
        npol = per_pol.shape[-1]
        par = np.nanmedian(per_pol[..., parallel_pols(npol)], axis=(-2, -1))
        level = par
        cross = np.full_like(par, np.nan)
        if 'x' in est:
            cross = np.nanmedian(est['x'], axis=-1)
            level = np.fmin(par, cross)
    return level, per_pol, par, cross


def estimate_sigma0_all(tin, lay, col, weight_col, has_flag, nslot_chunk, ncpu=1):
    """Noise of every baseline, see estimate_sigma0 / combine_estimates.

    The estimates are made per time chunk and the median over chunks is
    taken. Each chunk is split in baseline groups that are processed by ncpu
    threads, while the next chunk is read. Returns (level (nbl,), per_pol
    (nbl, nchan, npol), par, cross)."""
    nchunk = max(1, int(np.ceil(lay.ntime / max(nslot_chunk, 1))))
    nchunk = min(nchunk, max(1, lay.ntime // 20))  # keep >= ~20 timeslots per chunk
    groups = [(g[0], g[-1] + 1) for g in np.array_split(np.arange(lay.nbl), min(ncpu, lay.nbl))]
    pool = ThreadPoolExecutor(ncpu) if ncpu > 1 else None
    chunks, pending = [], []

    def collect(futures):
        parts = [fu.result() for fu in futures]
        chunks.append({m: np.concatenate([p[m] for p in parts], axis=0) for m in parts[0]})

    try:
        for slots in np.array_split(np.arange(lay.ntime), nchunk):
            s0, ns = int(slots[0]), len(slots)
            r0, nr = s0 * lay.nbl, ns * lay.nbl
            d = tin.getcol(col, r0, nr).astype(np.complex64)
            cshape = d.shape[1:]
            d = d.reshape((ns, lay.nbl) + cshape)
            f = tin.getcol('FLAG', r0, nr).reshape(d.shape) if has_flag else np.zeros(d.shape, bool)
            w = read_weights(tin, weight_col, r0, nr, 1, cshape).reshape(d.shape)
            if pool is None:
                chunks.append(estimate_sigma0(d, f, w))
            else:
                pending.append([pool.submit(estimate_sigma0, d[:, a:b], f[:, a:b], w[:, a:b])
                                for a, b in groups])
                while len(pending) > 1:  # at most one chunk in flight while reading the next
                    collect(pending.pop(0))
            del d, f, w
        while pending:
            collect(pending.pop(0))
    finally:
        if pool is not None:
            pool.shutdown()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        est = {m: nanmedian(np.stack([c[m] for c in chunks]), axis=0) for m in chunks[0]}
    return combine_estimates(est)


def fit_station_noise(level, per_pol, par, cross, a1, a2, nant):
    """Fit a per-station sensitivity model to the per-baseline noise.

    sigma0(p, q, ch, pol) = exp(g_p + g_q) * shape(ch, pol), i.e. the baseline
    noise is proportional to sqrt(SEFD_p * SEFD_q) (after normalisation by
    the weights). The fit is done in log space with iterative outlier
    rejection. shape(ch, pol) is the median normalised spectrum of the
    baselines whose parallel- and cross-hand estimates agree (noise dominated).
    Returns (g (nant, NaN for unconstrained stations), shape, info dict)."""
    ok = np.isfinite(level) & (level > 0) & (a1 != a2)
    if ok.sum() == 0:
        raise RuntimeError('no baseline with a valid noise estimate')
    ants = np.unique(np.concatenate([a1[ok], a2[ok]]))
    col = {a: j for j, a in enumerate(ants)}
    rows = np.flatnonzero(ok)
    design = np.zeros((len(rows), len(ants)))
    design[np.arange(len(rows)), [col[a] for a in a1[rows]]] += 1
    design[np.arange(len(rows)), [col[a] for a in a2[rows]]] += 1
    y = np.log(level[rows])
    sel = np.ones(len(rows), bool)
    for _ in range(10):
        sol = np.linalg.lstsq(design[sel], y[sel], rcond=None)[0]
        res = y - design @ sol
        scatter = 1.4826 * np.median(np.abs(res[sel]))
        new_sel = np.abs(res) <= max(4.0 * scatter, 0.02)
        if (new_sel == sel).all() or new_sel.sum() < len(ants):
            break
        sel = new_sel
    g = np.full(nant, np.nan)
    g[ants] = sol
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        clean = ok & (np.abs(np.log(par / cross)) < 0.1)
        if clean.sum() < 5:
            clean = ok
        shape = np.nanmedian(per_pol[clean] / level[clean, None, None], axis=0)
    shape = fill_nan(shape)
    shape = np.where(np.isfinite(shape), shape, 1.0)
    nbase = np.array([np.sum(((a1[rows] == a) | (a2[rows] == a)) & sel) for a in range(nant)])
    info = {'scatter': scatter, 'nused': int(sel.sum()), 'nrej': int((~sel).sum()),
            'rejected': rows[~sel], 'resid': dict(zip(rows, res)), 'nbase': nbase,
            'nclean': int(clean.sum())}
    return g, shape, info


def fill_nan(sig):
    """Fill NaN channels with the per-pol median over channels."""
    sig = sig.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(sig, axis=0)
    for p in range(sig.shape[1]):
        bad = ~np.isfinite(sig[:, p])
        sig[bad, p] = med[p]
    return sig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Add a new antenna (copy of a template antenna at a new position) '
                    'to a Measurement Set; DATA is replaced by Gaussian noise with the noise '
                    'properties of the input MS.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--msin', required=True, help='input MS (read only)')
    ap.add_argument('--msout', required=True, help='output MS (must not exist, see --overwrite)')
    ap.add_argument('--template', required=True,
                    help='name (or index) of the antenna whose properties are copied, e.g. DE604HBA')
    ap.add_argument('--name', required=True, help='name of the new antenna, e.g. DE614HBA')
    ap.add_argument('--position', required=True,
                    help='geodetic WGS84 position, e.g. N53.04,E8.60 (S/W for south/west)')
    ap.add_argument('--height', type=float, default=0.0, help='height above the WGS84 ellipsoid (m)')
    ap.add_argument('--seed', type=int, default=None, help='random seed for the noise')
    ap.add_argument('--noise-scale', type=float, default=1.0, help='multiply the estimated noise by this')
    ap.add_argument('--datacolumn', default='DATA',
                    help='input visibility column that is written as DATA to the output')
    ap.add_argument('--noise-model', choices=('station', 'baseline'), default='station',
                    help='station: sigma(p,q) = s_p * s_q * shape(chan,pol) with per-station '
                         'sensitivities s fitted to the noise of all baselines (the new station gets '
                         's of the template); baseline: use the noise measured on each baseline '
                         '(new baselines: on partner baseline (i, template)), with the station '
                         'model only for baselines without a valid estimate')
    ap.add_argument('--uvw-method', choices=('hybrid', 'exact'), default='hybrid',
                    help='hybrid: uvw(i,NEW) = stored uvw(i,template) + computed (NEW - template) '
                         'vector, rotated to match the stored UVW convention per timeslot; '
                         'exact: compute all new UVWs from scratch with casacore')
    ap.add_argument('--storage', choices=('same', 'plain'), default='same',
                    help='same: use the storage managers of the input (e.g. Dysco); '
                         'plain: uncompressed storage managers')
    ap.add_argument('--chunk-mb', type=float, default=512.0,
                    help='approximate memory size per data column chunk (MB)')
    ap.add_argument('--ncpu', type=int, default=0,
                    help='number of threads for the noise estimation (default 0: number of '
                         'available cores, at most 16)')
    ap.add_argument('--overwrite', action='store_true', help='remove msout if it exists')
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    msin = os.path.abspath(args.msin.rstrip('/'))
    msout = os.path.abspath(args.msout.rstrip('/'))
    if os.path.realpath(msin) == os.path.realpath(msout):
        sys.exit('ERROR: --msout must differ from --msin')
    if not os.path.isfile(os.path.join(msin, 'table.dat')):
        sys.exit('ERROR: %s is not a casacore table' % msin)
    if os.path.exists(msout):
        if not args.overwrite:
            sys.exit('ERROR: %s exists (use --overwrite)' % msout)
        if not os.path.isfile(os.path.join(msout, 'table.dat')):
            sys.exit('ERROR: %s exists and is not a table; refusing to remove it' % msout)
        log('Removing existing %s' % msout)
        shutil.rmtree(msout)

    lat, lon = parse_position(args.position)
    new_pos = geodetic_to_itrf(lat, lon, args.height)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    tin = ct.table(msin, readonly=True, ack=False)

    # ---------------- antennas ----------------
    tant = ct.table(os.path.join(msin, 'ANTENNA'), readonly=True, ack=False)
    names = list(tant.getcol('NAME'))
    positions = tant.getcol('POSITION')
    tant.close()
    if args.template in names:
        tmpl = names.index(args.template)
    elif args.template.upper() in [n.upper() for n in names]:
        tmpl = [n.upper() for n in names].index(args.template.upper())
    elif args.template.isdigit() and int(args.template) < len(names):
        tmpl = int(args.template)
    else:
        sys.exit('ERROR: template antenna %s not found; antennas: %s' % (args.template, ', '.join(names)))
    if args.name.upper() in [n.upper() for n in names]:
        sys.exit('ERROR: antenna %s already exists in the MS' % args.name)
    nant_old = len(names)
    new = nant_old
    tmpl_pos = positions[tmpl]
    tlat, tlon, theight = itrf_to_geodetic(tmpl_pos)
    rot_enu = enu_matrix(lat, lon).T @ enu_matrix(tlat, tlon)
    log('Template antenna : %d %s  (lat %.5f, lon %.5f, h %.1f m)' % (tmpl, names[tmpl], tlat, tlon, theight))
    log('New antenna      : %d %s  (lat %.5f, lon %.5f, h %.1f m)' % (new, args.name, lat, lon, args.height))
    log('New ITRF position: %.3f %.3f %.3f m' % tuple(new_pos))
    log('Distance to template: %.3f km' % (np.linalg.norm(new_pos - tmpl_pos) / 1e3))

    # ---------------- layout and new baselines ----------------
    lay = Layout(tin)
    log('Input: %d rows, %d timeslots x %d baselines' % (tin.nrows(), lay.ntime, lay.nbl))
    present = sorted(set(lay.a1.tolist()) | set(lay.a2.tolist()))
    if tmpl not in present:
        sys.exit('ERROR: template antenna has no baselines in the main table')
    has_autos = bool(np.any(lay.a1 == lay.a2))

    tmpl_rows = []  # rows in slot of all cross baselines with the template, with swap flag (template as ANTENNA2)
    for i in present:
        if i == tmpl:
            continue
        k, swapped = lay.find(i, tmpl)
        if k is not None:
            tmpl_rows.append((k, swapped))
    if not tmpl_rows:
        sys.exit('ERROR: no cross baselines with the template antenna')

    # new rows: (ant1, ant2, partner row in slot or -1, swapped, kind)
    new_bl = []
    for i in present:
        if i == tmpl:
            new_bl.append((tmpl, new, -1, False, 'template'))
            continue
        k, swapped = lay.find(i, tmpl)
        if k is None:
            log('WARNING: no baseline %s-%s in the MS, baseline %s-%s is not added'
                % (names[i], names[tmpl], names[i], args.name))
            continue
        new_bl.append((i, new, k, swapped, 'cross'))
    if has_autos:
        k, _ = lay.find(tmpl, tmpl)
        if k is not None:
            new_bl.append((new, new, k, False, 'auto'))
    nnew = len(new_bl)
    nbl_new = lay.nbl + nnew
    nrow_new = lay.ntime * nbl_new
    log('Adding %d baselines per timeslot -> %d baselines, %d rows' % (nnew, nbl_new, nrow_new))

    # ---------------- columns ----------------
    colnames = tin.colnames()
    defined = [c for c in colnames if tin.nrows() > 0 and tin.iscelldefined(c, 0)]
    skipped = [c for c in colnames if c not in defined]
    if skipped:
        log('Columns without data (left empty): %s' % ', '.join(skipped))
    complex_cols = [c for c in defined
                    if tin.getcoldesc(c).get('valueType') in ('complex', 'dcomplex')
                    and np.ndim(tin.getcell(c, 0)) == 2]
    if args.datacolumn not in complex_cols:
        sys.exit('ERROR: visibility column %s not found in the input; available: %s'
                 % (args.datacolumn, ', '.join(complex_cols)))
    # The output has a single visibility column: DATA (from --datacolumn).
    drop = [c for c in complex_cols if c != args.datacolumn]
    rename = {args.datacolumn: 'DATA'} if args.datacolumn != 'DATA' else {}
    if drop:
        log('Visibility columns not written to the output: %s' % ', '.join(drop))
    if rename:
        log('Input column %s is written as DATA' % args.datacolumn)
    defined = [c for c in defined if c not in drop]
    complex_cols = [args.datacolumn]
    noise_cols = complex_cols
    if 'WEIGHT_SPECTRUM' in defined:
        weight_col = 'WEIGHT_SPECTRUM'
    elif 'WEIGHT' in defined:
        weight_col = 'WEIGHT'
    else:
        weight_col = None
    log('Weights taken from: %s' % (weight_col or 'none (all weights 1)'))
    cell_shape = np.asarray(tin.getcell(complex_cols[0] if complex_cols else 'FLAG', 0)).shape
    npol = cell_shape[-1]
    polswap = np.array([0, 2, 1, 3]) if npol == 4 else np.arange(npol)

    # ---------------- noise estimation ----------------
    # sigma0[col]: weight-normalised noise (nbl_new, nchan, npol) for every
    # output baseline; sigma(t) = sigma0 / sqrt(weight(t)).
    row_bytes = max([np.asarray(tin.getcell(c, 0)).nbytes for c in defined] + [1])
    chunk = max(1, int(args.chunk_mb * 1e6 // (row_bytes * (lay.nbl + nnew))))
    out_a1 = np.concatenate([lay.a1, [b[0] for b in new_bl]]).astype(int)
    out_a2 = np.concatenate([lay.a2, [b[1] for b in new_bl]]).astype(int)
    try:
        ncpu_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu_avail = os.cpu_count() or 1
    ncpu = args.ncpu if args.ncpu > 0 else min(ncpu_avail, 16)
    sigma0 = {}
    for col in noise_cols:
        log('Estimating noise of %s on all baselines (clipped rms of time/frequency differences, '
            '%d thread(s))' % (col, ncpu))
        level, per_pol, par, cross = estimate_sigma0_all(tin, lay, col, weight_col, 'FLAG' in defined,
                                                         max(20, chunk // 4), ncpu)
        g, shape, info = fit_station_noise(level, per_pol, par, cross, lay.a1, lay.a2, nant_old)
        bad_g = ~np.isfinite(g[present])
        if bad_g.any():
            log('WARNING: no valid noise estimate for station(s) %s (fully flagged?); '
                'using the median sensitivity' % ', '.join(names[a] for a in np.array(present)[bad_g]))
        g = np.where(np.isfinite(g), g, np.nanmedian(g))
        g = np.append(g, g[tmpl])  # new station has the sensitivity of the template
        s_rel = np.exp(g - np.median(g[present]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            log('  %s: weight-normalised noise sigma0: median %.4g; parallel/cross-hand estimate ratio '
                'median %.3f (%d noise-dominated baselines used for the channel/pol shape)'
                % (col, np.nanmedian(level), np.nanmedian(par / cross), info['nclean']))
        log('  station noise model sigma0(p,q) = s_p*s_q*shape(chan,pol): fit uses %d baselines, '
            '%d rejected as outliers, residual scatter %.1f%%'
            % (info['nused'], info['nrej'], 100 * info['scatter']))
        log('  (s_rel: relative sensitivity factor on top of the weights; the weights themselves')
        log('   already carry station sensitivity if the MS was calibrated with weight updates)')
        log('  %-10s %8s %8s %6s' % ('station', 's_rel', 'SEFD_rel', 'n_bl'))
        for a in present + [new]:
            log('  %-10s %8.3f %8.3f %6s' % (names[a] if a < nant_old else args.name, s_rel[a],
                                             s_rel[a] ** 2, info['nbase'][a] if a < nant_old else 'new'))
        for k in info['rejected']:
            log('    outlier baseline %s-%s: measured/model %.3f'
                % (names[lay.a1[k]], names[lay.a2[k]], np.exp(info['resid'][k])))
        model = np.exp(g[out_a1] + g[out_a2])[:, None, None] * shape[None]
        sig = model.copy()
        if args.noise_model == 'baseline':
            # measured level per baseline, global channel/pol shape
            lev_out = np.full(len(out_a1), np.nan)
            lev_out[:lay.nbl] = level
            for j, (p, q, k, swapped, kind) in enumerate(new_bl):
                if kind == 'cross':
                    lev_out[lay.nbl + j] = level[k]
            sig = np.where(np.isfinite(lev_out)[:, None, None], lev_out[:, None, None] * shape[None], model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ratio = level / np.median(model[:lay.nbl], axis=(1, 2))
        log('  measured/model noise over existing baselines: median %.3f, 5-95%% range %.3f-%.3f'
            % (np.nanmedian(ratio), np.nanpercentile(ratio, 5), np.nanpercentile(ratio, 95)))
        sigma0[col] = sig * args.noise_scale

    # ---------------- create output ----------------
    log('Creating %s' % msout)
    tout = make_output(tin, msout, args.storage, nrow_new, drop, rename)
    update_subtables(msin, msout, tmpl, new, args, new_pos, rot_enu, tlat, tlon)

    # ---------------- UVW setup ----------------
    tfield = ct.table(os.path.join(msin, 'FIELD'), readonly=True, ack=False)
    field_id = int(tin.getcell('FIELD_ID', 0))
    phase_dir = tfield.getcell('PHASE_DIR', field_id)[0]
    dir_ref = tfield.getcoldesc('PHASE_DIR').get('keywords', {}).get('MEASINFO', {}).get('Ref', 'J2000')
    tfield.close()
    all_pos = np.vstack([positions, new_pos])
    uvwcalc = UVWCalculator(all_pos, phase_dir, dir_ref)
    cross_old = lay.a1 != lay.a2
    rot_angles, rot_resid, exact_resid = [], [], []

    # ---------------- write main table ----------------
    # Write order: index columns first (needed by e.g. Dysco), then the other
    # scalars, UVW/flags/weights, and finally the visibility columns (which
    # need the new weights).
    first = ['ANTENNA1', 'ANTENNA2', 'TIME', 'FIELD_ID', 'DATA_DESC_ID']
    weight_like = ['UVW', 'FLAG', 'FLAG_ROW', 'WEIGHT', 'SIGMA', 'WEIGHT_SPECTRUM']

    def write_rank(c):
        if c in first:
            return (0, first.index(c))
        if c in complex_cols:
            return (4, 0)
        if c in weight_like:
            return (2, weight_like.index(c))
        return (1 if np.ndim(tin.getcell(c, 0)) == 0 else 3, 0)
    ordered = sorted(defined, key=write_rank)
    partner = np.array([k if k >= 0 else tmpl_rows[0][0] for (_, _, k, _, _) in new_bl])
    swap_new = np.array([s for (_, _, _, s, _) in new_bl])
    special = np.array([kind == 'template' for (*_, kind) in new_bl])
    trows = np.array([k for k, _ in tmpl_rows])
    tswap = np.array([s for _, s in tmpl_rows])
    noise_rows = np.flatnonzero(out_a1 != out_a2)
    log('%s of all %d cross baselines is replaced by noise' % (', '.join(noise_cols), len(noise_rows)))
    ant1_new = np.array([b[0] for b in new_bl], dtype=np.int32)
    ant2_new = np.array([b[1] for b in new_bl], dtype=np.int32)
    log('Writing %d timeslots in chunks of %d' % (lay.ntime, chunk))

    for s0 in range(0, lay.ntime, chunk):
        ns = min(chunk, lay.ntime - s0)
        r_in, n_in = s0 * lay.nbl, ns * lay.nbl
        r_out, n_out = s0 * nbl_new, ns * nbl_new
        weights_all = None
        for col in ordered:
            old = tin.getcol(col, r_in, n_in)
            old = old.reshape((ns, lay.nbl) + old.shape[1:])
            arr = np.empty((ns, nbl_new) + old.shape[2:], dtype=old.dtype)
            arr[:, :lay.nbl] = old
            arr[:, lay.nbl:] = old[:, partner]
            pol_dep = old.ndim >= 3 and old.shape[-1] == npol and col != 'UVW'
            if pol_dep and swap_new.any():
                sw = old[:, partner][:, swap_new]
                arr[:, lay.nbl:][:, swap_new] = sw[..., polswap]

            if col == 'ANTENNA1':
                arr[:, lay.nbl:] = ant1_new
            elif col == 'ANTENNA2':
                arr[:, lay.nbl:] = ant2_new
            elif col == 'UVW':
                arr[:, lay.nbl:] = compute_new_uvw(
                    old, lay, s0, ns, uvwcalc, tmpl, new, new_bl, cross_old,
                    args.uvw_method, rot_angles, rot_resid, exact_resid)
            elif col in ('FLAG', 'FLAG_ROW') and special.any():
                tv = old[:, trows]
                if col == 'FLAG' and tswap.any():
                    tv = tv.copy()
                    tv[:, tswap] = tv[:, tswap][..., polswap]
                arr[:, lay.nbl:][:, special] = tv.all(axis=1)[:, None]
            elif col in ('WEIGHT_SPECTRUM', 'WEIGHT', 'SIGMA') and special.any():
                tv = old[:, trows].astype(np.float64)
                if tswap.any():
                    tv[:, tswap] = tv[:, tswap][..., polswap]
                arr[:, lay.nbl:][:, special] = np.median(tv, axis=1)[:, None].astype(old.dtype)
            if col == weight_col:
                weights_all = arr

            if col in noise_cols:
                # rows that get noise: new cross baselines, or all cross baselines
                rows = noise_rows
                cshape = arr.shape[2:]
                if weight_col == 'WEIGHT_SPECTRUM':
                    w = weights_all[:, rows].astype(np.float64)
                elif weight_col == 'WEIGHT':
                    w = np.broadcast_to(weights_all[:, rows][:, :, None, :].astype(np.float64),
                                        (ns, len(rows)) + cshape)
                else:
                    w = np.ones((ns, len(rows)) + cshape)
                with np.errstate(all='ignore'):
                    sig = np.where(w > 0, sigma0[col][rows][None] / np.sqrt(w), 0.0)
                sig = np.where(np.isfinite(sig), sig, 0.0)
                noise = sig * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
                arr[:, rows] = noise.astype(arr.dtype)
            tout.putcol(rename.get(col, col), arr.reshape((n_out,) + arr.shape[2:]), r_out, n_out)
        log('  timeslots %d-%d of %d done' % (s0, s0 + ns - 1, lay.ntime))

    if rot_angles:
        log('UVW (%s): stored-vs-casacore frame rotation median %.3f arcsec (max %.3f); '
            'residual after rotation: median %.3f m, max %.3f m (inlier baselines)'
            % (args.uvw_method, np.median(rot_angles), np.max(rot_angles),
               np.median(rot_resid), np.max(rot_resid)))
    if exact_resid:
        log('UVW: stored vs. pure casacore UVW on existing baselines: median %.3f m, max %.3f m'
            % (np.median(exact_resid), np.max(exact_resid)))
    tout.flush()
    tout.close()
    tin.close()
    log('Done in %.1f s: %s' % (time.time() - t_start, msout))


def compute_new_uvw(old_uvw, lay, s0, ns, uvwcalc, tmpl, new, new_bl, cross_old,
                    method, rot_angles, rot_resid, exact_resid):
    """UVW (ns, nnew, 3) of the new baselines for timeslots s0..s0+ns-1."""
    out = np.zeros((ns, len(new_bl), 3))
    for it in range(ns):
        c = uvwcalc(lay.times[s0 + it])
        stored = old_uvw[it]
        calc = c[lay.a2] - c[lay.a1]
        err = np.linalg.norm(calc - stored, axis=1)[cross_old]
        exact_resid.append(np.median(err))
        if method == 'hybrid':
            # rotation that maps casacore UVWs onto the stored convention,
            # with iterative rejection of inconsistent baselines (e.g. super stations)
            sel = cross_old.copy()
            for _ in range(4):
                rot = procrustes(calc[sel], stored[sel])
                res = np.linalg.norm(calc @ rot.T - stored, axis=1)
                lim = max(5.0 * np.median(res[cross_old]), 0.05)
                new_sel = cross_old & (res <= lim)
                if new_sel.sum() < 3 or (new_sel == sel).all():
                    break
                sel = new_sel
            rot_angles.append(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))) * 3600)
            rot_resid.append(np.median(res[sel]))
            delta = (c[new] - c[tmpl]) @ rot.T
        for j, (p, q, k, swapped, kind) in enumerate(new_bl):
            if kind == 'auto':
                continue
            if method == 'exact':
                out[it, j] = c[q] - c[p]
            elif kind == 'template':
                out[it, j] = delta
            else:
                # stored uvw(p, tmpl) = U[tmpl] - U[p];  uvw(p, new) = that + U[new] - U[tmpl]
                base = -stored[k] if swapped else stored[k]
                out[it, j] = base + delta
    return out


def update_subtables(msin, msout, tmpl, new, args, new_pos, rot_enu, tlat, tlon):
    up_tmpl = enu_matrix(tlat, tlon)[2]
    tmpl_name = None

    def has(sub):
        return os.path.isfile(os.path.join(msout, sub, 'table.dat'))

    # ANTENNA
    tant = ct.table(os.path.join(msout, 'ANTENNA'), readonly=False, ack=False)
    tmpl_name = tant.getcell('NAME', tmpl)
    tmpl_pos = tant.getcell('POSITION', tmpl)
    tant.addrows(1)
    copy_row(tant, tmpl, new)
    tant.putcell('NAME', new, args.name)
    tant.putcell('POSITION', new, new_pos)
    if 'LOFAR_PHASE_REFERENCE' in tant.colnames():
        ref = tant.getcell('LOFAR_PHASE_REFERENCE', tmpl)
        tant.putcell('LOFAR_PHASE_REFERENCE', new, new_pos + rot_enu @ (ref - tmpl_pos))
    if 'LOFAR_STATION_ID' in tant.colnames() and has('LOFAR_STATION'):
        tst = ct.table(os.path.join(msout, 'LOFAR_STATION'), readonly=False, ack=False)
        st_name = re.sub(r'(HBA[01]?|LBA)$', '', args.name, flags=re.I) or args.name
        if st_name in list(tst.getcol('NAME')):
            sys.exit('ERROR: station %s already exists in LOFAR_STATION' % st_name)
        row = tst.nrows()
        tst.addrows(1)
        copy_row(tst, int(tant.getcell('LOFAR_STATION_ID', tmpl)), row)
        tst.putcell('NAME', row, st_name)
        if 'CLOCK_ID' in tst.colnames():
            tst.putcell('CLOCK_ID', row, int(tst.getcol('CLOCK_ID').max()) + 1)
        if 'FLAG_ROW' in tst.colnames():
            tst.putcell('FLAG_ROW', row, False)
        tst.close()
        tant.putcell('LOFAR_STATION_ID', new, row)
        log('LOFAR_STATION: added %s (row %d)' % (st_name, row))
    tant.close()

    # FEED / POINTING: copy rows of the template
    for sub in ('FEED', 'POINTING'):
        if not has(sub):
            continue
        tsub = ct.table(os.path.join(msout, sub), readonly=False, ack=False)
        if tsub.nrows() > 0:
            rows = np.flatnonzero(tsub.getcol('ANTENNA_ID') == tmpl)
            for r in rows:
                dst = tsub.nrows()
                tsub.addrows(1)
                copy_row(tsub, int(r), dst)
                tsub.putcell('ANTENNA_ID', dst, new)
            log('%s: copied %d row(s) of %s' % (sub, len(rows), tmpl_name))
        tsub.close()

    # LOFAR_ANTENNA_FIELD: copy and rotate to the new site
    field_map = {}
    if has('LOFAR_ANTENNA_FIELD'):
        taf = ct.table(os.path.join(msout, 'LOFAR_ANTENNA_FIELD'), readonly=False, ack=False)
        rows = np.flatnonzero(taf.getcol('ANTENNA_ID') == tmpl)
        for r in rows:
            r = int(r)
            dst = taf.nrows()
            taf.addrows(1)
            copy_row(taf, r, dst)
            field_map[r] = dst
            taf.putcell('ANTENNA_ID', dst, new)
            taf.putcell('POSITION', dst, new_pos + rot_enu @ (taf.getcell('POSITION', r) - tmpl_pos))
            if 'COORDINATE_AXES' in taf.colnames():
                taf.putcell('COORDINATE_AXES', dst,
                            rotate_vectors(taf.getcell('COORDINATE_AXES', r), rot_enu, up_tmpl))
            for col in ('ELEMENT_OFFSET', 'TILE_ELEMENT_OFFSET'):
                if col in taf.colnames() and taf.iscelldefined(col, r):
                    taf.putcell(col, dst, rotate_vectors(taf.getcell(col, r), rot_enu))
        taf.close()
        log('LOFAR_ANTENNA_FIELD: copied and rotated %d field(s)' % len(rows))

    if field_map and has('LOFAR_ELEMENT_FAILURE'):
        tef = ct.table(os.path.join(msout, 'LOFAR_ELEMENT_FAILURE'), readonly=False, ack=False)
        if tef.nrows() > 0:
            ids = tef.getcol('ANTENNA_FIELD_ID')
            n = 0
            for r in np.flatnonzero(np.isin(ids, list(field_map))):
                dst = tef.nrows()
                tef.addrows(1)
                copy_row(tef, int(r), dst)
                tef.putcell('ANTENNA_FIELD_ID', dst, field_map[int(ids[r])])
                n += 1
            log('LOFAR_ELEMENT_FAILURE: copied %d row(s)' % n)
        tef.close()

    if has('HISTORY'):
        th = ct.table(os.path.join(msout, 'HISTORY'), readonly=False, ack=False)
        row = th.nrows()
        th.addrows(1)
        cols = th.colnames()
        msg = ('add_station_to_ms: added antenna %s at lat %s lon %s h %.1f m (ITRF %.3f %.3f %.3f), '
               'copy of %s' % (args.name, args.position.split(',')[0], args.position.split(',')[-1],
                               args.height, *new_pos, tmpl_name))
        values = {'TIME': (time.time() / 86400.0 + 40587.0) * 86400.0, 'OBSERVATION_ID': 0,
                  'MESSAGE': msg, 'PRIORITY': 'NORMAL', 'ORIGIN': 'add_station_to_ms.py',
                  'OBJECT_ID': 0, 'APPLICATION': 'add_station_to_ms',
                  'CLI_COMMAND': [' '.join(sys.argv)],
                  'APP_PARAMS': ['%s=%s' % kv for kv in sorted(vars(args).items())]}
        for col, val in values.items():
            if col in cols:
                th.putcell(col, row, val)
        th.close()


if __name__ == '__main__':
    main()
