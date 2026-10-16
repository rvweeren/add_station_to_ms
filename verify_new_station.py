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
Sanity checks for an MS written by add_station_to_ms.py.

  verify_new_station.py --msin IN.ms --msout OUT.ms --template DE604HBA --name DE614HBA
"""
import argparse
import os
import sys
import warnings

import numpy as np
import casacore.tables as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_station_to_ms import (Layout, UVWCalculator, combine_estimates, enu_matrix,  # noqa: E402
                               estimate_sigma0, itrf_to_geodetic)


def noise_level(d, f, w):
    """Weight-normalised noise level of one baseline (same estimator as add_station_to_ms)."""
    return float(combine_estimates(estimate_sigma0(d, f, w))[0])

ok = True
warnings.simplefilter('ignore', RuntimeWarning)


def check(cond, msg):
    global ok
    print('%s  %s' % ('PASS' if cond else 'FAIL', msg))
    ok &= bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msin', required=True)
    ap.add_argument('--msout', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()

    tin = ct.table(args.msin, ack=False)
    tout = ct.table(args.msout, ack=False)
    ain = ct.table(os.path.join(args.msin, 'ANTENNA'), ack=False)
    aout = ct.table(os.path.join(args.msout, 'ANTENNA'), ack=False)
    names_out = list(aout.getcol('NAME'))
    tmpl = names_out.index(args.template)
    new = names_out.index(args.name)
    pos = aout.getcol('POSITION')
    check(aout.nrows() == ain.nrows() + 1, 'ANTENNA has one extra row (%d)' % aout.nrows())
    lat, lon, h = itrf_to_geodetic(pos[new])
    print('      new antenna %s: lat %.5f lon %.5f h %.2f m' % (args.name, lat, lon, h))
    for col in aout.colnames():
        if col in ('NAME', 'POSITION', 'LOFAR_PHASE_REFERENCE', 'LOFAR_STATION_ID'):
            continue
        a, b = aout.getcell(col, tmpl), aout.getcell(col, new)
        check(np.array_equal(np.asarray(a), np.asarray(b)), 'ANTENNA.%s copied from template' % col)

    lin, lout = Layout(tin), Layout(tout)
    check(lout.ntime == lin.ntime, 'same number of timeslots (%d)' % lout.ntime)
    print('      baselines per slot: %d -> %d' % (lin.nbl, lout.nbl))
    check(np.array_equal(lout.a1[:lin.nbl], lin.a1) and np.array_equal(lout.a2[:lin.nbl], lin.a2),
          'existing baselines keep their order')
    check(np.all(lout.a2[lin.nbl:] == new), 'all new baselines have ANTENNA2 = new antenna')
    check(np.array_equal(lout.times, lin.times), 'TIME identical')

    # existing rows unchanged
    old_rows = (np.arange(lout.ntime)[:, None] * lout.nbl + np.arange(lin.nbl)[None]).ravel()
    dmtypes = {c: dm['TYPE'] for dm in tout.getdminfo().values() for c in dm['COLUMNS']}
    complex_cols = [c for c in tout.colnames() if tout.iscelldefined(c, 0)
                    and tout.getcoldesc(c).get('valueType') in ('complex', 'dcomplex')
                    and np.ndim(tout.getcell(c, 0)) == 2]
    noise_cols = [c for c in complex_cols if not c.upper().startswith('MODEL')]
    for col in tin.colnames():
        if col not in tout.colnames():
            print('      %s: not in output' % col)
            continue
        if not tin.iscelldefined(col, 0) or col in noise_cols:
            continue
        a = tin.getcol(col)
        b = tout.getcol(col)[old_rows]
        equal_nan = np.asarray(a).dtype.kind in 'fc'
        if np.array_equal(a, b, equal_nan=equal_nan):
            check(True, '%s: existing rows bit-identical' % col)
        else:
            with np.errstate(all='ignore'):
                rel = np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a))
                fin = np.isfinite(a) & np.isfinite(b)
                relmed = np.median(np.abs(a - b)[fin]) / np.median(np.abs(a)[fin])
            check(dmtypes.get(col) == 'DyscoStMan' and rel < 0.2,
                  '%s: existing rows differ (max rel %.2g, median rel %.2g; %s re-compression)'
                  % (col, rel, relmed, dmtypes.get(col)))

    # UVW of new baselines
    tf = ct.table(os.path.join(args.msout, 'FIELD'), ack=False)
    calc = UVWCalculator(pos, tf.getcell('PHASE_DIR', 0)[0])
    uvw = tout.getcol('UVW').reshape(lout.ntime, lout.nbl, 3)
    errs, rel, excess = [], [], []
    for it in np.linspace(0, lout.ntime - 2, 6).astype(int):
        c = calc(lout.times[it])
        for k in range(lin.nbl, lout.nbl):
            p, q = lout.a1[k], lout.a2[k]
            if p == q:
                continue
            dev_new = uvw[it, k] - (c[q] - c[p])
            errs.append(np.linalg.norm(dev_new))
            excess.append(np.linalg.norm(uvw[it, k]) - np.linalg.norm(pos[q] - pos[p]))
            if p != tmpl:
                # deviation from casacore of the partner baseline (p, template) as stored
                kp, sw = lin.find(p, tmpl)
                up = -uvw[it, kp] if sw else uvw[it, kp]
                rel.append(np.linalg.norm(dev_new - (up - (c[tmpl] - c[p]))))
    print('      new-baseline UVW vs pure casacore: median %.3f m, max %.3f m' % (np.median(errs), np.max(errs)))
    print('      ... relative to the partner baseline deviation: median %.3f m, max %.3f m'
          % (np.median(rel), np.max(rel)))
    check(np.max(rel) < 2.0, 'new-baseline UVWs consistent with partner baselines within 2 m')
    check(np.max(excess) < 1.0, '|uvw| <= |ITRF baseline| for new baselines')

    # LOFAR_ANTENNA_FIELD orientation
    taf = ct.table(os.path.join(args.msout, 'LOFAR_ANTENNA_FIELD'), ack=False)
    ids = taf.getcol('ANTENNA_ID')
    rt, rn = np.flatnonzero(ids == tmpl), np.flatnonzero(ids == new)
    check(len(rt) == len(rn) and len(rn) > 0, 'LOFAR_ANTENNA_FIELD rows copied (%d)' % len(rn))
    tlat, tlon, _ = itrf_to_geodetic(pos[tmpl])
    for r1, r2 in zip(rt, rn):
        ax1, ax2 = taf.getcell('COORDINATE_AXES', r1), taf.getcell('COORDINATE_AXES', r2)
        loc1 = enu_matrix(tlat, tlon) @ ax1.T
        loc2 = enu_matrix(lat, lon) @ ax2.T
        check(np.allclose(loc1, loc2, atol=1e-9), 'field axes identical in local ENU frame')
        e1 = taf.getcell('ELEMENT_OFFSET', r1) @ enu_matrix(tlat, tlon).T
        e2 = taf.getcell('ELEMENT_OFFSET', r2) @ enu_matrix(lat, lon).T
        check(np.allclose(e1, e2, atol=1e-6), 'element offsets identical in local ENU frame')
        print('      field position offset from antenna: %.3f m' % np.linalg.norm(taf.getcell('POSITION', r2) - pos[new]))
    st = ct.table(os.path.join(args.msout, 'LOFAR_STATION'), ack=False)
    sid = aout.getcell('LOFAR_STATION_ID', new)
    print('      LOFAR_STATION[%d] = %s, CLOCK_ID %d' % (sid, st.getcell('NAME', sid), st.getcell('CLOCK_ID', sid)))
    for sub in ('FEED', 'POINTING'):
        ts = ct.table(os.path.join(args.msout, sub), ack=False)
        check((ts.getcol('ANTENNA_ID') == new).sum() == (ts.getcol('ANTENNA_ID') == tmpl).sum(),
              '%s rows for new antenna' % sub)

    # noise, flags and weights of new baselines vs partner baselines
    wcol = 'WEIGHT_SPECTRUM' if 'WEIGHT_SPECTRUM' in tout.colnames() else None
    for col in complex_cols:
        ratios, flagdiff, wdiff = [], [], []
        for k in range(lin.nbl, lout.nbl):
            p = int(lout.a1[k])
            if p in (tmpl, new):
                continue
            kp, _ = lin.find(p, tmpl)
            dn = tout.getcol(col, k, lout.ntime, lout.nbl)
            fn = tout.getcol('FLAG', k, lout.ntime, lout.nbl)
            fp = tin.getcol('FLAG', kp, lin.ntime, lin.nbl)
            flagdiff.append(np.mean(fn != fp))
            wn = tout.getcol(wcol, k, lout.ntime, lout.nbl) if wcol else np.ones(dn.shape)
            if wcol:
                wp = tin.getcol(wcol, kp, lin.ntime, lin.nbl)
                wdiff.append(np.nanmax(np.abs(wn - wp)) / max(np.nanmax(np.abs(wp)), 1e-30))
            if col.upper().startswith('MODEL'):
                ratios.append(np.nan if np.any(dn != 0) else 1.0)
                continue
            dp = tin.getcol(col, kp, lin.ntime, lin.nbl)
            wp = tin.getcol(wcol, kp, lin.ntime, lin.nbl) if wcol else np.ones(dp.shape)
            sn = noise_level(dn, fn, wn)
            sp = noise_level(dp, fp, wp)
            if np.isfinite(sn) and np.isfinite(sp):
                ratios.append(sn / sp)
        if col.upper().startswith('MODEL'):
            check(np.all(np.isfinite(ratios)), '%s zero on all new baselines' % col)
        elif ratios:
            print('      %s: sigma0(new)/sigma0(partner) median %.3f, range %.3f-%.3f'
                  % (col, np.median(ratios), np.min(ratios), np.max(ratios)))
            check(abs(np.median(ratios) - 1) < 0.1, '%s: noise level matches partner baselines' % col)
        check(np.max(flagdiff) == 0, '%s: FLAG of new baselines equals partner (i, template)' % col)
        if wdiff:
            check(np.max(wdiff) < 0.01, 'WEIGHT_SPECTRUM equals partner within Dysco precision (max rel %.2g)'
                  % np.max(wdiff))

    check_noise_only(tin, tout, lin, lout, noise_cols, wcol, names_out)

    print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


def baseline_sigma0(tab, lay, col, wcol):
    """Per-baseline median weight-normalised sigma0 (time differences)."""
    out = np.full(lay.nbl, np.nan)
    d = tab.getcol(col).reshape(lay.ntime, lay.nbl, -1, tab.getcell(col, 0).shape[-1])
    f = tab.getcol('FLAG').reshape(d.shape)
    w = tab.getcol(wcol).reshape(d.shape) if wcol else np.ones(d.shape)
    for k in range(lay.nbl):
        if lay.a1[k] != lay.a2[k]:
            out[k] = noise_level(d[:, k], f[:, k], w[:, k])
    return out


def check_noise_only(tin, tout, lin, lout, noise_cols, wcol, names):
    """The output data must be pure noise with, per baseline, the noise
    level of the input data."""
    for col in noise_cols:
        s_in = baseline_sigma0(tin, lin, col, wcol)
        s_out = baseline_sigma0(tout, lout, col, wcol)
        ratio = s_out[:lin.nbl] / s_in
        good = np.isfinite(ratio)
        print('      %s: sigma0(out)/sigma0(in) per existing baseline: median %.3f, 5-95%% %.3f-%.3f'
              % (col, np.median(ratio[good]), np.percentile(ratio[good], 5), np.percentile(ratio[good], 95)))
        check(abs(np.median(ratio[good]) - 1) < 0.05, '%s: output noise matches the input noise level' % col)
        # noise level per class of baseline (e.g. superstation, international, remote)
        cls = np.array([''.join(sorted([names[p][:2], names[q][:2]])) for p, q in zip(lout.a1, lout.a2)])
        for c in np.unique(cls[:lin.nbl]):
            m = (cls[:lin.nbl] == c) & good
            if m.any():
                print('        %-6s in %.4g  out %.4g  (%d baselines)'
                      % (c, np.median(s_in[m]), np.median(s_out[:lin.nbl][m]), m.sum()))
        # pure noise: mean of the data should be ~0 (no sky signal left)
        d = tout.getcol(col, 0, 50 * lout.nbl)
        f = tout.getcol('FLAG', 0, 50 * lout.nbl)
        d = np.where(f, np.nan, d)
        ratio_mean = np.abs(np.nanmean(d)) / np.nanstd(d)
        check(ratio_mean < 0.01, '%s: zero-mean noise (|mean|/std = %.2g)' % (col, ratio_mean))


if __name__ == '__main__':
    main()
