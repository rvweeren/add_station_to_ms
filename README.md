# add_station_to_ms

`add_station_to_ms.py` adds a new antenna (station) to a Measurement Set (MS)
and writes the result to a new MS. The input MS is opened read-only and never
modified.

The new station is a copy of an existing *template* station, with all its
properties, placed at a position you choose. The script adds the baselines
between the new station and all existing stations and computes their UVW
coordinates. The data of **all** baselines, existing and new, are replaced by
Gaussian noise that matches the noise in the input MS. The output is a
noise-only simulation of the extended array.

Typical use: simulate a new LOFAR station, e.g. one in Bremen, and look at the
effect on uv-coverage, imaging or calibration.

## Requirements

- Python 3 with `numpy` and `python-casacore`.
- For Dysco-compressed MSs: casacore with the Dysco storage manager.
- Everything is available in the LOFAR/flocs containers.

## Example

Add a station `DE614HBA` at 53.04° N, 8.60° E (Bremen), copied from
`DE604HBA`:

```
python add_station_to_ms.py --msin=input.ms --msout=output.ms \
    --template=DE604HBA --name=DE614HBA --position=N53.04,E8.60
```

## Options

| Option | Default | Description |
|---|---|---|
| `--msin` | (required) | Input MS. It is only read. |
| `--msout` | (required) | Output MS. It must not exist, unless `--overwrite` is given. |
| `--template` | (required) | Name (or index) of the existing antenna whose properties are copied, e.g. `DE604HBA`. |
| `--name` | (required) | Name of the new antenna, e.g. `DE614HBA`. The station name in `LOFAR_STATION` is this name without the `HBA`/`LBA` suffix (`DE614`). |
| `--position` | (required) | Geodetic WGS84 position, e.g. `N53.04,E8.60`. `53.04N,8.60E` and plain signed degrees also work; use `S`/`W` for south/west. |
| `--height` | `0.0` | Height above the WGS84 ellipsoid in metres. |
| `--seed` | random | Random seed for the noise, for reproducible output. |
| `--noise-scale` | `1.0` | Multiply the estimated noise by this factor. |
| `--datacolumn` | `DATA` | Input visibility column whose noise properties are measured, e.g. `CORRECTED_DATA`. The output column is always `DATA`. |
| `--noise-model` | `station` | How the noise level per baseline is set, see [Noise](#4-noise). `station`: per-station sensitivity model. `baseline`: the level measured on each baseline. |
| `--uvw-method` | `hybrid` | How the UVWs of the new baselines are computed, see [UVW](#3-uvw). |
| `--storage` | `same` | `same`: use the input's storage managers (e.g. Dysco). `plain`: write uncompressed. |
| `--chunk-mb` | `512` | Approximate memory per data chunk (MB) when writing. |
| `--ncpu` | `0` | Number of threads for the noise estimation. `0` means the number of available cores, at most 16. |
| `--overwrite` | off | Remove `--msout` first if it exists. |

Run `python add_station_to_ms.py --help` for the same list.

## What the script does

### 1. New antenna

- The geodetic position is converted to ITRF coordinates.
- These subtables get a copy of the template's rows, with only the position-dependent values changed:
  - `ANTENNA`: new `NAME`, `POSITION` and `LOFAR_PHASE_REFERENCE`.
  - `LOFAR_STATION`: a new station with its own clock.
  - `FEED`, `POINTING` and `LOFAR_ELEMENT_FAILURE`.
  - `LOFAR_ANTENNA_FIELD`: the field position, coordinate axes and element and tile offsets are rotated from the template's local frame to the new location. The station keeps the same layout and orientation relative to local north and up, so beam models (e.g. EveryBeam in DP3) give a proper beam for the new station.
- A row describing the run is added to `HISTORY`.

### 2. New baselines

- The input must be a regular MS: the same baselines in every timeslot.
- For each existing station *i*, the baseline (*i*, NEW) is added at the end of every timeslot, plus (template, NEW).
- For each new baseline (*i*, NEW), the partner baseline is (*i*, template). From it the script copies:
  - `FLAG`, `WEIGHT_SPECTRUM`, `WEIGHT`, `SIGMA`;
  - all scalar columns (`TIME`, `INTERVAL`, `EXPOSURE`, ...).
- The new station is assumed to have the same hardware as the template, so it gets the same weights and flags.
- The baseline (template, NEW) has no partner baseline. It gets the median weights over all template baselines, and the flags that are set on all of them.

### 3. UVW

- `hybrid` (default): uvw(*i*, NEW) = stored uvw(*i*, template) + the computed (NEW − template) vector.
  - That vector is computed with casacore measures and rotated, per timeslot, to match the convention of the UVWs stored in the MS.
  - This stays consistent with the existing UVWs, even when those come from slightly different software or settings.
- `exact`: all new UVWs are computed directly with casacore measures.

UVW and `ANTENNA1`/`ANTENNA2` are always written uncompressed. This is the same as DP3's `msout.uvwcompression=false`, even if the input was written with DP3 UVW or antenna compression.

### 4. Noise

The `DATA` of every cross baseline, existing and new, is replaced by noise.
The noise level is measured from the input data on every baseline.

- **Weight normalisation.** The data are multiplied by √weight, so pure noise has the same level σ0 everywhere. The noise of one visibility is then σ0 / √weight.
- **Sky signal removal.** The sky signal is suppressed by taking differences between neighbouring samples:
  - in time;
  - in frequency;
  - in time and frequency together;
  - for 4-polarisation data, also the difference of the two cross-hands (XY − YX, or RL − LR), where the sky is nearly zero.
- **Robust rms.** Each estimate is an iteratively clipped rms (3.5σ, corrected for the clipping), so RFI and outliers have little effect.
- **Choosing the estimate.** The lowest estimate is used, because remaining sky signal can only raise it. The estimates from different time chunks are combined with a median.
- **Station sensitivity.** Stations differ in sensitivity: superstations, core, remote and international stations differ in collecting area. A per-station model is fitted to the measured noise of all baselines:

  σ0(p, q) = s_p · s_q · shape(channel, polarisation)

  - The fitted factors s are printed as a table in the log.
  - The new station gets the factor of its template.
  - Baselines that don't fit the model are rejected from the fit and listed.
  - Fully flagged stations get the median factor.
  - If the weights already include the station sensitivity, which is common after calibration, all factors are close to 1.
- **Noise model.** With `--noise-model baseline`, each baseline uses its own measured level instead of the model, and each new baseline uses its partner's level.
- **Generated noise.** The noise is complex Gaussian with σ = σ0 / √weight per visibility, multiplied by `--noise-scale`. Flagged samples also get noise. Samples with weight 0 get zero.

Every existing baseline keeps its original noise level, and the new baselines get the noise statistics of their partner baselines. The output contains no sky signal.

### 5. Output MS

- The output has one visibility column, `DATA`, containing the noise. Other visibility columns (`MODEL_DATA`, `CORRECTED_DATA`, ...) are not written.
- All other columns and subtables are copied.
- By default the input's storage managers are kept, including Dysco compression. Columns whose manager can't be used for writing fall back to standard uncompressed managers. `--storage plain` writes everything uncompressed.

## Verification

`verify_new_station.py` checks an output MS against its input:

```
python verify_new_station.py --msin=input.ms --msout=output.ms \
    --template=DE604HBA --name=DE614HBA
```

It checks:
- the subtables (antenna copy, rotated antenna field);
- that the existing rows are unchanged, apart from `DATA`;
- the UVW consistency of the new baselines;
- the flags, weights and noise level of the new baselines against their partners;
- the noise per baseline against the input, grouped by station type.

## Notes and limitations

- Only regular MSs are supported: one spectral window, and the same baseline list in every timeslot.
- The input MS is never changed. With Dysco, copied columns such as `WEIGHT_SPECTRUM` are recompressed, so they are not bit-identical to the input.
- `QUALITY_*` statistics tables are copied unchanged and don't include the new station.
- On a typical LOFAR MS (28 stations, 480 timeslots, 115 channels) a run takes about 15 s.

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
