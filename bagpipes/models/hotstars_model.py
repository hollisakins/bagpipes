from __future__ import print_function, division, absolute_import

import os
import itertools
import numpy as np

"""hotstars: a hot ionizing source (blackbody) plus the nebular lines and continuum that Cloudy
predicts for it, from a precomputed grid stored per unit Q(H). Built 2026-10-06 for the Twin
Peaks nebular-continuum test (GRID_PROPOSAL.md in
~/Dropbox/research/projects/zenith/twinpeaks/results/nebular_test_2026-10-04/).

Grid file (npz, written by hotgrid/make_hot_grid.py --compile): axes T_bb [K], leak, Z_gas
[Zsun], logU, logn; arrays cont and inc (erg s-1 A-1 per photon s-1) on `wavelengths` [A],
lines (erg s-1 per photon s-1) with `line_names` (Cloudy labels), Te [K], ok mask.

Component parameters (model_components["hotstars"]):
  T_bb   : blackbody temperature [K]
  leak   : fraction of photons above 4 Ryd retained (He+-ionizing 'leakage'; 1 = pure blackbody)
  logQ   : log10 of the hydrogen-ionizing photon rate [s-1]. Sets source light, lines, continuum.
  Z_gas  : gas metallicity [Zsun]
  logU   : ionization parameter (bagpipes definition, at r = 1e19 cm)
  logn   : log10 n_H [cm-3]
  fwhm   : line FWHM [km/s] (default 300)
  fesc   : escape fraction; nebular part scaled by (1 - fesc) (default 0)
  source : include the source's own light (default True)
  lines  : include the lines (default True)
  grid   : path to the npz (default: BAGPIPES_HOTGRID env or the Twin Peaks results path)

Output: self.spectrum (source + nebular), self.source, self.nebular, all in L_sun / A on the
model wavelength grid (model_galaxy multiplies by 3.826e33 at the end, as for pyneb_nebular);
self.line_lums {name: erg/s}; self.Te.
"""

DEFAULT_GRID = os.environ.get(
    "BAGPIPES_HOTGRID",
    os.path.expanduser("~/Dropbox/research/projects/zenith/twinpeaks/results/"
                       "nebular_test_2026-10-04/hotgrid/hotgrid_bb.npz"))
LSUN = 3.826e33
C_KMS = 2.99792458e5


def _line_wav(name):
    tok = name.split()[-1]
    if tok.endswith("A"):
        return float(tok[:-1])
    if tok.endswith("m"):
        return float(tok[:-1]) * 1e4
    raise ValueError(name)


class hotstars(object):

    def __init__(self, wavelengths, param):
        self.wavelengths = np.asarray(wavelengths, float)
        self.param = param
        g = np.load(param.get("grid", DEFAULT_GRID), allow_pickle=False)
        axes = [np.log10(g["T_bb"]), np.asarray(g["leak"], float), np.log10(g["Z_gas"]),
                np.asarray(g["logU"], float), np.asarray(g["logn"], float)]
        # the grid file may list an axis in any order (e.g. leak 1.0, 0.25): sort ascending
        order = [np.argsort(a) for a in axes]
        self.axes = [a[o] for a, o in zip(axes, order)]
        def sort_arr(arr):
            for k, o in enumerate(order):
                arr = np.take(arr, o, axis=k)
            return arr
        g = {k: (sort_arr(g[k]) if k in ("cont", "inc", "lines", "Te", "ok") else g[k]) for k in g.files}
        self.ok = g["ok"]
        if not self.ok.all():
            # fill missing nodes with the nearest finished node along the last axis, then raise
            # if anything is still missing (a partial grid is only for development)
            import warnings
            warnings.warn("hotstars grid has %d missing models" % (~self.ok).sum())
        wg = g["wavelengths"]
        # resample the stored continua onto the model wavelengths once
        def resample(arr):
            flat = arr.reshape(-1, arr.shape[-1])
            out = np.zeros((flat.shape[0], self.wavelengths.shape[0]), np.float32)
            for i in range(flat.shape[0]):
                out[i] = np.interp(self.wavelengths, wg, flat[i], left=0., right=0.)
            return out.reshape(arr.shape[:-1] + (self.wavelengths.shape[0],))
        self.cont = resample(g["cont"])
        self.inc = resample(g["inc"])
        self.lines = np.asarray(g["lines"], np.float64)
        self.line_names = [str(n) for n in g["line_names"]]
        self.line_wavs = np.array([_line_wav(n) for n in self.line_names])
        self.Te_grid = g["Te"]
        self._dlam = np.gradient(self.wavelengths)
        self.spectrum = np.zeros_like(self.wavelengths)
        self.source = np.zeros_like(self.wavelengths)
        self.nebular = np.zeros_like(self.wavelengths)
        self.line_lums = {}
        self.Te = np.nan
        self.update(param)

    # ---- multilinear interpolation -----------------------------------------------------------
    def _corners(self, values):
        idx, frac = [], []
        for ax, v in zip(self.axes, values):
            v = float(np.clip(v, ax[0], ax[-1]))
            i = int(np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2))
            idx.append(i); frac.append((v - ax[i]) / (ax[i + 1] - ax[i]) if len(ax) > 1 else 0.)
        corners = []
        for bits in itertools.product((0, 1), repeat=len(self.axes)):
            w = 1.
            ii = []
            for b, i, f, ax in zip(bits, idx, frac, self.axes):
                if len(ax) == 1:
                    ii.append(0); continue
                w *= f if b else (1. - f); ii.append(i + b)
            if w > 0:
                corners.append((tuple(ii), w))
        return corners

    def _interp(self, arr, corners):
        out = 0.
        for ii, w in corners:
            out = out + w * np.asarray(arr[ii], dtype=np.float64)     # float32 storage; scale in float64
        return out

    # ---- evaluation ----------------------------------------------------------------------------
    def _add_gaussian(self, out, central_wav, fwhm_vel, luminosity):
        """Add a Gaussian line integrating to `luminosity` to `out` in place (local window only)."""
        sigma = central_wav * fwhm_vel / C_KMS / 2.3548
        lo, hi = np.searchsorted(self.wavelengths, [central_wav - 6. * sigma, central_wav + 6. * sigma])
        if hi - lo < 3:
            # unresolved on this grid: put the luminosity into the nearest pixel (per A)
            i = min(max(lo, 0), len(self.wavelengths) - 1)
            out[i] += luminosity / self._dlam[i]
            return
        w = self.wavelengths[lo:hi]
        g = np.exp(-0.5 * ((w - central_wav) / sigma) ** 2)
        out[lo:hi] += g * (luminosity / np.trapezoid(g, x=w))

    def update(self, param):
        self.param = param
        values = [np.log10(param["T_bb"]), param.get("leak", 1.0), np.log10(param["Z_gas"]),
                  param["logU"], param["logn"]]
        corners = self._corners(values)
        Q = 10. ** param["logQ"]
        fneb = 1. - param.get("fesc", 0.)
        cont = self._interp(self.cont, corners) * Q * fneb / LSUN
        src = self._interp(self.inc, corners) * Q / LSUN if param.get("source", True) else 0. * cont
        lums = self._interp(self.lines, corners) * Q * fneb                     # erg/s
        self.line_lums = dict(zip(self.line_names, lums))
        self.Te = float(self._interp(self.Te_grid, corners))
        neb = cont.copy()
        if param.get("lines", True):
            fwhm = param.get("fwhm", 300.)
            lo, hi = self.wavelengths[0], self.wavelengths[-1]
            for w0, L in zip(self.line_wavs, lums):
                # Lya and anything blueward are absorbed by the IGM/damping wing at the redshifts
                # this component is built for; bagpipes' own Cloudy nebular loses Lya to the IGM
                # model (its grid pixel sits at <= 1215.67 A), so skip them here for consistency
                if L > 0 and lo < w0 < hi and w0 > 1216.:
                    self._add_gaussian(neb, w0, fwhm, L / LSUN)
        self.nebular = neb
        self.source = src
        self.spectrum = src + neb
