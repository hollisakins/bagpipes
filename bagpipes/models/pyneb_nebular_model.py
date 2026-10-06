from __future__ import print_function, division, absolute_import

import os
import hashlib
import numpy as np

import pyneb as pn
from pyneb.core.continuum import Continuum


# upper level -> (name, rest wavelength in air, Angstroms)
BALMER = {3: ("Halpha", 6562.80), 4: ("Hbeta", 4861.33), 5: ("Hgamma", 4340.47),
          6: ("Hdelta", 4101.74), 7: ("Hepsilon", 3970.08), 8: ("H8", 3889.06),
          9: ("H9", 3835.40), 10: ("H10", 3797.91), 11: ("H11", 3770.63),
          12: ("H12", 3750.15)}
PASCHEN = {4: ("Paalpha", 18751.0), 5: ("Pabeta", 12818.1),
           6: ("Pagamma", 10938.1), 7: ("Padelta", 10049.4)}
HB = (4, 2)
LSUN = 3.826e33
C_KMS = 2.998e5
TE_MAX = 30000.  # PyNeb H I recombination data (Storey & Hummer 1995) stop here


class pyneb_nebular(object):
    """Nebular continuum AND hydrogen recombination lines from one normalisation, L(Hbeta).

    The H I free-bound + free-free, He I free-bound and two-photon continua (PyNeb
    `Continuum`) and the Balmer/Paschen lines (PyNeb `RecAtom('H', 1)` case-B emissivities)
    are evaluated at the same (Te, ne) and scaled by the same L(Hbeta). More continuum
    therefore forces brighter Balmer lines and a larger Balmer jump with no free knob in
    between. The component is decoupled from the stellar population: no assumption about
    Q(H), f_esc, the IMF or whether the ionizing source is stellar at all.

    Shapes are tabulated once on a (log Te, log ne) grid, cached on disk, and interpolated
    bilinearly in log so an update costs ~0.1 ms. The grid stops at 30 kK because PyNeb's
    H I line emissivities do.

    Unlike `pyneb_continuum`, this component is attenuated by the diffuse-ISM dust screen
    in `model_galaxy` (set "attenuate": False to add it after dust, e.g. for an unobscured
    NLR).

    Model parameters
    ----------------
    Te : float          Electron temperature in K (5000-30000).
    ne : float          Electron density in cm^-3 (10-1e5).
    logLHb : float      log10 L(Hbeta) in erg/s. Sets both continuum and lines.
    fwhm : float        Line FWHM in km/s. Default 300 (unresolved on the model grid at
                        config.R_spec = 1000; the R_curve convolution dominates anyway).
    velshift : float    Line velocity offset in km/s. Default 0.
    He1_H : float       He+/H+ abundance for the He I continuum. Default 0.1.
    He2_H : float       He++/H+ abundance for the He II continuum. Default 0.
    cont_2p, cont_ff, cont_HeI : bool   Continuum pieces. Default True.
    lines : bool        Include the H I lines. Default True.
    paschen : bool      Include Paschen lines as well as Balmer. Default True.
    attenuate : bool    Apply the stellar diffuse-ISM dust screen. Default True.
    logne_max : float   Upper edge of the tabulated density grid (log cm^-3). Default 5;
                        values of ne above the grid edge are clipped to it.

    Attributes after update()
    -------------------------
    spectrum : continuum + lines, L_sun / A on the model wavelength grid.
    continuum : continuum only, L_sun / A.
    line_lums : dict name -> (rest wavelength A, luminosity erg/s).
    """

    def __init__(self, wavelengths, param, logTe=(3.7, np.log10(TE_MAX), 24),
                 logne=None):
        self.wavelengths = np.asarray(wavelengths, float)
        if logne is None:   # grid ceiling in density; "logne_max" in the param dict raises it
            logne_max = float(param.get("logne_max", 5.0))
            logne = (1.0, logne_max, int(round(4 * (logne_max - 1.0))) + 1)
        self.param = param
        self.He1_H = float(param.get("He1_H", 0.10))
        self.He2_H = float(param.get("He2_H", 0.0))
        self.flags = dict(cont_HI=True, cont_HeI=bool(param.get("cont_HeI", True)),
                          cont_HeII=self.He2_H > 0, cont_2p=bool(param.get("cont_2p", True)),
                          cont_ff=bool(param.get("cont_ff", True)))
        self.logTe = np.linspace(*logTe)
        self.logne = np.linspace(*logne)
        self.mask = (self.wavelengths > 1000.) & (self.wavelengths < 1e5)  # PyNeb range
        self.spectrum = np.zeros_like(self.wavelengths)
        self.continuum = np.zeros_like(self.wavelengths)
        self.line_lums = {}
        self._build_or_load_grid()

    # ---- grid ---------------------------------------------------------------------------
    def _cache_path(self):
        h = hashlib.md5()
        h.update(self.wavelengths[self.mask].tobytes())
        h.update(repr((self.He1_H, self.He2_H, sorted(self.flags.items()),
                       tuple(self.logTe), tuple(self.logne))).encode())
        d = os.environ.get("BAGPIPES_CACHE", os.path.expanduser("~/.cache/bagpipes"))
        return os.path.join(d, "pyneb_nebular_" + h.hexdigest()[:16] + ".npz")

    def _build_or_load_grid(self):
        path = self._cache_path()
        keys = list(BALMER) + [100 + k for k in PASCHEN]
        if os.path.exists(path):
            f = np.load(path)
            self.cont = f["cont"]
            self.lines = {k: f["line_%d" % k] for k in keys}
            return
        C = Continuum()
        H = pn.RecAtom("H", 1)
        nT, nn = len(self.logTe), len(self.logne)
        self.cont = np.zeros((nT, nn, self.mask.sum()))      # L_lambda per unit L(Hb) [1/A]
        self.lines = {k: np.zeros((nT, nn)) for k in keys}    # L_line / L(Hb)
        wl = self.wavelengths[self.mask]
        for i, lt in enumerate(self.logTe):
            for j, ln in enumerate(self.logne):
                Te, ne = 10**lt, 10**ln
                jHb = H.getEmissivity(Te, ne, *HB)           # erg s-1 cm3
                # HI_label=None -> absolute erg s-1 cm3 A-1 (the default normalises to H11)
                self.cont[i, j] = C.get_continuum(Te, ne, wl=wl, He1_H=self.He1_H,
                                                  He2_H=self.He2_H, HI_label=None,
                                                  **self.flags) / jHb
                for u in BALMER:
                    self.lines[u][i, j] = H.getEmissivity(Te, ne, u, 2) / jHb
                for u in PASCHEN:
                    self.lines[100 + u][i, j] = H.getEmissivity(Te, ne, u, 3) / jHb
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, cont=self.cont, **{"line_%d" % k: v for k, v in self.lines.items()})

    def _weights(self, Te, ne):
        lt = np.clip(np.log10(Te), self.logTe[0], self.logTe[-1])
        ln = np.clip(np.log10(ne), self.logne[0], self.logne[-1])
        i = int(np.clip(np.searchsorted(self.logTe, lt) - 1, 0, len(self.logTe) - 2))
        j = int(np.clip(np.searchsorted(self.logne, ln) - 1, 0, len(self.logne) - 2))
        fi = (lt - self.logTe[i]) / (self.logTe[i + 1] - self.logTe[i])
        fj = (ln - self.logne[j]) / (self.logne[j + 1] - self.logne[j])
        return i, j, fi, fj

    def _interp(self, arr, Te, ne):
        """Bilinear interpolation in log of the tabulated quantity."""
        i, j, fi, fj = self._weights(Te, ne)
        la = np.log(np.maximum(arr[i:i + 2, j:j + 2], 1e-300))
        v = ((1 - fi) * (1 - fj) * la[0, 0] + fi * (1 - fj) * la[1, 0]
             + (1 - fi) * fj * la[0, 1] + fi * fj * la[1, 1])
        return np.exp(v)

    # ---- evaluation ---------------------------------------------------------------------
    def update(self, param):
        self.param = param
        Te, ne = float(param["Te"]), float(param["ne"])
        LHb = 10**float(param["logLHb"])                        # erg/s
        fwhm = float(param.get("fwhm", 300.))
        velshift = float(param.get("velshift", 0.))

        cont = np.zeros_like(self.wavelengths)
        cont[self.mask] = LHb * self._interp(self.cont, Te, ne) / LSUN   # Lsun/A
        self.continuum = cont
        spectrum = cont.copy()

        self.line_lums = {}
        if param.get("lines", True):
            series = dict(BALMER)
            if param.get("paschen", True):
                series.update({100 + u: v for u, v in PASCHEN.items()})
            for key, (name, wav) in series.items():
                L = LHb * float(self._interp(self.lines[key], Te, ne))
                self.line_lums[name] = (wav, L)
                spectrum += self._gaussian(wav * (1. + velshift / C_KMS), fwhm, L / LSUN)
        self.spectrum = spectrum

    def _gaussian(self, central_wav, fwhm_vel, luminosity):
        """Gaussian line in Lsun/A integrating to `luminosity` (Lsun); local window only."""
        sigma_aa = fwhm_vel / 2.355 * central_wav / C_KMS
        lo, hi = np.searchsorted(self.wavelengths, [central_wav - 6 * sigma_aa,
                                                    central_wav + 6 * sigma_aa])
        out = np.zeros_like(self.wavelengths)
        if hi - lo < 3:
            return out
        w = self.wavelengths[lo:hi]
        g = np.exp(-0.5 * (w - central_wav)**2 / sigma_aa**2)
        integral = np.trapezoid(g, x=w)
        if integral > 0:
            out[lo:hi] = g / integral * luminosity
        return out

    # ---- diagnostics --------------------------------------------------------------------
    def budget(self, Te, ne):
        """Pure-nebular diagnostics at (Te, ne): EWs (A), UV slope, f_lambda Balmer jump."""
        w = self.wavelengths
        c = np.zeros_like(w)
        c[self.mask] = self._interp(self.cont, Te, ne)
        L = {BALMER[u][0]: (BALMER[u][1], float(self._interp(self.lines[u], Te, ne)))
             for u in BALMER}
        ew = lambda n: L[n][1] / np.interp(L[n][0], w, c)
        m = (w > 1340) & (w < 2600)
        beta = np.polyfit(np.log10(w[m]), np.log10(c[m]), 1)[0]
        blue = c[(w > 3200) & (w < 3600)].mean()
        red = c[(w > 3650) & (w < 3850)].mean()
        return dict(EW_Hbeta=ew("Hbeta"), EW_Hgamma=ew("Hgamma"), EW_Hdelta=ew("Hdelta"),
                    beta_neb=beta, jump_flam=blue / red, Hg_Hb=L["Hgamma"][1])
