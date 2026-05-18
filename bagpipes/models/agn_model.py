from __future__ import print_function, division, absolute_import

import numpy as np


def Bnu(nu, T):
    """Planck function for blackbody.

    Parameters
    ----------
    nu : float or array
        Frequency in GHz
    T : float
        Temperature in K

    Returns
    -------
    Bnu : float or array
        Spectral radiance (proportional to Jy)
    """
    return nu**3 * 1474.49946476 * 1 / (np.exp(0.04799243*nu/T) - 1)


class agn(object):
    """AGN continuum and emission line models.

    Supports multiple AGN model types:

    - 'MBB': Modified blackbody continuum with Ha emission line.
             Good for dust-heated AGN emission.

    - 'carnall': Broken power-law continuum with configurable
                 broad and narrow emission lines.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values in Angstroms.

    param : dict
        Parameters for the AGN model. Must include 'type' key.

    BREAKING CHANGES
    ----------------
    This model has been completely rewritten from the original bagpipes
    AGN model. The API and parameters are incompatible.

    Original parameters (NO LONGER SUPPORTED):
        - alphalam: power-law slope below 5000A
        - betalam: power-law slope above 5000A
        - f5100A: flux at 5100A (in spectrum units)
        - sigma: velocity dispersion for lines (km/s)
        - hanorm: Ha line normalization (Hb was hanorm/2.86)

    Migration to 'carnall' type (closest equivalent):
        OLD:
            agn = {
                'alphalam': -1.5,
                'betalam': -2.0,
                'f5100A': 1e-17,
                'sigma': 3000,
                'hanorm': 1e-16,
            }

        NEW:
            agn = {
                'type': 'carnall',
                'logL5100': 44.0,  # log10(erg/s), not flux!
                'plwvbrk': 5000,   # break wavelength (was fixed at 5000)
                'plbeta1': -1.5,   # = alphalam
                'plbeta2': -2.0,   # = betalam
                'broad_fwhm': 3000 * 2.355,  # FWHM, not sigma
                'narrow_fwhm': 500,
                'broad_lines': {'Ha': 6564.614, 'Hb': 4862.721},
                'narrow_lines': {},
                'logL_Ha': 42.0,   # log10(erg/s)
                'logL_Hb': 41.5,   # log10(erg/s)
            }

    Key differences:
        - Must specify 'type' ('MBB' or 'carnall')
        - Luminosities in log10(erg/s), not flux
        - Line widths as FWHM, not sigma
        - Flexible line definitions via dicts
        - Break wavelength is configurable
    """

    def __init__(self, wavelengths, param):
        self.wavelengths = wavelengths

        if param['type'] not in ['MBB', 'carnall']:
            raise NotImplementedError(
                f"AGN type '{param['type']}' not implemented. "
                "Supported types: 'MBB', 'carnall'"
            )

        self.param = param

    def update(self, param):
        """Update the AGN model with new parameters."""
        self.param = param

        if param['type'] == 'MBB':
            self._update_mbb(param)

        elif param['type'] == 'carnall':
            self._update_carnall(param)

        else:
            raise NotImplementedError

    def _update_mbb(self, param):
        """
        Modified blackbody AGN model with Ha emission.

        Parameters (in param dict):
        ---------------------------
        logL5100 : float
            Log10 of luminosity at 5100A in erg/s
        T : float
            Dust temperature in Kelvin
        beta : float
            Emissivity index for modified blackbody
        FWHM : float
            FWHM of Ha line in km/s
        logLHaL5100 : float
            Log10 ratio of Ha luminosity to 5100A luminosity
        """
        logL5100 = param['logL5100']
        FWHMHa = param['FWHM']
        logLHaL5100 = param['logLHaL5100']
        T = param['T']
        beta = param['beta']
        balmer_break = param.get('balmer_break', 1.)

        L5100 = np.power(10., logL5100)
        nu = 2.998e9 / self.wavelengths  # in GHz
        nu0 = 2.998e9 / 5500
        fnu = Bnu(nu, T) * (nu / nu0)**(beta)
        agn_spec = fnu / self.wavelengths**2
        agn_spec /= agn_spec[np.argmin(np.abs(self.wavelengths - 5100.))]

        # Normalize to L5100
        lamLlam5100 = L5100 / 3.826e33  # in Lsun
        Llam5100 = lamLlam5100 / 5100   # in Lsun/angstrom
        agn_spec *= Llam5100

        # Add Ha emission line
        logLHa = logL5100 + logLHaL5100
        LHa = np.power(10., logLHa) / 3.826e33  # in Lsun
        vel = 2.998e5 * (self.wavelengths - 6564.614) / 6564.614
        sig = FWHMHa / 2.355
        gauss = np.exp(-0.5 * vel**2 / sig**2)

        Ha = gauss / np.trapezoid(gauss, x=self.wavelengths) * LHa
        agn_spec += Ha

        # Add Balmer break at 3646A
        bb_mask = self.wavelengths <= 3645
        agn_spec[bb_mask] *= balmer_break  # simple factor for break

        self.spectrum = agn_spec

    def _update_carnall(self, param):
        """
        Broken power-law AGN continuum with emission lines.

        Parameters (in param dict):
        ---------------------------
        logL5100 : float
            Log10 of luminosity at 5100A in erg/s
        plwvbrk : float
            Wavelength of power-law break in Angstroms
        plbeta1 : float
            Power-law slope below the break
        plbeta2 : float
            Power-law slope above the break
        broad_fwhm : float
            FWHM of broad lines in km/s
        narrow_fwhm : float
            FWHM of narrow lines in km/s
        broad_lines : dict
            Dictionary mapping line names to rest wavelengths (Angstroms)
            for broad emission lines
        narrow_lines : dict
            Dictionary mapping line names to rest wavelengths (Angstroms)
            for narrow emission lines
        logL_{line} : float
            Log10 luminosity in erg/s for each line defined in
            broad_lines and narrow_lines

        Example:
        --------
        agn = {
            'type': 'carnall',
            'logL5100': (42, 46),
            'plwvbrk': 5000,
            'plbeta1': (-2, 0),
            'plbeta2': (-3, -1),
            'broad_fwhm': 3000,
            'narrow_fwhm': 500,
            'broad_lines': {'Ha': 6564.614, 'Hb': 4862.721},
            'narrow_lines': {'OIII': 5008.24},
            'logL_Ha': (40, 44),
            'logL_Hb': (40, 44),
            'logL_OIII': (40, 44),
        }
        """
        lam_break = param['plwvbrk']
        mask1 = (self.wavelengths < lam_break)
        mask2 = np.invert(mask1)

        agn_spec = np.zeros_like(self.wavelengths)

        agn_spec[mask1] = self.wavelengths[mask1] ** param["plbeta1"]
        agn_spec[mask2] = self.wavelengths[mask2] ** param["plbeta2"]

        # Normalize at break point
        agn_spec[mask1] /= agn_spec[mask1][-1]
        agn_spec[mask2] /= agn_spec[mask2][0]

        # Normalize to L5100
        agn_spec /= agn_spec[np.argmin(np.abs(self.wavelengths - 5100.))]
        lamLlam5100 = np.power(10., param['logL5100']) / 3.826e33  # in Lsun
        Llam5100 = lamLlam5100 / 5100  # in Lsun/angstrom
        agn_spec *= Llam5100

        # Add broad emission lines
        sigma = param['broad_fwhm'] / 2.355
        for line in param['broad_lines'].keys():
            line_lum = np.power(10., param[f"logL_{line}"]) / 3.826e33
            agn_spec += self.gaussian_model(
                param['broad_lines'][line], sigma, line_lum
            )

        # Add narrow emission lines
        sigma = param['narrow_fwhm'] / 2.355
        for line in param['narrow_lines'].keys():
            line_lum = np.power(10., param[f"logL_{line}"]) / 3.826e33
            agn_spec += self.gaussian_model(
                param['narrow_lines'][line], sigma, line_lum
            )

        self.spectrum = agn_spec

    def gaussian_model(self, central_wav, sigma_vel, norm):
        """Generate a Gaussian emission line profile.

        Parameters
        ----------
        central_wav : float
            Central wavelength in Angstroms
        sigma_vel : float
            Velocity dispersion in km/s
        norm : float
            Integrated line flux normalization

        Returns
        -------
        line : np.ndarray
            Line profile flux density
        """
        x = self.wavelengths
        sigma_aa = sigma_vel * central_wav / (3e5)
        gauss = (norm / (sigma_aa * np.sqrt(2*np.pi)))
        gauss *= np.exp(-0.5 * (x - central_wav)**2 / sigma_aa**2)

        return gauss
