from __future__ import print_function, division, absolute_import

import numpy as np
import warnings

from .. import config


class dust_emission(object):
    """Dust emission models.

    Supports multiple dust emission model types:

    - 'DL07': Draine & Li (2007) dust emission models
    - 'DC22': Modified blackbody with CMB correction (Casey+ style)

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values in Angstroms.

    redshift : float
        Redshift of the source (needed for CMB correction in DC22).

    model_comp : dict
        Model component dictionary. Must include 'type' key.

    BREAKING CHANGE: Constructor signature changed from (wavelengths)
    to (wavelengths, redshift, model_comp). The spectrum() method now
    takes model_comp dict instead of individual parameters.

    Migration:
        OLD: dust_emission(wavelengths).spectrum(qpah, umin, gamma)
        NEW: dust_emission(wavelengths, redshift, {'type': 'DL07'}).spectrum(
                 {'qpah': qpah, 'umin': umin, 'gamma': gamma})
    """

    def __init__(self, wavelengths, redshift, model_comp):
        self.wavelengths = wavelengths
        self.redshift = redshift
        self.model_comp = model_comp

        if self.model_comp['type'] == 'DL07':
            self.spectrum = self.spectrum_DL07
        elif self.model_comp['type'] == 'DC22':
            self.spectrum = self.spectrum_DC22
        else:
            raise NotImplementedError(
                f"Dust emission type '{model_comp['type']}' not implemented. "
                "Supported types: 'DL07', 'DC22'"
            )

    def spectrum_DL07(self, model_comp):
        """
        Draine & Li (2007) dust emission model.

        Parameters (in model_comp dict):
        --------------------------------
        qpah : float
            PAH mass fraction (0.1 to 4.58)
        umin : float
            Minimum radiation field intensity
        gamma : float
            Fraction of dust exposed to higher radiation fields
        """
        qpah = model_comp['qpah']
        umin = model_comp['umin']
        gamma = model_comp['gamma']

        qpah_ind = config.qpah_vals[config.qpah_vals < qpah].shape[0]
        umin_ind = config.umin_vals[config.umin_vals < umin].shape[0]

        qpah_fact = ((qpah - config.qpah_vals[qpah_ind-1])
                     / (config.qpah_vals[qpah_ind]
                        - config.qpah_vals[qpah_ind-1]))

        umin_fact = ((umin - config.umin_vals[umin_ind-1])
                     / (config.umin_vals[umin_ind]
                        - config.umin_vals[umin_ind-1]))

        umin_w = np.array([(1 - umin_fact), umin_fact])

        lqpah_only = config.dust_grid_umin_only[qpah_ind]
        hqpah_only = config.dust_grid_umin_only[qpah_ind+1]
        tqpah_only = (qpah_fact*hqpah_only[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_only[:, umin_ind:umin_ind+2])

        lqpah_umax = config.dust_grid_umin_umax[qpah_ind]
        hqpah_umax = config.dust_grid_umin_umax[qpah_ind+1]
        tqpah_umax = (qpah_fact*hqpah_umax[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_umax[:, umin_ind:umin_ind+2])

        interp_only = np.sum(umin_w*tqpah_only, axis=1)
        interp_umax = np.sum(umin_w*tqpah_umax, axis=1)

        model = gamma*interp_umax + (1 - gamma)*interp_only

        spectrum = np.interp(self.wavelengths,
                             config.dust_grid_umin_only[1][:, 0],
                             model, left=0., right=0.)

        # Normalize spectrum
        spectrum_norm = spectrum / np.trapezoid(spectrum, x=self.wavelengths)
        return spectrum_norm

    def spectrum_DC22(self, model_comp):
        """
        Modified blackbody dust emission with CMB correction.

        Implements a general opacity modified blackbody with a mid-IR
        power-law extension and CMB contrast correction for high-z sources.

        Parameters (in model_comp dict):
        --------------------------------
        Tdust : float
            Dust temperature in Kelvin
        beta : float
            Dust emissivity index
        alpha : float
            Mid-IR power-law slope
        lam0 : float or 'ot'
            Opacity wavelength in microns. Use 'ot' for optically thin.
        """
        T = model_comp['Tdust']
        beta = model_comp['beta']
        alpha = model_comp['alpha']
        lam0 = model_comp['lam0']  # in micron

        hck = 143877687.75039333  # Kelvin*angstrom
        c = 2.9979e18  # angstrom/s

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            z = self.redshift
            T_CMB_z0 = 2.73
            T_CMB = T_CMB_z0 * (1 + z)

            # Correct dust temperature for CMB heating
            Tdust_z = (T**(beta+4) + T_CMB_z0**(beta+4)
                       * ((1+z)**(beta+4) - 1))**(1/(beta+4))

            # CMB contrast factor
            CMB = 1 - ((np.power(c/self.wavelengths, 3)
                       / (np.exp(hck/(self.wavelengths*T_CMB)) - 1))
                       / (np.power(c/self.wavelengths, 3)
                       / (np.exp(hck/(self.wavelengths*Tdust_z)) - 1)))
            T = Tdust_z
            CMB = np.where(self.wavelengths < 1e4, 1, CMB)

            # Calculate MBB and find power-law intersection
            lam_fine = np.logspace(3, 6.5, 1000)
            if lam0 == 'ot':
                # Optically thin
                MBB = (np.power(c/lam_fine, beta)
                       * np.power(c/lam_fine, 3)
                       / (np.exp(hck/(lam_fine*T)) - 1))
            else:
                # General opacity
                MBB = ((1 - np.exp(-(lam0*1e4/lam_fine)**beta))
                       * np.power(c/lam_fine, 3)
                       / (np.exp(hck/(lam_fine*T)) - 1))

            delta_y = np.diff(np.log10(MBB))
            delta_x = np.diff(np.log10(lam_fine))
            deriv = delta_y / delta_x

            lam_fine = 0.5 * (lam_fine[1:] + lam_fine[:-1])
            lam_int = lam_fine[np.nanargmin(np.abs(deriv - alpha))]

            # Power-law normalization
            Npl = MBB[np.argmin(np.abs(lam_fine - lam_int))] * lam_int**(-alpha)
            PL = Npl * self.wavelengths**alpha

            # Recalculate MBB at output wavelengths
            if lam0 == 'ot':
                MBB = (np.power(c/self.wavelengths, beta)
                       * np.power(c/self.wavelengths, 3)
                       / (np.exp(hck/(self.wavelengths*T)) - 1))
            else:
                MBB = ((1 - np.exp(-(lam0*1e4/self.wavelengths)**beta))
                       * np.power(c/self.wavelengths, 3)
                       / (np.exp(hck/(self.wavelengths*T)) - 1))

            # Combine power-law and MBB
            spectrum = np.where(self.wavelengths < lam_int, PL, MBB)

            # Convert from F_nu to F_lambda
            spectrum = spectrum / self.wavelengths**2

            # Smooth transition at short wavelengths
            spectrum = np.power(
                10.,
                np.log10(spectrum)
                * 1 / (1 + np.exp(-20 * (np.log10(self.wavelengths) - 3.8)))
            )

            # Normalize
            spectrum = spectrum / np.trapezoid(spectrum, x=self.wavelengths)

        return spectrum * CMB
