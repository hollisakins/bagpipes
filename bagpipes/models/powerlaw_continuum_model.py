from __future__ import print_function, division, absolute_import

import numpy as np


class powerlaw_continuum(object):
    """Simple power-law continuum model: f_lambda proportional to lambda^beta.

    This model provides a featureless power-law continuum suitable for
    representing AGN continua, quasar spectra, or other non-thermal sources.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values in Angstroms.

    param : dict
        Parameters for the power-law model.

    Example
    -------
    model_components = {
        "redshift": 2.0,
        "powerlaw": {
            "logL5100": 44.0,  # log10(erg/s)
            "beta": -1.5       # power-law slope
        }
    }
    """

    def __init__(self, wavelengths, param):
        self.wavelengths = wavelengths
        self.param = param
        self.spectrum = np.zeros_like(wavelengths)

    def update(self, param):
        """Update the power-law model with new parameters.

        Parameters (in param dict)
        --------------------------
        logL5100 : float
            Log10 of luminosity at 5100A in erg/s. This is the
            monochromatic luminosity lambda * L_lambda at 5100A.

        beta : float
            Power-law slope. The spectrum follows f_lambda proportional
            to lambda^beta. Typical AGN values are around -1.5 to -2.
        """
        self.param = param

        logL5100 = param["logL5100"]
        beta = param["beta"]
        balmer_jump = param.get("balmer_jump", 1.0)

        # Build power-law spectrum
        spectrum = self.wavelengths ** beta

        # Normalize spectrum to unity at 5100A
        idx_5100 = np.argmin(np.abs(self.wavelengths - 5100.))
        spectrum /= spectrum[idx_5100]

        # Convert logL5100 (erg/s) to L_lambda at 5100A (Lsun/A)
        # L5100 is lambda * L_lambda at 5100A in erg/s
        lamLlam5100 = 10**logL5100 / 3.826e33  # lambda*L_lambda in Lsun
        Llam5100 = lamLlam5100 / 5100.          # L_lambda in Lsun/A

        spectrum *= Llam5100

        spectrum[self.wavelengths > 3646.] *= balmer_jump

        self.spectrum = spectrum
