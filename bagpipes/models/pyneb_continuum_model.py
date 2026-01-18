from __future__ import print_function, division, absolute_import

import numpy as np

from pyneb.core.continuum import Continuum


class pyneb_continuum(object):
    """PyNeb-based nebular continuum model.

    This model computes nebular continuum emission (free-free, free-bound,
    2-photon) using PyNeb's Continuum class.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values in Angstroms.

    param : dict
        Parameters for the nebular continuum model.

    Model Parameters
    ----------------
    Te : float
        Electron temperature in Kelvin.

    ne : float
        Electron density in cm^-3.

    lognorm : float
        Log10 of luminosity at 5100A in erg/s (lambda * L_lambda).
        Used to normalize the continuum.

    cont_HI : bool, optional
        Include HI recombination continuum. Default: True.

    cont_HeI : bool, optional
        Include HeI recombination continuum. Default: False.

    cont_HeII : bool, optional
        Include HeII recombination continuum. Default: False.

    cont_2p : bool, optional
        Include 2-photon continuum. Default: True.

    cont_ff : bool, optional
        Include free-free continuum. Default: True.

    Example
    -------
    model_components = {
        "redshift": 2.0,
        "pyneb_continuum": {
            "Te": 10000,       # K
            "ne": 100,         # cm^-3
            "lognorm": 42.0,   # log10(erg/s) at 5100A
            "cont_HI": True,   # Include HI (default)
            "cont_HeI": False, # Exclude HeI (default)
            "cont_HeII": False,# Exclude HeII (default)
            "cont_2p": True,   # Include 2-photon (default)
            "cont_ff": True    # Include free-free (default)
        }
    }
    """

    def __init__(self, wavelengths, param):
        self.wavelengths = wavelengths
        self.param = param
        self.spectrum = np.zeros_like(wavelengths)
        self.cont = Continuum()

    def update(self, param):
        """Update the nebular continuum model with new parameters.

        Parameters (in param dict)
        --------------------------
        Te : float
            Electron temperature in Kelvin.

        ne : float
            Electron density in cm^-3.

        lognorm : float
            Log10 of luminosity at 5100A in erg/s.

        cont_HI : bool, optional
            Include HI recombination continuum. Default: True.

        cont_HeI : bool, optional
            Include HeI recombination continuum. Default: False.

        cont_HeII : bool, optional
            Include HeII recombination continuum. Default: False.

        cont_2p : bool, optional
            Include 2-photon continuum. Default: True.

        cont_ff : bool, optional
            Include free-free continuum. Default: True.
        """
        self.param = param

        Te = param["Te"]
        ne = param["ne"]
        lognorm = param["lognorm"]

        # Get optional continuum component flags with defaults
        cont_HI = param.get("cont_HI", True)
        cont_HeI = param.get("cont_HeI", False)
        cont_HeII = param.get("cont_HeII", False)
        cont_2p = param.get("cont_2p", True)
        cont_ff = param.get("cont_ff", True)

        # Calculate continuum using PyNeb
        # PyNeb breaks for wavelengths outside ~1000-100000 A range, so we only
        # compute within that range and set the rest to 0
        raw_spectrum = np.zeros_like(self.wavelengths)
        mask = (self.wavelengths > 1000.) & (self.wavelengths < 1e5)

        if np.any(mask):
            raw_spectrum[mask] = self.cont.get_continuum(
                Te, ne,
                wl=self.wavelengths[mask],
                cont_HI=cont_HI,
                cont_HeI=cont_HeI,
                cont_HeII=cont_HeII,
                cont_2p=cont_2p,
                cont_ff=cont_ff
            )

        # Normalize to unity at 5100A
        idx_5100 = np.argmin(np.abs(self.wavelengths - 5100.))
        if raw_spectrum[idx_5100] > 0:
            raw_spectrum = raw_spectrum / raw_spectrum[idx_5100]

        # Apply normalization (same pattern as power-law model)
        # lognorm is log10(lambda * L_lambda at 5100A in erg/s)
        lamLlam5100 = 10**lognorm / 3.826e33  # lambda*L_lambda in Lsun
        Llam5100 = lamLlam5100 / 5100.         # L_lambda in Lsun/A

        self.spectrum = raw_spectrum * Llam5100
