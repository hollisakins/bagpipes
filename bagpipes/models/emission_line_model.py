from __future__ import print_function, division, absolute_import

import numpy as np


class emission_lines(object):
    """Flexible emission line model with linking support.

    This model allows manual specification of emission lines with support
    for linking line parameters (FWHM, flux ratios) between lines.

    Parameters
    ----------
    wavelengths : np.ndarray
        1D array of wavelength values in Angstroms.

    param : dict
        Dictionary of line definitions, keyed by line name.

    Per-Line Parameters
    -------------------
    wav : float
        Rest-frame wavelength in Angstroms. Required.

    loglum : float
        Log10 line luminosity in erg/s. Required unless linked with ratio.

    fwhm : float
        Line width (FWHM) in km/s. Required unless linked.

    velshift : float, optional
        Velocity offset from redshift in km/s. Default: 0.

    profile : str, optional
        Line profile type: "gaussian", "lorentzian", or "voigt".
        Default: "gaussian".

    gamma : float, optional
        Lorentzian width (FWHM) in km/s for Voigt profiles.
        Default: 0 (pure Gaussian). Only used when profile="voigt".

    linked_to : str, optional
        Name of line to link to. Linked lines inherit FWHM.

    ratio : float, optional
        Flux ratio relative to linked line. Only used with linked_to.
        If specified, L_child = ratio * L_parent.
        If not specified, flux is free (set by loglum).

    Example
    -------
    model_components = {
        "redshift": 2.0,
        "emission_lines": {
            "OIII_5007": {
                "wav": 5008.24,
                "loglum": 42.5,
                "fwhm": 300,
                "velshift": 50,
                "profile": "gaussian"
            },
            "OIII_4959": {
                "wav": 4960.295,
                "linked_to": "OIII_5007",
                "ratio": 0.333  # Fixed 1:3 ratio
            },
            "Ha": {
                "wav": 6564.614,
                "loglum": 43.0,
                "fwhm": 500,
                "profile": "voigt",
                "gamma": 100  # Lorentzian component
            },
            "NII_6583": {
                "wav": 6585.27,
                "linked_to": "Ha",  # No ratio = free flux, linked FWHM
                "loglum": 42.0
            }
        }
    }
    """

    def __init__(self, wavelengths, param):
        self.wavelengths = wavelengths
        self.param = param
        self.spectrum = np.zeros_like(wavelengths)

    def update(self, param):
        """Update the emission line model with new parameters."""
        self.param = param

        spectrum = np.zeros_like(self.wavelengths)

        # Process lines in two passes:
        # 1. First pass: process independent lines
        # 2. Second pass: process linked lines

        line_data = {}  # Store processed line parameters

        # First pass: independent lines
        for line_name, line_params in param.items():
            if "linked_to" not in line_params:
                line_data[line_name] = self._process_line(line_params)

        # Second pass: linked lines (may need multiple iterations for chains)
        max_iterations = 10
        for _ in range(max_iterations):
            remaining = []
            for line_name, line_params in param.items():
                if "linked_to" in line_params and line_name not in line_data:
                    parent_name = line_params["linked_to"]
                    if parent_name in line_data:
                        line_data[line_name] = self._process_linked_line(
                            line_params, line_data[parent_name]
                        )
                    else:
                        remaining.append(line_name)
            if not remaining:
                break

        # Generate spectrum from all lines
        for line_name, data in line_data.items():
            spectrum += self._generate_line_profile(data)

        self.spectrum = spectrum

    def _process_line(self, line_params):
        """Process an independent line's parameters."""
        data = {
            "wav": line_params["wav"],
            "loglum": line_params["loglum"],
            "fwhm": line_params["fwhm"],
            "velshift": line_params.get("velshift", 0.),
            "profile": line_params.get("profile", "gaussian"),
            "gamma": line_params.get("gamma", 0.),
        }
        return data

    def _process_linked_line(self, line_params, parent_data):
        """Process a linked line's parameters."""
        # Always inherit FWHM from parent
        fwhm = parent_data["fwhm"]

        # Determine luminosity
        if "ratio" in line_params:
            # Linked flux: L_child = ratio * L_parent
            loglum = parent_data["loglum"] + np.log10(line_params["ratio"])
        else:
            # Free flux, linked FWHM only
            loglum = line_params["loglum"]

        data = {
            "wav": line_params["wav"],
            "loglum": loglum,
            "fwhm": fwhm,
            "velshift": line_params.get("velshift", parent_data["velshift"]),
            "profile": line_params.get("profile", parent_data["profile"]),
            "gamma": line_params.get("gamma", parent_data["gamma"]),
        }
        return data

    def _generate_line_profile(self, data):
        """Generate a line profile from processed line data."""
        # Apply velocity shift to wavelength
        central_wav = data["wav"] * (1. + data["velshift"] / 2.998e5)

        # Convert luminosity from erg/s to Lsun
        luminosity = 10**data["loglum"] / 3.826e33  # Lsun

        profile_type = data["profile"].lower()

        if profile_type == "gaussian":
            return self._gaussian(central_wav, data["fwhm"], luminosity)
        elif profile_type == "lorentzian":
            return self._lorentzian(central_wav, data["fwhm"], luminosity)
        elif profile_type == "voigt":
            return self._voigt(central_wav, data["fwhm"], data["gamma"],
                               luminosity)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}. "
                             "Supported: gaussian, lorentzian, voigt")

    def _gaussian(self, central_wav, fwhm_vel, luminosity):
        """Gaussian profile, normalized to integrate to luminosity.

        Parameters
        ----------
        central_wav : float
            Central wavelength in Angstroms
        fwhm_vel : float
            FWHM in km/s
        luminosity : float
            Integrated line luminosity in Lsun

        Returns
        -------
        profile : np.ndarray
            Line profile in Lsun/A
        """
        # Convert FWHM to sigma
        sigma_vel = fwhm_vel / 2.355

        # Convert velocity to wavelength
        sigma_aa = sigma_vel * central_wav / 2.998e5

        # Gaussian profile
        gauss = np.exp(-0.5 * (self.wavelengths - central_wav)**2 / sigma_aa**2)

        # Normalize to integrate to luminosity
        integral = np.trapezoid(gauss, x=self.wavelengths)
        if integral > 0:
            gauss = gauss / integral * luminosity

        return gauss

    def _lorentzian(self, central_wav, fwhm_vel, luminosity):
        """Lorentzian profile, normalized to integrate to luminosity.

        Parameters
        ----------
        central_wav : float
            Central wavelength in Angstroms
        fwhm_vel : float
            FWHM in km/s
        luminosity : float
            Integrated line luminosity in Lsun

        Returns
        -------
        profile : np.ndarray
            Line profile in Lsun/A
        """
        # Convert FWHM to gamma (half-width at half-maximum)
        gamma_vel = fwhm_vel / 2.

        # Convert velocity to wavelength
        gamma_aa = gamma_vel * central_wav / 2.998e5

        # Lorentzian profile
        lorentz = gamma_aa / (np.pi * ((self.wavelengths - central_wav)**2
                                       + gamma_aa**2))

        # Normalize to integrate to luminosity
        integral = np.trapezoid(lorentz, x=self.wavelengths)
        if integral > 0:
            lorentz = lorentz / integral * luminosity

        return lorentz

    def _voigt(self, central_wav, fwhm_vel, gamma_vel, luminosity):
        """Voigt profile, normalized to integrate to luminosity.

        The Voigt profile is a convolution of Gaussian and Lorentzian profiles.

        Parameters
        ----------
        central_wav : float
            Central wavelength in Angstroms
        fwhm_vel : float
            Gaussian FWHM in km/s
        gamma_vel : float
            Lorentzian FWHM in km/s. If 0, returns pure Gaussian.
        luminosity : float
            Integrated line luminosity in Lsun

        Returns
        -------
        profile : np.ndarray
            Line profile in Lsun/A
        """
        # If gamma is 0, just return Gaussian
        if gamma_vel <= 0:
            return self._gaussian(central_wav, fwhm_vel, luminosity)

        try:
            from scipy.special import voigt_profile
        except ImportError:
            raise ImportError("scipy is required for Voigt profiles")

        # Convert FWHM to sigma for Gaussian
        sigma_vel = fwhm_vel / 2.355
        sigma_aa = sigma_vel * central_wav / 2.998e5

        # Convert FWHM to gamma (HWHM) for Lorentzian
        gamma_aa = (gamma_vel / 2.) * central_wav / 2.998e5

        # Voigt profile using scipy
        profile = voigt_profile(self.wavelengths - central_wav,
                                sigma_aa, gamma_aa)

        # Normalize to integrate to luminosity
        integral = np.trapezoid(profile, x=self.wavelengths)
        if integral > 0:
            profile = profile / integral * luminosity

        return profile
