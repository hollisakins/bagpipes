"""IGM Lyman-alpha damping wing from a uniformly neutral IGM (Miralda-Escude 1998).

Analytic optical depth redward of Lya for a source at z_s seen through an IGM with neutral
fraction x_HI between z_end (where reionization is complete, below which the gas is ionized)
and z_u, the upper edge of the neutral region. z_u = z_s unless the source sits in an
ionized bubble of proper radius R_ion (Mpc), in which case 1 + z_u = (1 + z_s)(1 - H(z_s) R_ion / c).
The form is that of Miralda-Escude (1998, eq. 3) as written by Totani et al. (2006, eq. 11-13):

    tau(lambda_obs) = x_HI tau_GP(z_s) R_alpha / pi * ((1+z_obs)/(1+z_s))^(3/2) * [I(x_u) - I(x_l)]
    1 + z_obs = lambda_obs / lambda_alpha,  x_u = (1+z_u)/(1+z_obs),  x_l = (1+z_end)/(1+z_obs)
    I(x) = x^(9/2)/(1-x) + 9/7 x^(7/2) + 9/5 x^(5/2) + 3 x^(3/2) + 9 x^(1/2) - 9/2 ln[(1+sqrt x)/(1-sqrt x)]

with R_alpha = Lambda_alpha lambda_alpha / (4 pi c) = 2.02e-8 and tau_GP the Gunn-Peterson optical
depth of a fully neutral IGM at z_s (bagpipes cosmology, Omega_b h^2 = 0.0224, X_H = 0.76).
Wavelengths blueward of lambda_alpha (1 + z_u) have passed through resonance inside the neutral
region and are set to zero transmission; bagpipes' Inoue (2014) IGM is applied independently.

Model component (fit_instructions / model_components):

    "igm_dw": {"x_HI": 0.5,        # volume-averaged neutral fraction, 0-1
               "z_end": 6.0,       # optional, lower edge of the neutral IGM (default 6)
               "R_ion": 0.0}       # optional, proper radius of an ionized bubble around the source, Mpc
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from .. import utils

c_cgs = 2.99792458e10        # cm/s
e_charge = 4.80320425e-10    # esu
m_e = 9.1093837e-28          # g
m_H = 1.6735575e-24          # g
f_Lya = 0.4164
wl_Lya = 1215.67             # Angstrom
A_Lya = 6.265e8              # s^-1
Omega_b_h2 = 0.0224
X_H = 0.76
Mpc_cm = 3.0856776e24

R_alpha = A_Lya * wl_Lya * 1e-8 / (4.0 * np.pi * c_cgs)   # 2.02e-8


def n_H(z):
    """Mean cosmic hydrogen number density (cm^-3)."""
    rho_crit_h2 = 1.8785e-29                                 # g cm^-3 for h = 1
    return X_H * Omega_b_h2 * rho_crit_h2 / m_H * (1.0 + z)**3


def H_z(z):
    """Hubble parameter in s^-1 for the bagpipes cosmology."""
    return utils.cosmo.H(z).to("1/s").value


def tau_GP(z):
    """Gunn-Peterson optical depth of a fully neutral IGM at redshift z."""
    return np.pi * e_charge**2 * f_Lya * wl_Lya * 1e-8 * n_H(z) / (m_e * c_cgs * H_z(z))


def _I(x):
    sx = np.sqrt(x)
    return (x**4.5 / (1.0 - x) + 9.0/7.0 * x**3.5 + 9.0/5.0 * x**2.5 + 3.0 * x**1.5 + 9.0 * sx
            - 4.5 * np.log((1.0 + sx) / (1.0 - sx)))


def igm_dw_trans(wl_rest, z_s, x_HI, z_end=6.0, R_ion=0.0):
    """Transmission e^-tau on a rest-frame wavelength grid (Angstrom) for a source at z_s."""
    trans = np.ones_like(wl_rest, dtype=float)
    if x_HI <= 0.0:
        return trans
    one_pz_u = (1.0 + z_s) * (1.0 - H_z(z_s) * R_ion * Mpc_cm / c_cgs)
    one_pz_obs = wl_rest / wl_Lya * (1.0 + z_s)
    red = one_pz_obs > one_pz_u                      # redward of Lya at the edge of the neutral region
    trans[~red] = 0.0
    x_u = one_pz_u / one_pz_obs[red]
    x_l = np.minimum((1.0 + z_end) / one_pz_obs[red], x_u)
    tau = (x_HI * tau_GP(z_s) * R_alpha / np.pi * (one_pz_obs[red] / (1.0 + z_s))**1.5
           * (_I(x_u) - _I(x_l)))
    trans[red] = np.exp(-np.clip(tau, 0.0, 700.0))
    return trans
