# CLAUDE.md - Bagpipes Codebase Guide

## Project Overview

**Bagpipes** (Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation) is a Python package for:
- Generating synthetic galaxy spectra from physical models
- Fitting spectroscopic and photometric observations to extract galaxy properties
- Recovering star formation histories, stellar masses, dust properties, and AGN contributions

**Version:** 1.3.4
**Author:** Adam Carnall
**Documentation:** https://bagpipes.readthedocs.io
**License:** Apache 2.0

## Directory Structure

```
bagpipes/
├── bagpipes/                  # Main package
│   ├── models/               # Galaxy spectral modeling
│   │   ├── model_galaxy.py   # Core model class (main entry point)
│   │   ├── stellar_model.py  # Stellar continuum emission
│   │   ├── nebular_model.py  # Emission line models
│   │   ├── dust_attenuation_model.py
│   │   ├── dust_emission_model.py
│   │   ├── agn_model.py      # AGN continuum + lines
│   │   ├── star_formation_history.py
│   │   └── grids/            # Large FITS model grids (~2.2GB)
│   ├── fitting/              # Bayesian fitting framework
│   │   ├── fit.py            # Main fitting class
│   │   ├── fitted_model.py   # Model wrapper for fitting
│   │   ├── posterior.py      # Posterior sampling
│   │   └── prior.py          # Prior distributions
│   ├── input/                # Data handling
│   │   └── galaxy.py         # Galaxy data container
│   ├── filters/              # Photometric filter curves
│   ├── plotting/             # Visualization utilities
│   ├── config.py             # Global configuration
│   └── utils.py              # Utility functions
├── docs/                     # Sphinx documentation
├── examples/                 # Jupyter notebook tutorials
└── setup.py                  # Package configuration
```

## Key Entry Points

### 1. Model Galaxy Creation
```python
import bagpipes as pipes

model_components = {
    "redshift": 0.5,
    "burst": {"age": 1.0, "massformed": 11.0, "metallicity": 1.0},
    "nebular": {"logU": -2.0},
    "dust_atten": {"type": "Calzetti", "Av": 0.5}
}

model = pipes.model_galaxy(model_components, filt_list=filters, spec_wavs=wavs)
```

### 2. Fitting Observations
```python
galaxy = pipes.galaxy("id", load_data=my_loader, filt_list=filters)
fit = pipes.fit(galaxy, fit_instructions)
fit.fit(sampler="nautilus")  # or "multinest"
```

## Core Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `model_galaxy` | `models/model_galaxy.py` | Generate synthetic galaxy spectra |
| `galaxy` | `input/galaxy.py` | Container for observational data |
| `fit` | `fitting/fit.py` | Bayesian parameter estimation |
| `fit_catalogue` | `catalogue/fit_catalogue.py` | Batch fitting multiple galaxies |

## Model Components

Available components for `model_components` dict:
- **SFH types:** burst, constant, exponential, delayed, lognormal, dblplaw, iyer, custom
- **dust_atten:** Calzetti, Cardelli, CF00, Salim, VW07, QSO
- **dust_emission:** Draine & Li 2007 models
- **nebular:** Cloudy-based emission lines
- **agn:** MBB or broken power-law AGN models
- **igm:** Inoue 2014 IGM attenuation
- **pyneb_continuum:** (fork) PyNeb nebular continuum, normalised at 5100 A, added *after* dust
- **pyneb_nebular:** (fork, 2026-10-06) PyNeb nebular continuum **and** H I Balmer/Paschen
  lines from one normalisation `logLHb` at the same `Te`, `ne`; attenuated by the diffuse-ISM
  screen (`attenuate: False` to add after dust). Decoupled from the stars (no Q(H)/f_esc
  assumption). Te <= 30 kK (PyNeb line tables). Shapes are tabulated on a (Te, ne) grid and
  cached in `~/.cache/bagpipes/pyneb_nebular_<hash>.npz` (`BAGPIPES_CACHE` overrides);
  first build ~75 s, then ~0.1 ms per update. Posterior stores
  `spectrum_full_pyneb_nebular`. Built for the Twin Peaks nebular-continuum test
  (`~/Dropbox/research/projects/zenith/twinpeaks/results/nebular_test_2026-10-04/`).

## Development Commands

```bash
# Install in development mode
pip install -e .

# Build documentation
cd docs && make html

# Run example notebooks
jupyter notebook examples/
```

## Dependencies

**Required:** numpy, scipy, astropy, h5py, pandas, matplotlib, corner, spectres, msgpack

**Sampling (one required):**
- nautilus-sampler (>=1.0.2) - Pure Python, default
- pymultinest (>=2.11) - Fortran-based, requires MultiNest installation

## Code Patterns

1. **Component-based modeling:** Physical components (stellar, nebular, dust, AGN) are modular and combined in `model_galaxy`

2. **Lazy evaluation:** Spectral calculations triggered by `model.update()` method

3. **Configuration-driven:** Global settings in `config.py` control grids, wavelength sampling

4. **Energy balance:** Dust attenuation energy is conserved in dust emission

## Important Notes

- Cannot run from cloned repo without downloading large FITS grid files (~2.2GB)
- Recent changes split `'dust'` dict into separate `'dust_atten'` and `'dust_emission'`
- Filter curves stored in `bagpipes/filters/` as plain text files
- Posteriors saved as HDF5 files in `pipes/posterior/`

## Testing

No formal test suite. Functionality validated through:
- Example Jupyter notebooks in `examples/`
- Real-world usage in ~230 published papers
