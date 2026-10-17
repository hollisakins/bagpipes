**Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation**

This is a personal fork of bagpipes maintained by Hollis Akins, used for fitting JWST data in COSMOS-Web, ZENITH, and EMBER. For the canonical, stable version of the code see the upstream repository at `github.com/ACCarnall/bagpipes <https://github.com/ACCarnall/bagpipes>`_ and the documentation at `bagpipes.readthedocs.io <http://bagpipes.readthedocs.io>`_. Expect this fork to diverge from upstream and to contain breaking changes; if you are not actively collaborating with me, you probably want the upstream version.

**Notable changes in this fork**

- BPASS stellar models as the default option (BC03 still supported)
- Rewritten AGN model with multiple component types (Carnall-style + modified blackbody)
- Balmer-edge factor in AGN and power-law continuum models
- New dust attenuation and emission models, with separate stellar/nebular dust components in ``model_galaxy``
- Custom SFH models, plus an ``age_frac`` reparametrization for age-based SFHs
- Custom user-defined priors
- `UltraNest <https://github.com/JohannesBuchner/UltraNest>`_ as an additional nested-sampling backend
- Simplified ``fesc`` implementation
- Updated fitting infrastructure for the new model API; new filter curves

See the git log for the full history.

**Installing this fork**

The pip-installable upstream wheel ships the large model grid files, but the GitHub repo does not. The included ``install.sh`` script handles this by downloading the upstream wheel to a temp directory, copying the grids into your cloned repo, and then running an editable install:

.. code::

    git clone https://github.com/hollisakins/bagpipes.git
    cd bagpipes
    ./install.sh

Note that the upstream wheel only ships BC03 (+ DL07) grids. Because this fork defaults to BPASS, you will likely also need the BPASS grid files — these are not redistributed and should be obtained separately and dropped into ``bagpipes/models/grids/``.

**If you already have bagpipes installed**

The script's final step is ``pip install -e .``, which will replace any existing ``bagpipes`` install in the currently active environment. If you want to keep your upstream install intact, create a dedicated environment for the fork before running the script:

.. code::

    conda create -n bagpipes-fork python=3.11
    conda activate bagpipes-fork
    ./install.sh

Your original install in the other environment is untouched.

**Sampling algorithms within Bagpipes**

The default sampler (and historically the only option) for fitting models to data is the `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ code, however this requires separate installation, which can be challenging on some systems. Bagpipes is also compatible with the pure Python `nautilus <https://github.com/johannesulf/nautilus>`_ nested sampling algorithm, which is installed by default and will be automatically used for fitting if MultiNest is not installed. This fork additionally supports `UltraNest <https://github.com/JohannesBuchner/UltraNest>`_ as a nested-sampling backend (install separately with ``pip install ultranest``). For more information please see the `bagpipes documentation <http://bagpipes.readthedocs.io>`_.

**Published papers and citing the code**

Bagpipes is described primarily in Section 3 of `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_, with further development specific to spectroscopic fitting described in Section 4 of `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_. These papers are the best place to start if you want to understand how the code works.

If you make use of Bagpipes, please include a citation to `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_ in any publications. You may also consider citing `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_, particularly if you are fitting spectroscopy.

Please note development of the code has been ongoing since these works were published, so certain parts of the code are no longer as described. Please inquire if in doubt.


.. image:: docs/images/sfh_from_spec.png
