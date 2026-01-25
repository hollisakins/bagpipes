from __future__ import print_function, division, absolute_import

import numpy as np
import os
import re
import time
import warnings
import h5py
import contextlib
import shutil

from copy import deepcopy

try:
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        import pymultinest as pmn
    multinest_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: PyMultiNest import failed, fitting will use the Nautilus" +
          " sampler instead.")
    multinest_available = False

try:
    from nautilus import Sampler
    nautilus_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: Nautilus import failed, fitting with Nautilus will be " +
          "unavailable.")
    nautilus_available = False

try:
    import ultranest
    import ultranest.stepsampler
    ultranest_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: UltraNest import failed, fitting with UltraNest will be " +
          "unavailable.")
    ultranest_available = False

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0

from .. import utils
from .. import plotting

from .fitted_model import fitted_model
from .posterior import posterior


def _read_multinest_data(filename):
    """
    Read MultiNest data.

    By default, Fortran drops the "E" symbol for 3-digit exponent output
    (e.g., '0.148232-104'). This impacts the output files currently
    being written by MultiNest. For such caes, this reader inserts
    the "E" symbol into the number string so that the number can be
    converted to a float.

    Parameters
    ----------
    filename : str
        The filename to read.
    """
    # count the columns in the first row of data;
    # without a converter, genfromtxt will read 3-digit exponents as np.nan
    ncolumns = np.genfromtxt(filename, max_rows=1).shape[0]

    # insert "E" before the "+" or "-" exponent symbol if it is missing,
    # so the string can be converted to a float
    # '1.148232-104'  -> 1.148232e-104
    # '1.148232+104'  -> 1.148232e+104
    # '-1.148232-104' -> -1.148232e-104
    # '+1.148232+104' -> 1.148232e+104
    # '0.148232-104'  -> 1.482320e-105
    # '0.148232E-10'  -> 1.482320e-011
    # '1.148232'      -> 1.48232e+000
    convert = lambda s: float(re.sub(r'(\d)([\+\-])(\d)', r'\1E\2\3', s))
    converters = dict(zip(range(ncolumns), [convert] * ncolumns))

    return np.genfromtxt(filename, converters=converters, encoding=None)


class fit(object):
    """
    Top-level class for fitting models to observational data.

    Interfaces with MultiNest or nautilus to sample from the posterior
    distribution of a fitted_model object. Performs loading and saving of
    results.

    Parameters
    ----------
    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int - optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    """

    def __init__(self, galaxy, fit_instructions, run=".", time_calls=False,
                 n_posterior=500):

        self.run = run
        self.galaxy = galaxy
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = n_posterior

        # Set up the directory structure for saving outputs.
        if rank == 0:
            utils.make_dirs(run=run)

        # The base name for output files.
        self.fname = "pipes/posterior/" + run + "/" + self.galaxy.ID + "_"

        # A dictionary containing properties of the model to be saved.
        self.results = {}

        # If a posterior file already exists load it.
        if os.path.exists(self.fname[:-1] + ".h5"):
            file = h5py.File(self.fname[:-1] + ".h5", "r")

            self.posterior = posterior(self.galaxy, run=run,
                                       n_samples=n_posterior)

            fit_info_str = file.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float32", "float")
            fit_info_str = fit_info_str.replace("float64", "float")
            fit_info_str = fit_info_str.replace("np.float", "float")
            self.fit_instructions = eval(fit_info_str)

            for k in file.keys():
                self.results[k] = np.array(file[k])
                if np.sum(self.results[k].shape) == 1:
                    self.results[k] = self.results[k][0]

            if rank == 0:
                print("\nResults loaded from " + self.fname[:-1] + ".h5\n")

        # Set up the model which is to be fitted to the data.
        self.fitted_model = fitted_model(galaxy, self.fit_instructions,
                                         time_calls=time_calls)

    def fit(self, verbose=False, n_live=400, use_MPI=True,
            sampler="multinest", n_eff=0, discard_exploration=False,
            n_networks=4, pool=1, min_ess=400, resume="resume",
            use_stepsampler=True):
        """ Fit the specified model to the input galaxy data.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.

        sampler : string - optional
            The sampler to use. Available options are "multinest",
            "nautilus", and "ultranest".

        n_eff : float - optional
            Target minimum effective sample size. Only used by nautilus.

        discard_exploration : bool - optional
            Whether to discard the exploration phase to get more accurate
            results. Only used by nautilus.

        n_networks : int - optional
            Number of neural networks. Only used by nautilus.

        pool : int - optional
            Pool size used for parallelization. Only used by nautilus.
            MultiNest is parallelized with MPI.

        min_ess : int - optional
            Target minimum effective sample size. Only used by ultranest.

        resume : string - optional
            Resume behavior for ultranest: 'resume', 'overwrite',
            'subfolder', or 'resume-similar'. Default is 'resume'.

        use_stepsampler : bool - optional
            Whether to use a slice step sampler for ultranest. Recommended
            for high-dimensional problems (>20 parameters). Default is True.

        """
        if "lnz" in list(self.results):
            if rank == 0:
                print("Fitting not performed as results have already been"
                      + " loaded from " + self.fname[:-1] + ".h5. To start"
                      + " over delete this file or change run.\n")

            return

        # Figure out which sampling algorithm to use
        sampler = sampler.lower()

        if sampler == "multinest" and not multinest_available:
            if nautilus_available:
                sampler = "nautilus"
                print("MultiNest not available. Switching to nautilus.")
            elif ultranest_available:
                sampler = "ultranest"
                print("MultiNest not available. Switching to ultranest.")

        elif sampler == "nautilus" and not nautilus_available:
            if multinest_available:
                sampler = "multinest"
                print("Nautilus not available. Switching to MultiNest.")
            elif ultranest_available:
                sampler = "ultranest"
                print("Nautilus not available. Switching to ultranest.")

        elif sampler == "ultranest" and not ultranest_available:
            if nautilus_available:
                sampler = "nautilus"
                print("UltraNest not available. Switching to nautilus.")
            elif multinest_available:
                sampler = "multinest"
                print("UltraNest not available. Switching to MultiNest.")

        if sampler not in ["multinest", "nautilus", "ultranest"]:
            raise ValueError("Sampler {} not supported.".format(sampler))

        if not (multinest_available or nautilus_available or ultranest_available):
            raise RuntimeError("No sampling algorithm could be loaded.")

        if rank == 0 or not use_MPI:
            print("\nBagpipes: fitting object " + self.galaxy.ID + "\n")

            start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

            if sampler == "multinest":
                pmn.run(self.fitted_model.lnlike,
                        self.fitted_model.prior.transform,
                        self.fitted_model.ndim, n_live_points=n_live,
                        importance_nested_sampling=False, verbose=verbose,
                        sampling_efficiency="model",
                        outputfiles_basename=self.fname, use_MPI=use_MPI)

            elif sampler == "nautilus":
                n_sampler = Sampler(self.fitted_model.prior.transform,
                                    self.fitted_model.lnlike, n_live=n_live,
                                    n_networks=n_networks, pool=pool,
                                    n_dim=self.fitted_model.ndim,
                                    filepath=self.fname + ".h5")

                n_sampler.run(verbose=verbose, n_eff=n_eff,
                              discard_exploration=discard_exploration)

            elif sampler == "ultranest":
                # Wrapper functions for UltraNest's expected signatures
                def ultranest_transform(cube):
                    params = cube.copy()
                    self.fitted_model.prior.transform(params, ndim=0, nparam=0)
                    return params

                def ultranest_lnlike(params):
                    return self.fitted_model.lnlike(params, ndim=0, nparam=0)

                ultranest_dir = self.fname + "ultranest/"
                ndim = self.fitted_model.ndim

                u_sampler = ultranest.ReactiveNestedSampler(
                    self.fitted_model.params,
                    ultranest_lnlike,
                    transform=ultranest_transform,
                    log_dir=ultranest_dir,
                    resume=resume
                )

                # Use step sampler for better performance in high dimensions
                if use_stepsampler:
                    u_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                        nsteps=max(2 * ndim, 20),
                        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                    )

                # Adjust convergence criteria for high-dimensional problems
                u_result = u_sampler.run(
                    min_num_live_points=n_live,
                    min_ess=min_ess,
                    dlogz=0.5 + 0.1 * ndim if ndim > 20 else 0.5,
                    update_interval_volume_fraction=0.4 if ndim > 20 else 0.2,
                    show_status=verbose
                )

            os.environ["PYTHONWARNINGS"] = ""

        if rank == 0 or not use_MPI:
            runtime = time.time() - start_time

            print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

            # Load MultiNest outputs and save basic quantities to file.
            if sampler == "multinest":
                multinest_fname = self.fname + 'post_equal_weights.dat'
                samples2d = _read_multinest_data(multinest_fname)
                if (np.all(np.abs(samples2d[:, -1]) < 1e-300)
                    or np.all(samples2d[:, -1] < -9.9e+99)):
                    raise RuntimeError("Bagpipes loaded a corrupted Multinest "
                                       "posterior. This is usually because the "
                                       "bagpipes likelihood function crashed or"
                                       " returned  all NaNs or all infs. Common"
                                       " causes are bad input models/data e.g.,"
                                       " zero errors, or config.max_redshift "
                                       "set too low. Once fixed, you will need "
                                       "to delete the corrupted MultiNest files"
                                       " in pipes/posterior. There may be more"
                                       " information in terminal output above.")

                lnz_line = open(self.fname + "stats.dat").readline().split()
                self.results["samples2d"] = samples2d[:, :-1]
                self.results["lnlike"] = samples2d[:, -1]
                self.results["lnz"] = float(lnz_line[-3])
                self.results["lnz_err"] = float(lnz_line[-1])

            elif sampler == "nautilus":
                samples2d = np.zeros((0, self.fitted_model.ndim))
                log_l = np.zeros(0)
                while len(samples2d) < self.n_posterior:
                    result = n_sampler.posterior(equal_weight=True)
                    samples2d = np.vstack((samples2d, result[0]))
                    log_l = np.concatenate((log_l, result[2]))
                self.results["samples2d"] = samples2d
                self.results["lnlike"] = log_l
                self.results["lnz"] = n_sampler.log_z
                self.results["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)

            elif sampler == "ultranest":
                samples2d = u_result['samples']
                # Compute log-likelihoods for posterior samples
                log_l = np.array([ultranest_lnlike(s) for s in samples2d])

                self.results["samples2d"] = samples2d
                self.results["lnlike"] = log_l
                self.results["lnz"] = u_result['logz']
                self.results["lnz_err"] = u_result['logzerr']

            self.results["median"] = np.median(samples2d, axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                     (16, 84), axis=0)

            file = h5py.File(self.fname[:-1] + ".h5", "w")

            # This is necessary for converting large arrays to strings
            np.set_printoptions(threshold=10**7)
            file.attrs["fit_instructions"] = str(self.fit_instructions)
            np.set_printoptions(threshold=10**4)

            for k in self.results.keys():
                file.create_dataset(k, data=self.results[k])

            self.results["fit_instructions"] = self.fit_instructions

            file.close()

            os.system("rm " + self.fname + "*")

            # Clean up UltraNest directory if it exists
            ultranest_dir = self.fname + "ultranest/"
            if os.path.exists(ultranest_dir):
                shutil.rmtree(ultranest_dir)

            self._print_results()

            # Create a posterior object to hold the results of the fit.
            self.posterior = posterior(self.galaxy, run=self.run,
                                       n_samples=self.n_posterior)

    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        print("{:<25}".format("Parameter")
              + "{:>31}".format("Posterior percentiles"))

        print("{:<25}".format(""),
              "{:>10}".format("16th"),
              "{:>10}".format("50th"),
              "{:>10}".format("84th"))

        print("-"*58)

        for i in range(self.fitted_model.ndim):
            print("{:<25}".format(self.fitted_model.params[i]),
                  "{:>10.3f}".format(self.results["conf_int"][0, i]),
                  "{:>10.3f}".format(self.results["median"][i]),
                  "{:>10.3f}".format(self.results["conf_int"][1, i]))

        print("\n")

    def plot_corner(self, show=False, save=True):
        return plotting.plot_corner(self, show=show, save=save)

    def plot_1d_posterior(self, show=False, save=True):
        return plotting.plot_1d_posterior(self, show=show, save=save)

    def plot_sfh_posterior(self, show=False, save=True, colorscheme="bw"):
        return plotting.plot_sfh_posterior(self, show=show, save=save,
                                           colorscheme=colorscheme)

    def plot_spectrum_posterior(self, show=False, save=True):
        return plotting.plot_spectrum_posterior(self, show=show, save=save)

    def plot_calibration(self, show=False, save=True):
        return plotting.plot_calibration(self, show=show, save=save)
