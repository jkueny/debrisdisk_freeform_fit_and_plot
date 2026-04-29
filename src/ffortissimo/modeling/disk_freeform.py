'''
NOTE: Turn parallelism OFF in fm.py (i.e., debug=True in the preamble)

Using the manager() thing in WindFM to create the shared memory dictionaries
increases the compute time by like 2 orders of magnitude... and is needed for
doing parallel processing with a worker pool. At least for my current pyklip settings,
single threaded is so much faster. 07/16/2025

'''

import os
import logging
import sys
import glob
from astropy.io import fits
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, rotate
from ffortissimo.utils.regularization import fit_elgauss_window 
from datetime import datetime

from ffortissimo.io.yaml_handling import read_config
from ffortissimo.io.fits_handling import save_fits
from ffortissimo.utils.sci_image_utils import parang_sort, diskprep_image_frames_parangs
from ffortissimo.utils.masks import make_annular_mask
from ffortissimo.utils.improc_tools import fft_power_spectrum, subtract_median_profile_np
from ffortissimo.dev.pyklip.klip import high_pass_filter
from ffortissimo.dev.pyklip.instruments.Instrument import GenericData
from ffortissimo.dev.pyklip.rdi import PSFLibrary
from ffortissimo.dev.pyklip.fmlib.diskfm import DiskFM
import ffortissimo.dev.pyklip.fm as fm
from ffortissimo.utils.make_gpi_psf_for_disks import make_disk_mask
import ffortissimo.utils.astro_unit_conversion as convert
from ffortissimo.modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
from scipy.signal import fftconvolve
from scipy.special import huber
from scipy.optimize import minimize
# import multiprocessing as mp

def get_basedir():
    basedir = os.environ.get("DISKFIT_BASEDIR",f'{os.environ["HOME"]}/data')
    return basedir

def apodize_mask_gaussian(mask, sigma=2.0):
    """Apodize a binary mask using Gaussian smoothing."""
    smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
    max_val = np.max(smoothed)
    if max_val > 0:
        smoothed = smoothed / max_val
    return smoothed

def generate_powerlaw_noise(shape, power_law_index, seed):
    """
    Generate a 2D noise array with a power spectrum following a given power law index.

    Parameters:
        shape (tuple): Shape of the output array (ny, nx).
        power_law_index (float): Power law index (beta) for the power spectrum (P(k) ~ k^beta).
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: 2D noise array.


    Example usage:
    noise = generate_powerlaw_noise((256, 256), -2.0, random_seed=42)
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    # Create frequency grid
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = np.inf  # avoid division by zero at the zero frequency

    # Power spectrum amplitude
    amplitude = k**(power_law_index / 2.0)
    amplitude[0, 0] = 0  # set DC component to zero

    # Generate random complex noise
    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    noise_ft = noise * amplitude

    # Inverse FFT to get spatial noise
    noise_spatial = np.fft.ifft2(noise_ft).real

    # Normalize to zero mean and unit variance
    noise_spatial -= np.mean(noise_spatial)
    noise_spatial /= np.std(noise_spatial)

    return noise_spatial


class FreeFormDisk:
    def __init__(self, config):
        self.params_file = read_config(config)
        self._load_dirs()
        self._load_metadata()
        self._load_klparams()
        self._load_initial_model_params()
        # self._render_reference_model()
        # self.klbasis = self._loadbasis()

        #load instrument PSF

        psf_path = os.path.join(self.klipdir, f"{self.file_prefix}_instrPSF.fits")

        psf = fits.getdata(psf_path)

        self.psf = psf / np.sum(psf)
        
    
    def _load_dirs(self):
        # Look for the project files in ~/projects by default
        # or in $DISKFIT_BASEDIR
        basedir = get_basedir()
        self.basedir = basedir
        klipdir = os.path.join(basedir, self.params_file["BAND_DIR"],
                                    "klip_fm_files")
        
        os.makedirs(klipdir, exist_ok=True)
        self.klipdir = klipdir


        results_dir_name = self.params_file.get('results_dir_name', 'results_freeform')
        resultsdir = os.path.join(basedir, self.params_file["BAND_DIR"],
                                  results_dir_name)
        os.makedirs(resultsdir, exist_ok=True)
        self.resultsdir = resultsdir

        self.datadir = os.path.join(basedir, self.params_file["BAND_DIR"])

        self.file_prefix = self.params_file["FILE_PREFIX"]

    def _load_metadata(self):
        self.pixscale = self.params_file["PIXSCALE_INS"]
        self.distance = self.params_file["DISTANCE_STAR"]
        self.wl = self.params_file["WL"]
        self.diam = self.params_file["PRIM_MIRROR_SZ"]

    def _load_klparams(self):
        self.numbasis = self.params_file["KLMODE_NUMBER"]
        self.annuli = self.params_file["ANNULI"]
        self.iwa = self.params_file["IWA"]
        self.owa = self.params_file["OWA"]
        self.minrot = self.params_file["MOVE_HERE"]
        self.mode = self.params_file["MODE"]
        self.move_here = self.params_file["MOVE_HERE"]
        aligned_center = self.params_file["ALIGNED_CENTER"]
        self.aligned_center = aligned_center
        self.image_size = int(np.ceil(aligned_center[0]) * 2)
        self.image_shape = (self.image_size, self.image_size)
        self.hp = self.params_file["HP_FILTER"]
        self.clean_final_fm = self.params_file["CLEAN_FINAL_FM"]

    
    def _engineer_disk_mask(self, mask2engineer, angle_sweep_factor):
        # self.par_angs should already be sorted, first element is negative
        delta_parang = (np.max(self.par_angs) - np.min(self.par_angs)) / angle_sweep_factor
        sweep_angle = np.arange(-delta_parang, delta_parang, 1)
        combined_masks = np.zeros_like(mask2engineer)
        for angle in sweep_angle:
            disk_mask = rotate(mask2engineer, angle, reshape=False)
            combined_masks += disk_mask
        combined_masks[combined_masks > 0.5] = 1
        combined_masks[combined_masks < 0.5] = 0
        return combined_masks.astype(np.int32)
    
    def _load_initial_model_params(self):
        disk_params = {}
        disk_params["r1"] = self.params_file["r1_init"]
        disk_params["r2"] = self.params_file["r2_init"]
        disk_params["rc"] = self.params_file["rc_init"]
        disk_params["alpha_in"] = self.params_file["alpha_in_init"]
        disk_params["alpha_out"] = self.params_file["alpha_out_init"]
        disk_params["beta"] = self.params_file["beta_init"]
        disk_params["a_r"] = self.params_file["a_r_init"]
        disk_params["inc"] = self.params_file["inc_init"]
        disk_params["pa"] = self.params_file["pa_init"]
        disk_params["dx"] = self.params_file["dx_init"]
        disk_params["dy"] = self.params_file["dy_init"]
        disk_params["Norm"] = self.params_file["N_init"]
        disk_params["g1"] = self.params_file["g1_init"]
        disk_params["g2"] = self.params_file["g2_init"]
        disk_params["alpha1"] = self.params_file["alpha1_init"]

        self.params_init = disk_params    

        param_priors = {}
        param_priors["rc"] = self.params_file["rc_prior"]
        param_priors["alpha_in"] = self.params_file["alpha_in_prior"]
        param_priors["alpha_out"] = self.params_file["alpha_out_prior"]
        param_priors["beta"] = self.params_file["beta_prior"]
        param_priors["a_r"] = self.params_file["a_r_prior"]
        param_priors["inc"] = self.params_file["inc_prior"]
        param_priors["pa"] = self.params_file["pa_prior"]
        param_priors["dx"] = self.params_file["dx_prior"]
        param_priors["dy"] = self.params_file["dy_prior"]
        param_priors["Norm"] = self.params_file["N_prior"]
        param_priors["g1"] = self.params_file["g1_prior"]
        param_priors["g2"] = self.params_file["g2_prior"]
        param_priors["alpha1"] = self.params_file["alpha1_prior"]
        
        # If we assign None in the param file it will be a string,
        # so we need to convert it to None
        for key, value in param_priors.items():
            for each, v in enumerate(value):
                if isinstance(v, str):
                    param_priors[key][each] = None
        self.param_priors = param_priors
        print(f"Using parameter priors: {self.param_priors}")
    
    def render_initial_disk_model(self):

        self._load_initial_model_params()

        beta = self.params_init["beta"]
        a_r = self.params_init["a_r"]
        inc = self.params_init["inc"]
        pa = self.params_init["pa"]
        dx = self.params_init["dx"]
        dy = self.params_init["dy"]

        R1 = self.params_init['r1']
        R2 = self.params_init['r2']

        Norm = self.params_init['Norm']
        g1 = self.params_init['g1']
        g2 = self.params_init['g2']
        alpha1 = self.params_init['alpha1']

        max_fov = self.image_size / 2. * self.pixscale  #maximum radial distance in AU from the center to the edge
        n_pts = int(np.floor(self.image_size / 1))
        xsize = max_fov * self.distance  #maximum radial distance in AU from the center to the edge

        #The coordinate system here [x,y,z] is defined :
        # +ve x is the line of sight
        # +ve y is going right from the center
        # +ve z is going up from the center

        # y = np.linspace(0,xsize,num=npts/2)
        y = np.linspace(-xsize, xsize, num=n_pts)
        z = np.linspace(-xsize, xsize, num=n_pts)
        
        beta = 1.
        rc = self.params_init['rc']
        m = self.params_init['alpha_in']
        n = self.params_init['alpha_out']
        model = fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                                    g1, g2, alpha1, a_r, rc, m, n,
                                    y_arr=y,
                                    z_arr=z,
                                    npts=n_pts,
                                    mask=(1 - self.mask2generatedisk))
        return model
    
    def _render_disk_model(self, params):
        # Unpack the parameters
        # Fixed parameters
        R1 = self.R1
        R2 = self.R2
        beta = 1.
        # Free parameters
        rc = params[0]
        m = params[1]
        n = params[2]
        a_r = params[3]
        inc = params[4]
        pa = params[5]
        dx = params[6]
        dy = params[7]
        Norm = np.exp(params[8])
        g1 = params[9]
        g2 = params[10]
        alpha1 = params[11]


        max_fov = self.image_size / 2. * self.pixscale  #maximum radial distance in AU from the center to the edge
        n_pts = int(np.floor(self.image_size / 1))
        xsize = max_fov * self.distance  #maximum radial distance in AU from the center to the edge

        #The coordinate system here [x,y,z] is defined :
        # +ve x is the line of sight
        # +ve y is going right from the center
        # +ve z is going up from the center

        # y = np.linspace(0,xsize,num=npts/2)
        y = np.linspace(-xsize, xsize, num=n_pts)
        z = np.linspace(-xsize, xsize, num=n_pts)
        
        model = fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                                    g1, g2, alpha1, a_r, rc, m, n,
                                    y_arr=y,
                                    z_arr=z,
                                    npts=n_pts,
                                    mask=(1 - self.mask2generatedisk)) 
        return model
    def _objective_function(self, params):
        '''
        Objective function for the simple disk model fit.
        '''
        # The parameter a_r can be a problem and needs to be regularized
        # print(f"Iteration {self.iteration}: {params}")
        penalty = np.square((params[3] - self.params_init["a_r"]) / (self.params_init["a_r"]))
        model = self._render_disk_model(params)
        # Convolve the disk model with the PSF
        psf = self.psf
        model_image = fftconvolve(model, psf, mode="same")

        weights = 1. / self.noise_map
        raw_loss = (model_image - self.data)**2 * weights
        mean_huber = np.mean(huber(0.1, raw_loss))

        # self.iteration += 1
        return mean_huber + penalty
    
    def fit_simple_disk_model(self):
        '''
        Use scipy.optimize.minimize to fit a simple disk model to the data.
        '''
        # Use scipy.optimize.minimize to fit a simple disk model to the data.
        self.iteration = 0
        self.R1 = self.params_init["r1"]
        self.R2 = self.params_init["r2"]
        # self.a_r = self.params_init["a_r"]
        x0 = np.asarray([
                         self.params_init["rc"], #0
                         self.params_init["alpha_in"], #1
                         self.params_init["alpha_out"], #2
                         self.params_init["a_r"], #3
                         self.params_init["inc"], #4
                         self.params_init["pa"], #5
                         self.params_init["dx"], #6
                         self.params_init["dy"], #7
                         np.log(self.params_init["Norm"]), #8
                         self.params_init["g1"], #9    
                         self.params_init["g2"], #10
                         self.params_init["alpha1"]]) #11
        
        bounds = [(self.param_priors["rc"][0], self.param_priors["rc"][1]), #0
                  (self.param_priors["alpha_in"][0], self.param_priors["alpha_in"][1]), #1
                  (self.param_priors["alpha_out"][0], self.param_priors["alpha_out"][1]), #2
                  (self.param_priors["a_r"][0], self.param_priors["a_r"][1]), #3
                  (self.param_priors["inc"][0], self.param_priors["inc"][1]), #4
                  (self.param_priors["pa"][0], self.param_priors["pa"][1]), #5
                  (self.param_priors["dx"][0], self.param_priors["dx"][1]), #6
                  (self.param_priors["dy"][0], self.param_priors["dy"][1]), #7
                  (self.param_priors["Norm"][0], self.param_priors["Norm"][1]), #8
                  (self.param_priors["g1"][0], self.param_priors["g1"][1]), #9
                  (self.param_priors["g2"][0], self.param_priors["g2"][1]), #10
                  (self.param_priors["alpha1"][0], self.param_priors["alpha1"][1]), #11
                  ]

        result = minimize(self._objective_function, x0,
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-8,
                          "disp": True},
                        #   callback=self._callback_function,
                          )
        self.params_opt = result.x
        best_model = self._render_disk_model(self.params_opt)
        self.best_model = best_model
        print(f"Reference model fit converged in {result.nit} iterations")
        print(f"Final loss: {result.fun}")
        print(f"Final parameters: {self.params_opt}")
        return best_model
    
    def _callback_function(self, params):
        '''
        Callback function for the simple disk model fit.
        '''
        print(f"Iteration {self.iteration}: {params}")
        self.iteration += 1

    def fit_reference_model(self, noise_map, reduced_data):
        reduced_data_local = np.array(reduced_data, copy=True)
        reduced_data_local[reduced_data_local != reduced_data_local] = 0.
        reduced_data_local *= self.mask2generatedisk
        self.data = reduced_data_local
        noise_map_local = np.array(noise_map, copy=True)
        noise_map_local[noise_map_local != noise_map_local] = 1.
        self.noise_map = noise_map_local
        if self.params_file["OPTIMIZE_REFERENCE"]:
            reference_model = self.fit_simple_disk_model()
        else:
            reference_model = self.render_initial_disk_model()
        reference_model_outpath = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel.fits")
        try:
            save_fits(reference_model_outpath, reference_model)
        except Exception as e:
            logging.warning(f"Failed to write {reference_model_outpath}: {e}")
        return reference_model

    def high_pass_reference_model(self, reference_model):
        sigma_size = min(self.window_params["width_x"], self.window_params["width_y"])
        filter_size = (self.image_size / sigma_size) / (2*np.sqrt(2*np.log(2)))
        print(f"HP filtering model ref w/ Gaussian FWHM of {filter_size} pixels in Fourier space...")
        reference_model_highpass = high_pass_filter(reference_model, filtersize=filter_size)
        # enforce positivity
        reference_model_hp_clamped = np.clip(reference_model_highpass, a_min=0, a_max=np.max(reference_model_highpass))
        reference_model_hp_rounded = np.round(reference_model_hp_clamped)
        reference_model_hp_rounded[reference_model_hp_rounded > 0] = 1
        reference_model_disk_spine = reference_model_hp_rounded

        refhp_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_HighPass.fits")
        save_fits(refhp_saveto, reference_model_disk_spine)
        return reference_model_disk_spine
    
    def get_reference_model_psd(self, reference_model):
        # reference_model_hp = high_pass_filter(reference_model, filtersize=100)
        # reference_model_hp_clamped = np.clip(reference_model_hp, a_min=0, a_max=np.max(reference_model_hp))
        # reference_model_hp_norm = reference_model_hp_clamped / np.linalg.norm(reference_model_hp_clamped)
        # reference_model_hp_norm_meansub = reference_model_hp_norm - np.mean(reference_model_hp_norm)
        reference_model_norm = reference_model / np.linalg.norm(reference_model)
        reference_model_meansub = reference_model_norm - np.mean(reference_model_norm)
        psd_ref_model = fft_power_spectrum(reference_model_meansub)
        # psd_ref_model_hp = fft_power_spectrum(reference_model_hp_norm_meansub)
        psd_ref_model = np.asarray(psd_ref_model)
        # psd_ref_model_hp = np.asarray(psd_ref_model_hp)
        print("Fitting the optimal window func to the reference model PSD...")
        params, window_opt = fit_elgauss_window(psd_ref_model)
        self.window_params = params
        psd_ref_model_win = window_opt + psd_ref_model
        # psd_ref_model_win_hp = window_opt + psd_ref_model_hp
        ref_model_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel.fits")
        try:
            save_fits(ref_model_saveto, reference_model)
        except Exception as e:
            logging.warning(f"Failed to write {ref_model_saveto}: {e}")
        psd_ref_model_win_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_Frequencies.fits")
        # psd_ref_model_win_hp_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_PSD_HighPass.fits")
        window_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_Window.fits")
        freq_template_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_Reference_FreqTemplate.fits")
        # save_fits(ref_model_hp_saveto, reference_model_hp_clamped)
        try:
            save_fits(psd_ref_model_win_saveto, psd_ref_model)
        except Exception as e:
            logging.warning(f"Failed to write {psd_ref_model_win_saveto}: {e}")
        try:
            save_fits(freq_template_saveto, psd_ref_model_win)
        except Exception as e:
            logging.warning(f"Failed to write {freq_template_saveto}: {e}")
        # save_fits(psd_ref_model_win_hp_saveto, psd_ref_model_win_hp)
        try:
            save_fits(window_saveto, window_opt)
        except Exception as e:
            logging.warning(f"Failed to write {window_saveto}: {e}")
        return psd_ref_model_win, window_opt
    

    def get_initial_model(self, loc_init_model=None, random_seed=0, init_min=-1.0, init_max=1.0):
        rng = np.random.default_rng(random_seed)

        # load in the model unless we're deliberately randomizing
        if loc_init_model is not None and random_seed == 0:
            logging.info(f"Loading the initial model from {loc_init_model}")
            model_init = fits.getdata(loc_init_model)
            assert model_init.shape == self.image_shape, "Shape mismatch with loaded initial model"
        else:
            logging.info(f"Randomizing the model with U~[{init_min}, {init_max}] in {self.image_shape}")

            if loc_init_model is not None:
                logging.warning(f"Got {loc_init_model=}, but non-default random seed ({random_seed}), so we're randomizing the initialization anyway")
            # we init the model fitting with just a noise image
            model_init = rng.uniform(init_min, init_max, self.image_shape)
        
        # model_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
        # save_fits(model_saveto, model_init)

        model_init *= self.mask2generatedisk

        return model_init

    def allocate_dataset(self):
        filelist = sorted(glob.glob(f'{self.datadir}/camsci*.fits'),
                          key=parang_sort)
        if len(filelist) == 0:
            raise ValueError(f"Could not find files in the dir: {self.datadir}")
        input_data, par_angs  = diskprep_image_frames_parangs(filelist)
        self.filelist = filelist
        self.par_angs = par_angs
        self.input_data = input_data
        self.frame_shape = input_data[0].shape
    
    def prep_dataset(self):
        
        input_centers = np.array([self.aligned_center for _ in range(len(self.filelist))])
        # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
        IWA = self.iwa
        dataset = GenericData(self.input_data,
                             input_centers,
                             parangs=self.par_angs,
                             IWA=IWA,filenames=self.filelist)

        dataset.OWA = self.params_file["OWA"]
        if dataset.input.shape[1] != dataset.input.shape[2]:
            raise ValueError(""" Data slices are not square (dimx!=dimy), 
                            please make them square""")
        if self.mode == "RDI":
            psflib = self.initialize_rdi(dataset)
        else:
            psflib = None
        
        return dataset, psflib
    
    def initialize_rdi(self, dataset):
        do_rdi_correlation = self.params_file["DO_RDI_CORRELATION"]
        hp = self.hp
        rdi_matrix_dir = os.path.join(self.datadir, "rdi_matrix")
        IWA = self.iwa
        ref_files = sorted(glob.glob(f'{self.datadir}/*parang*.fits'))
        if len(ref_files) == 0:
            raise ValueError(f"Could not find files in the dir: {self.datadir}")
        ref_data, ref_par_angs  = diskprep_image_frames_parangs(ref_files)
        ref_centers = np.array([self.aligned_center for _ in range(len(ref_files))])
        existing_corr_matrix = os.path.exists(os.path.join(rdi_matrix_dir, 'corr_matrix.fits'))
        if do_rdi_correlation or not existing_corr_matrix:
            if not existing_corr_matrix:
                print('No existing RDI correlation matrix found, computing...')
            else:
                print('RDI correlation matrix redo requested, recomputing...')
            psflib = PSFLibrary(ref_data,
                                self.aligned_center,
                                ref_files,
                                compute_correlation=True,
                                highpass=hp)

            # save the correlation matrix to disk so that we also don't need to
            # recomptue this ever again. In the future we can just pass in the
            # correlation matrix into the PSFLibrary object rather than having it
            # compute it
            os.makedirs(rdi_matrix_dir, exist_ok=True)
            psflib.save_correlation(os.path.join(rdi_matrix_dir,
                                                "corr_matrix.fits"),
                                overwrite=True)


        # load the correlation matrix
        corr_matrix = fits.getdata(os.path.join(rdi_matrix_dir,
                                            'corr_matrix.fits'))
        psflib = PSFLibrary(ref_data,
                            self.aligned_center,
                            filenames=ref_files,
                            correlation_matrix=corr_matrix,
                            highpass=hp)
        psflib.prepare_library(dataset)
        return psflib
        
    
    def prep_binary_masks(self):
        # mask2generatedisk = 1 - mask_disk_zeros
        # Make the mask
        x_off = self.params_file["MASK_DX"]
        y_off = self.params_file["MASK_DY"]
        aligned_center = self.aligned_center
        image_size = (round(aligned_center[0]) * 2, round(aligned_center[1]) * 2)
        mask_center = aligned_center[0] + x_off, aligned_center[1] + y_off


        save_mask_part = os.path.join(self.klipdir,
                                    f"{self.file_prefix}")

        in_scaling = self.params_file["MASK_IN_SCALING"]
        out_scaling = self.params_file["MASK_OUT_SCALING"]
        noise_in_scaling = self.params_file["MASK_NOISE_IN"]
        noise_out_scaling = self.params_file["MASK_NOISE_OUT"]
        mask_speckles = self.params_file["MASK_SPECKLES"]
        inc_init = self.params_file['inc_init']
        pa_init = self.params_file['pa_init']
        mask_disk_zeros = make_disk_mask(
            image_size[0],
            pa_init,
            inc_init,
            convert.au_to_pix(self.params_file['r1_init'],
                              self.pixscale,
                              self.distance) -
            in_scaling / np.cos(np.radians(inc_init)),
            convert.au_to_pix(self.params_file['r2_init'],
                              self.pixscale,
                              self.distance) +
            out_scaling / np.cos(np.radians(inc_init)),
            aligned_center=mask_center)
        
        mask_noise_zeros = make_disk_mask(
            image_size[0],
            pa_init,
            inc_init,
            convert.au_to_pix(self.params_file['r1_init'],
                              self.pixscale,
                              self.distance) -
            noise_in_scaling / np.cos(np.radians(inc_init)),
            convert.au_to_pix(self.params_file['r2_init'],
                              self.pixscale,
                              self.distance) +
            noise_out_scaling / np.cos(np.radians(inc_init)),
            aligned_center=mask_center)
        
        mask2generatedisk = 1 - mask_disk_zeros

        mask4noisemap = 1 - mask_noise_zeros


        ### a few lines to create a circular central mask to hide center regions with a lot
        ### of speckles. Currently not using it but it's there
        mask_out_of_bounds = make_annular_mask(image_size, mask_speckles, self.owa)
        # x = np.arange(image_size[0], dtype=float)[None,:] - aligned_center[0]
        # y = np.arange(image_size[1], dtype=float)[:,None] - aligned_center[1]
        # rho2d = np.sqrt(x**2 + y**2)
        # out_of_bounds = rho2d < self.owa | rho2d > mask_speckles
        # mask_out_of_bounds[np.where(out_of_bounds)] = 0.
        mask2generatedisk = mask2generatedisk*(mask_out_of_bounds)
        mask2generatedisk[np.where(mask2generatedisk < 0.5)] = 0
        mask2generatedisk[np.where(mask2generatedisk > 0.5)] = 1

        self.mask2generatedisk = mask2generatedisk
        self.mask4noisemap = mask4noisemap
        disk_mask_apod = apodize_mask_gaussian(mask2generatedisk, sigma=3.0)
        self.disk_mask_apod = disk_mask_apod
        # engineered_optimization_map = self._engineer_disk_mask(mask2generatedisk, angle_sweep_factor=4)
        optimization_mask = mask2generatedisk
        # optimization_mask = engineered_optimization_map
        fits.writeto(f"{save_mask_part}_optimization_mask.fits",
                     optimization_mask, overwrite=True)
        fits.writeto(f"{save_mask_part}_mask2generatedisk.fits",
                     mask2generatedisk, overwrite=True)
        fits.writeto(f"{save_mask_part}_disk_mask_apod.fits",
                     disk_mask_apod, overwrite=True)
        fits.writeto(f"{save_mask_part}_mask_out_of_bounds.fits",
                     mask_out_of_bounds, overwrite=True)
        return optimization_mask

    def run_klip_reduction(self, dataset, model_init=None, psflib=None):
        """
        Run KLIP data reduction and save the basis file and reduced image.
        
        This method handles only the KLIP reduction process:
        - Creates DiskFM object and saves KL basis
        - Runs KLIP reduction via fm.klip_dataset()
        - Loads and optionally cleans the reduced data
        - Saves the reduced data to disk
        
        Args:
            dataset: GenericData object containing the input images
            model_init: Initial disk model image (2D array, optional). If None, 
                       runs pure KLIP reduction without forward modeling.
            psflib: PSF library for RDI mode (optional)
            
        Returns:
            reduced_data: The KLIP-reduced image (2D array)
        """
        # Create DiskFM object for forward modeling (or pure KLIP if no model)
        if model_init is not None:
            model_disk = np.asarray(model_init)
        else:
            model_disk = None
        
        diskobj = DiskFM(dataset.input.shape,
                        self.numbasis,
                        dataset,
                        model_disk=model_disk,
                        basis_filename=os.path.join(
                            self.klipdir, self.file_prefix + '_klbasis.h5'),
                        save_basis=True,
                        aligned_center=self.aligned_center)
        
        # Run KLIP reduction
        maxnumbasis = dataset.input.shape[0]
        time_start = datetime.now()
        fm.klip_dataset(dataset,
                        fm_class=diskobj,
                        numbasis=self.numbasis,
                        maxnumbasis=maxnumbasis,
                        annuli=self.annuli,
                        mode=self.mode,
                        subsections=1,
                        outputdir=self.klipdir,
                        fileprefix=self.file_prefix,
                        aligned_center=self.aligned_center,
                        highpass=self.hp,
                        minrot=self.move_here,
                        calibrate_flux=False,
                        time_collapse='median',
                        psf_library=psflib)
        
        print(f"klip_dataset() took {datetime.now() - time_start}.")

        # Load the reduced data
        path_rd = os.path.join(self.klipdir, f"{self.file_prefix}-klipped-KLmodes-all.fits")
        reduced_data = fits.getdata(path_rd)

        # Optionally clean the reduced data with median profile subtraction
        if self.clean_final_fm:
            reduced_data, _ = subtract_median_profile_np(reduced_data,
                                                    self.aligned_center)
            save_fits(path_rd, reduced_data)
        
        return reduced_data

    def initialize_diskfm(self, dataset, model_init, psflib=None):
        """
        Initialize disk forward modeling: run KLIP reduction and perform model operations.
        
        This method:
        1. Saves initial model files
        2. Runs KLIP reduction (via run_klip_reduction)
        3. Creates noise maps and masked data files
        4. Performs forward modeling of the initial model
        
        Args:
            dataset: GenericData object containing the input images
            model_init: Initial disk model image (2D array)
            psflib: PSF library for RDI mode (optional)
        """
        # Save initial model files
        model_convolved = convolve2d(model_init, self.psf, mode="same")
        model_init_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
        model_convolved_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel_Conv.fits")
        save_fits(model_init_saveto, model_init)
        save_fits(model_convolved_saveto, model_convolved)

        # Run KLIP reduction (this saves the basis and reduced image)
        reduced_data = self.run_klip_reduction(dataset, model_init, psflib)

        bespoke_noise_mask = self._engineer_disk_mask(self.mask4noisemap, angle_sweep_factor=4)
        if self.mode == "ADI":
            reduced_noise_masked = reduced_data * (1 - bespoke_noise_mask)
            tosave_reduced_noise_masked = reduced_data * bespoke_noise_mask
        else:
            reduced_noise_masked = reduced_data * (1 - self.mask4noisemap)
            tosave_reduced_noise_masked = reduced_data * self.mask4noisemap

        reduced_disk_masked = reduced_data * self.mask2generatedisk

        # Make the noisemap now, to inspect after initialization in case changes need to occur
        noise_map = self.make_noise_map_rings(reduced_data_no_disk=reduced_noise_masked)

        noise_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_noisemap.fits")
        fits.writeto(noise_saveto, noise_map, overwrite=True)

        saveto_masked_data = os.path.join(self.klipdir, f"{self.file_prefix}_masked_data.fits")
        saveto_masked_noise = os.path.join(self.klipdir, f"{self.file_prefix}_mask4noisemap.fits")
        if self.mode == "ADI":
            saveto_bespoke_noise = os.path.join(self.klipdir, f"{self.file_prefix}_bespoke_noise_mask.fits")
            saveto_bespoke_noise = os.path.join(self.klipdir, f"{self.file_prefix}_mask4noisemap.fits")
            save_fits(saveto_bespoke_noise, bespoke_noise_mask)
            self.noise_map = bespoke_noise_mask
        save_fits(saveto_masked_data, reduced_disk_masked)
        save_fits(saveto_masked_noise, tosave_reduced_noise_masked)

        model_fm_init = self.single_fm(np.asarray(model_convolved))

        model_fm_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel_FM.fits")

        save_fits(model_fm_saveto, model_fm_init)

        sys.stdout = sys.__stdout__
    

    def single_fm(self, model_image):
        # Refresh the windFM object
        basis_file_path = os.path.join(self.klipdir, f"{self.file_prefix}_klbasis.h5")

        diskfm = DiskFM(inputs_shape=None,
                            numbasis=None,
                            dataset=None,
                            model_disk=model_image,
                            basis_filename=basis_file_path,
                            load_from_basis=True)
        
        diskfm.update_disk(model_image)

        model_fm = diskfm.fm_parallelized()[0]



        return model_fm

    def high_pass_model(self, model_image, hp_filtersize=None):
        if hp_filtersize is None:
            hp_filtersize = (self.psf.shape[0]/self.hp)*2/np.sqrt(2*np.log(2))
        model_image_hp = high_pass_filter(model_image, filtersize=hp_filtersize)
        return model_image_hp
    
    def make_noise_map_rings(self, reduced_data_no_disk, delta_radii=1):
        if len(reduced_data_no_disk.shape) > 2:
            reduced_data_no_disk = np.squeeze(reduced_data_no_disk)
        h, w = reduced_data_no_disk.shape
        reduced_data_no_disk = np.array(reduced_data_no_disk, copy=True)
        reduced_data_no_disk[reduced_data_no_disk == 0.] = np.nan
        image_center = (h / 2 - 0.5, w / 2 - 0.5)
        # create rho2D for the rings
        x = np.arange(h, dtype=np.float64)[None, :] - image_center[0]
        y = np.arange(w, dtype=np.float64)[:, None] - image_center[1]
        rho2d = np.sqrt(x**2 + y**2)

        n_rings = int(self.params_file["OWA"] / delta_radii) - 2
        ring_radii = np.array([(i_ring + 0.5) * delta_radii for i_ring in range(n_rings)])
        ring_values = np.full(n_rings, np.nan)

        noise_map = np.zeros((h, w))
        for i_ring in range(n_rings):
            wh_rings = (rho2d >= i_ring * delta_radii) & (rho2d < (i_ring + 1) * delta_radii)
            val = np.nanstd(reduced_data_no_disk[wh_rings])
            noise_map[wh_rings] = val
            ring_values[i_ring] = val

        check_nans_mask = (rho2d + 1) > self.iwa  # region of interest (outside IWA)

        if np.sum(np.isnan(noise_map[check_nans_mask])) > 0:
            print(f"Warning: NaNs found in noise map at radii > IWA, inpainting affected annuli")
            valid = np.isfinite(ring_values)
            if np.sum(valid) >= 2:
                # Interpolate NaN ring values from valid neighboring annuli
                ring_values_inpainted = np.interp(
                    ring_radii,
                    ring_radii[valid],
                    ring_values[valid]
                )
                nan_rings = ~valid
                ring_values[nan_rings] = ring_values_inpainted[nan_rings]
                for i_ring in np.where(nan_rings)[0]:
                    wh_rings = (rho2d >= i_ring * delta_radii) & (rho2d < (i_ring + 1) * delta_radii)
                    noise_map[wh_rings] = ring_values[i_ring]
            else:
                fill_val = np.nanmedian(ring_values[valid]) if np.any(valid) else 1.0
                noise_map[np.isnan(noise_map) & check_nans_mask] = fill_val

        return noise_map

    def make_ring_stat_maps(self, reduced_data, delta_radii=1):
        if len(reduced_data.shape) > 2:
            reduced_data = np.squeeze(reduced_data)
        h, w = reduced_data.shape
        reduced_data = np.array(reduced_data, copy=True)
        reduced_data[reduced_data == 0.] = np.nan
        image_center = (h / 2 - 0.5, w / 2 - 0.5)

        x = np.arange(h, dtype=np.float64)[None, :] - image_center[0]
        y = np.arange(w, dtype=np.float64)[:, None] - image_center[1]
        rho2d = np.sqrt(x**2 + y**2)

        std_map = np.zeros((h, w))
        mean_map = np.zeros((h, w))
        median_map = np.zeros((h, w))
        for i_ring in range(0,
                            int(self.params_file["OWA"] / delta_radii) - 2):
            wh_rings = (rho2d >= i_ring * delta_radii) & (rho2d < (i_ring + 1) * delta_radii)
            std_map[wh_rings] = np.nanstd(reduced_data[wh_rings])
            mean_map[wh_rings] = np.nanmean(reduced_data[wh_rings])
            median_map[wh_rings] = np.nanmedian(reduced_data[wh_rings])

        return std_map, mean_map, median_map

    def inject_recover_mode(self, injected_dir):
        """
        Prepare for recovery of the injected disk.
        """
        print(f"Using injected data directory: {injected_dir}")
        # Override datadir to point to injected directory (where FITS files are)
        self.datadir = injected_dir
        # Override klipdir to point to klip_fm_files subdirectory in injected directory
        self.klipdir = os.path.join(injected_dir, "klip_fm_files")
        os.makedirs(self.klipdir, exist_ok=True)
        # Override resultsdir to point to configured subdirectory in injected directory
        self.resultsdir = os.path.join(injected_dir, self.params_file.get('results_dir_name', 'results_freeform'))
        os.makedirs(self.resultsdir, exist_ok=True)
        print(f"Output will be saved to: {self.klipdir}")
        print(f"Results will be saved to: {self.resultsdir}")
        
        
        # Override current disk params with injected test params
        self.params_init['pa'] = self.params_file.get('pa_test')
        self.params_init['inc'] = self.params_file.get('inc_test')
        self.params_init['r1'] = self.params_file.get('r1_test')
        self.params_init['r2'] = self.params_file.get('r2_test')
        self.params_init['rc'] = self.params_file.get('rc_test')
        self.params_init['alpha_in'] = self.params_file.get('alpha_in_test')
        self.params_init['alpha_out'] = self.params_file.get('alpha_out_test')
        self.params_init['beta'] = self.params_file.get('beta_test')
        self.params_init['a_r'] = self.params_file.get('a_r_test')
        self.params_init['dx'] = self.params_file.get('dx_test')
        self.params_init['dy'] = self.params_file.get('dy_test')
        self.params_init['N'] = self.params_file.get('N_test')
        self.params_init['g1'] = self.params_file.get('g1_test')
        self.params_init['g2'] = self.params_file.get('g2_test')
        self.params_init['alpha1'] = self.params_file.get('alpha1_test')
        
        # Overrride prior params with injected test params
        self.param_priors['rc'] = self.params_file.get('rc_test_prior')
        self.param_priors['alpha_in'] = self.params_file.get('alpha_in_test_prior')
        self.param_priors['alpha_out'] = self.params_file.get('alpha_out_test_prior')
        self.param_priors['beta'] = self.params_file.get('beta_test_prior')
        self.param_priors['a_r'] = self.params_file.get('a_r_test_prior')
        self.param_priors['inc'] = self.params_file.get('inc_test_prior')
        self.param_priors['pa'] = self.params_file.get('pa_test_prior')
        self.param_priors['dx'] = self.params_file.get('dx_test_prior')
        self.param_priors['dy'] = self.params_file.get('dy_test_prior')
        self.param_priors['N'] = self.params_file.get('N_test_prior')
        self.param_priors['g1'] = self.params_file.get('g1_test_prior')
        self.param_priors['g2'] = self.params_file.get('g2_test_prior')
        self.param_priors['alpha1'] = self.params_file.get('alpha1_test_prior')


        print(f"Using overriden test model params: {self.params_init}")
        print(f"Using overriden param priors: {self.param_priors}")
        
        
        # Ensure bounds are in [0, 360) range
        self.params_file['pa_prior'][0] = self.params_file['pa_prior'][0] % 360
        self.params_file['pa_prior'][1] = self.params_file['pa_prior'][1] % 360
        
