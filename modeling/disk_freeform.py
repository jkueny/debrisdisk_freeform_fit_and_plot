'''
NOTE: Turn parallelism OFF in fm.py (i.e., debug=True in the preamble)

Using the manager() thing in WindFM to create the shared memory dictionaries
increases the compute time by like 2 orders of magnitude... and is needed for
doing parallel processing with a worker pool. At least for my current pyklip settings,
single threaded is so much faster. 07/16/2025

'''

import os
import sys
import glob
from astropy.io import fits
import numpy as np
from scipy.signal import convolve2d
from utils.regularization import fit_elgauss_window 
from datetime import datetime

from utils.io.yaml_handling import read_config
from utils.io.fits_handling import save_fits
from utils.sci_image_utils import parang_sort, diskprep_image_frames_parangs
from utils.improc_tools import fft_power_spectrum, subtract_median_profile_np
from dev.pyklip.klip import high_pass_filter
from dev.pyklip.instruments.Instrument import GenericData
from dev.pyklip.rdi import PSFLibrary
from dev.pyklip.fmlib.diskfm import DiskFM
import dev.pyklip.fm as fm
from utils.make_gpi_psf_for_disks import make_disk_mask
import utils.astro_unit_conversion as convert
from modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
# import multiprocessing as mp

def get_basedir():
    basedir = os.environ.get("DISKFIT_BASEDIR",f'{os.environ["HOME"]}/data')
    return basedir

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

        resultsdir = os.path.join(basedir, self.params_file["BAND_DIR"],
                                  "results_freeform")
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

    def _load_reference_model_params(self):
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

        self.disk_params = disk_params

    def render_reference_model(self):

        self._load_reference_model_params()

        beta = self.disk_params["beta"]
        a_r = self.disk_params["a_r"]
        inc = self.disk_params["inc"]
        pa = self.disk_params["pa"]
        dx = self.disk_params["dx"]
        dy = self.disk_params["dy"]

        R1 = self.disk_params['r1']
        R2 = self.disk_params['r2']

        Norm = self.disk_params['Norm']
        g1 = self.disk_params['g1']
        g2 = self.disk_params['g2']
        alpha1 = self.disk_params['alpha1']

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
        rc = self.disk_params['rc']
        m = self.disk_params['alpha_in']
        n = self.disk_params['alpha_out']
        model = fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                                    g1, g2, alpha1, a_r, rc, m, n,
                                    y_arr=y,
                                    z_arr=z,
                                    npts=n_pts,
                                    mask=(1 - self.mask2generatedisk))
        save_fits(os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel.fits"), model)
        return model

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
        psd_ref_model_win = window_opt #+ psd_ref_model
        # psd_ref_model_win_hp = window_opt + psd_ref_model_hp
        # ref_model_hp_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_HighPass.fits")
        psd_ref_model_win_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_PSD.fits")
        # psd_ref_model_win_hp_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_PSD_HighPass.fits")
        window_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_ReferenceModel_Window.fits")
        # save_fits(ref_model_hp_saveto, reference_model_hp_clamped)
        save_fits(psd_ref_model_win_saveto, psd_ref_model)
        # save_fits(psd_ref_model_win_hp_saveto, psd_ref_model_win_hp)
        save_fits(window_saveto, window_opt)
        return psd_ref_model_win, window_opt
    

    def get_initial_model(self, loc_init_model=None, random_seed=0):
        rng = np.random.default_rng(random_seed)

        #load in the model
        if loc_init_model is not None:
            model_init = fits.getdata(loc_init_model)
        else:
            # we init the model fitting with just a noise image
            model_init = rng.uniform(0.0, 1.0, self.image_shape)
        
        # model_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
        # save_fits(model_saveto, model_init)

        model_init *= self.mask2generatedisk

        return model_init
    
    def prep_dataset(self):
        
        filelist = sorted(glob.glob(f'{self.datadir}/camsci*.fits'),
                          key=parang_sort)
        if len(filelist) == 0:
            raise ValueError(f"Could not find files in the dir: {self.datadir}")
        input_data, par_angs  = diskprep_image_frames_parangs(filelist)
        self.frame_shape = input_data[0].shape
        input_centers = np.array([self.aligned_center for _ in range(len(filelist))])
        # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
        IWA = self.iwa
        dataset = GenericData(input_data,
                             input_centers,
                             parangs=par_angs,
                             IWA=IWA,filenames=filelist)

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
        mask_speckle_region = np.ones(image_size)
        x = np.arange(image_size[0], dtype=float)[None,:] - aligned_center[0]
        y = np.arange(image_size[1], dtype=float)[:,None] - aligned_center[1]
        rho2d = np.sqrt(x**2 + y**2)
        mask_speckle_region[np.where(rho2d < mask_speckles)] = 0.
        mask2generatedisk = mask2generatedisk*mask_speckle_region
        mask2generatedisk[np.where(mask2generatedisk < 0.5)] = 0
        mask2generatedisk[np.where(mask2generatedisk > 0.5)] = 1

        self.mask2generatedisk = mask2generatedisk
        self.mask4noisemap = mask4noisemap

        fits.writeto(f"{save_mask_part}_mask2generatedisk.fits",
                     mask2generatedisk, overwrite=True)
        
        fits.writeto(f"{save_mask_part}_mask4noisemap.fits",
                     mask4noisemap, overwrite=True)
        return mask2generatedisk

    def initialize_diskfm(self, dataset, model_init, psflib=None):

        model_convolved = convolve2d(model_init, self.psf, mode="same")
        model_init_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
        model_convolved_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel_Conv.fits")
        save_fits(model_init_saveto, model_init)
        save_fits(model_convolved_saveto, model_convolved)


        diskobj = DiskFM(dataset.input.shape,
                        self.numbasis,
                        dataset,
                        model_disk=np.asarray(model_init),
                        basis_filename=os.path.join(
                            self.klipdir, self.file_prefix + '_klbasis.h5'),
                        save_basis=True,
                        aligned_center=self.aligned_center)
        # nofm_obj = BasisOnly(dataset.input.shape,
        #                 np.atleast_1d([self.numbasis]),
        #                 # dataset,
        #                 # model_wdh_list=np.asarray(first_models),
        #                 # model_pas_mask=model_pas_mask,
        #                 basis_filename=os.path.join(
        #                     self.klipdir, self.file_prefix + '_klbasis.h5'),
        #                 save_basis=True,
        #                 # aligned_center=self.aligned_center,
        #                 )
        maxnumbasis = dataset.input.shape[0]
        time_start = datetime.now()
        fm.klip_dataset(dataset,
                        fm_class=diskobj,
                        # fm_class=nofm_obj,
                        numbasis=self.numbasis,
                        maxnumbasis=maxnumbasis,
                        annuli=self.annuli,
                        mode=self.mode,
                        subsections=1,
                        outputdir=self.klipdir,
                        fileprefix=self.file_prefix,
                        aligned_center=self.aligned_center,
                        # mute_progression=True,
                        highpass=self.hp,
                        minrot=self.move_here,
                        calibrate_flux=False,
                        # numthreads=mp.cpu_count(), #default: use all
                        time_collapse='median',
                        psf_library=psflib)
        
        print(f"klip_dataset() took {datetime.now() - time_start}.")

        path_rd = os.path.join(self.klipdir, f"{self.file_prefix}-klipped-KLmodes-all.fits")
        reduced_data = fits.getdata(path_rd)

        if self.clean_final_fm:
            reduced_data, _ = subtract_median_profile_np(reduced_data,
                                                    self.aligned_center)
            save_fits(path_rd, reduced_data)

        

        reduced_noise_masked = reduced_data * self.mask4noisemap

        reduced_disk_masked = reduced_data * self.mask2generatedisk

        # Make the noisemap now, to inspect after initialization in case changes need to occur
        noise_map = self.make_noise_map_rings(reduced_data_no_disk=reduced_noise_masked)
        self.noise_map = noise_map

        noise_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_noisemap.fits")
        fits.writeto(noise_saveto, noise_map, overwrite=True)

        saveto_masked_data = os.path.join(self.klipdir, f"{self.file_prefix}_masked_data.fits")
        saveto_masked_noise = os.path.join(self.klipdir, f"{self.file_prefix}_use4noisemap.fits")

        save_fits(saveto_masked_data, reduced_disk_masked)
        save_fits(saveto_masked_noise, reduced_noise_masked)

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
        image_center = (h / 2 - 0.5, w / 2 - 0.5)
        # print('Generating noise cube...')
        # nodisk_data[nodisk_data != nodisk_data] = 0
        # create rho2D for the rings
        x = np.arange(h, dtype=np.float64)[None, :] - image_center[0]
        y = np.arange(w, dtype=np.float64)[:, None] - image_center[1]
        rho2d = np.sqrt(x**2 + y**2)

        noise_map = np.zeros((h, w))
        for i_ring in range(0,
                            # int(np.floor(image_center[0] / delta_radii)) - 2):
                            int(self.params_file["OWA"] / delta_radii) - 2):
            wh_rings = (rho2d >= i_ring * delta_radii) & (rho2d < (i_ring + 1) * delta_radii)
            noise_map[wh_rings] = np.nanstd(reduced_data_no_disk[wh_rings])
        
        return noise_map

