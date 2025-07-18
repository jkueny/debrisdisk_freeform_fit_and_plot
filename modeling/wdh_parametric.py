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
import jax.numpy as jnp
from datetime import datetime

from utils.io.yaml_handling import read_config
from utils.io.fits_handling import save_fits
from utils.masks import control_region_mask
from utils.sci_image_utils import parang_sort, prep_image_frames_parangs
from dev.pyklip.instruments.Wind import GenericWDH
from dev.pyklip.fmlib.windfm import WindFM
from dev.pyklip.fmlib.nofm import NoFM, BasisOnly
import dev.pyklip.fm as fm
from utils.make_gpi_psf_for_disks import make_disk_mask
import utils.astro_unit_conversion as convert
# import multiprocessing as mp

class ParametricWDH:
    def __init__(self, config):
        self.params_file = read_config(config)
        self.params_init = self._get_initial_params()
        self._load_dirs()
        self._load_metadata()
        self._load_klparams()
        # self.klbasis = self._loadbasis()
    def _get_initial_params(self):
        # Clamp the number of wind layers to between 1 and 3
        n_wdhs = int(self.params_file["wdh_model"]["N_WIND_LAYERS"])
        self.n_models = n_wdhs
        if n_wdhs < 1 or n_wdhs > 3:
            raise ValueError("N_WIND_LAYERS must be between 1 and 3.")

        # Shared parameters for all WDHs
        all_wdhs = {
            "fwhm": self.params_file["wdh_model"]["fwhm_init"]
        }

        # Create each WDH component's dictionary dynamically
        ps_indiv = []
        for i in range(1, n_wdhs + 1):
            suffix = "" if i == 1 else str(i)
            wdh = {
                "beta": self.params_file["wdh_model"][f"beta{suffix}_init"],
                "a_r": self.params_file["wdh_model"][f"a_r{suffix}_init"],
                "sig": self.params_file["wdh_model"][f"sig{suffix}_init"],
                "PA": self.params_file["wdh_model"][f"pa{suffix}_init"],
                "dx": self.params_file["wdh_model"][f"dx{suffix}_init"],
                "Norm": self.params_file["wdh_model"][f"Norm{suffix}_init"]
            }
            ps_indiv.append(wdh)

        return {
            "ps_global": all_wdhs,
            "ps_indiv": ps_indiv
        }               
    
    def _load_dirs(self):
        basedir = f'{os.environ["HOME"]}/projects'
        self.basedir = basedir
        klipdir = os.path.join(basedir, self.params_file["BAND_DIR"],
                                    "wind_fm_files")
        
        os.makedirs(klipdir, exist_ok=True)
        self.klipdir = klipdir

        resultsdir = os.path.join(basedir, self.params_file["BAND_DIR"],
                                  "results_wdh")
        os.makedirs(resultsdir, exist_ok=True)
        self.resultsdir = resultsdir

        self.datadir = os.path.join(basedir, self.params_file["BAND_DIR"])

        self.file_prefix = self.params_file["FILE_PREFIX"]
        self.path_wind_parquet = os.path.join(klipdir, self.params_file["WIND_LOOKUP_PREFIX"])

    def _load_metadata(self):
        self.pixscale = self.params_file["METADATA"]["PIXSCALE_INS"]
        self.distance = self.params_file["METADATA"]["DISTANCE_STAR"]
        self.wl = self.params_file["METADATA"]["WL"]
        self.diam = self.params_file["METADATA"]["PRIM_MIRR_SZ"]

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
        self.image_size = np.ceil(aligned_center[0]) * 2
    
    def prep_dataset(self):
        
        filelist = sorted(glob.glob(f'{self.datadir}/*parang*.fits'),
                          key=parang_sort)
        if len(filelist) == 0:
            raise ValueError(f"Could not find files in the dir: {self.datadir}")
        input_data, par_angs, wdh_angs, model_pa_mask = prep_image_frames_parangs(filelist,
                                                                               self.path_wind_parquet,
                                                                               self.n_models)
        self.PAmask = model_pa_mask
        frames = []
        # refs = []
        derot_angs = []
        for name in filelist:
            dat_unit, hdr_unit = fits.getdata(name,header=True)
            derot_angs.append(hdr_unit['PARANG'])
            frames.append(dat_unit)
        input_data = np.asarray(frames)
        self.frame_shape = input_data[0].shape
        par_angs = np.asarray(derot_angs)
        input_centers = np.array([self.aligned_center for _ in range(len(filelist))])
        # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
        IWA = self.params_file['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23
        dataset = GenericWDH(input_data,
                             input_centers,
                             obj_parangs=par_angs,
                             wdh_parangs=wdh_angs,
                             IWA=IWA,filenames=filelist)

        dataset.OWA = self.params_file["OWA"]
        if dataset.input.shape[1] != dataset.input.shape[2]:
            raise ValueError(""" Data slices are not square (dimx!=dimy), 
                            please make them square""")
        
        
        return dataset
    
    def prep_binary_masks(self):
        # mask2generatedisk = 1 - mask_disk_zeros
        # Make the mask
        x_off = self.params_file["MASK"]["DX"]
        y_off = self.params_file["MASK"]["DY"]
        aligned_center = self.aligned_center
        image_size = (round(aligned_center[0]) * 2, round(aligned_center[1]) * 2)
        mask_center = aligned_center[0] + x_off, aligned_center[1] + y_off
        lyot_sm_rad = 3 #lambda / D
        ctrl_rad = 24 #lambda / D

        apdiam = self.diam
        reselem = self.wl * 1e-6 / apdiam * 180 / np.pi * 3600 #arcseconds
        reselem_pix = reselem / self.pixscale #num pixels, int
        coron_reg = lyot_sm_rad * reselem_pix
        seeing_limited = ctrl_rad * reselem_pix * np.sqrt(2)
        mask2generatehalo = control_region_mask(image_size,
                                                coronrad=coron_reg,
                                                seeinglimited=seeing_limited,
                                                )
        save_mask_part = os.path.join(self.klipdir,
                                    f"{self.file_prefix}")
        fits.writeto(f"{save_mask_part}_mask2generatehalo.fits",
                     mask2generatehalo, overwrite=True)

        in_scaling = self.params_file["MASK"]["IN_SCALING"]
        out_scaling = self.params_file["MASK"]["OUT_SCALING"]
        mask_speckles = self.params_file["MASK"]["SPECKLES"]
        inc_init = self.params_file["disk_model"]['inc_init']
        pa_init = self.params_file["disk_model"]['pa_init']
        mask_disk_zeros = make_disk_mask(
            image_size[0],
            pa_init,
            inc_init,
            convert.au_to_pix(self.params_file["disk_model"]['r1_init'],
                              self.pixscale,
                              self.distance) -
            in_scaling / np.cos(np.radians(inc_init)),
            convert.au_to_pix(self.params_file["disk_model"]['r2_init'],
                              self.pixscale,
                              self.distance) +
            out_scaling / np.cos(np.radians(inc_init)),
            aligned_center=mask_center)
        # mask2minimize = (1 - mask_disk_zeros)
        mask2minimize = mask_disk_zeros * mask2generatehalo

        self.mask2minimize = mask2minimize

        ### a few lines to create a circular central mask to hide center regions with a lot
        ### of speckles. Currently not using it but it's there
        mask_coron_region = np.ones(image_size)
        mask_speckle_region = np.ones(image_size)
        x = np.arange(image_size[0], dtype=float)[None,:] - aligned_center[0]
        y = np.arange(image_size[1], dtype=float)[:,None] - aligned_center[1]
        rho2d = np.sqrt(x**2 + y**2)
        mask_coron_region[np.where(rho2d < coron_reg)] = 0.
        mask_speckle_region[np.where(rho2d < mask_speckles)] = 0.
        mask2minimize = mask2minimize*mask_speckle_region
        mask2minimize[np.where(mask2minimize < 0.5)] = 0
        mask2minimize[np.where(mask2minimize > 0.5)] = 1

        fits.writeto(f'{save_mask_part}_mask2minimize.fits',
                     mask2minimize,
                     overwrite='True')
        return mask2generatehalo, mask2minimize

    def initialize_windfm(self, dataset, first_models):

        wdh_model_here = np.asarray(jnp.sum(first_models, axis=0))
        model_pas_mask = self.PAmask
        model_init_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
        save_fits(model_init_saveto, wdh_model_here)
        windobj = WindFM(dataset.input.shape,
                        self.numbasis,
                        dataset,
                        model_wdh_list=np.asarray(first_models),
                        model_pas_mask=model_pas_mask,
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
                        fm_class=windobj,
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
                        highpass=False,
                        minrot=self.move_here,
                        calibrate_flux=False,
                        # numthreads=mp.cpu_count(), #default: use all
                        time_collapse='median',
                        psf_library=None)
        
        print(f"klip_dataset() took {datetime.now() - time_start}.")

        path_rd = os.path.join(self.klipdir, f"{self.file_prefix}-klipped-KLmodes-all.fits")
        reduced_data = fits.getdata(path_rd)

        reduced_data_masked = reduced_data * self.mask2minimize
        
        saveto_masked = os.path.join(self.klipdir, f"{self.file_prefix}_masked_data.fits")

        save_fits(saveto_masked, reduced_data_masked)

        model_fm_init = self.single_fm(np.asarray(first_models))

        model_fm_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel_FM.fits")

        save_fits(model_fm_saveto, model_fm_init)

        sys.stdout = sys.__stdout__
        # reduced_data = fits.getdata(os.path.join(self.klipdir,
        #                                         self.file_prefix + '-klipped-KLmodes-all.fits'))[0]
        # self.reduced_data = reduced_data

    def single_fm(self, wdh_models):
        # Refresh the windFM object
        basis_file_path = os.path.join(self.klipdir, f"{self.file_prefix}_klbasis.h5")

        windfm = WindFM(inputs_shape=None,
                            numbasis=None,
                            dataset=None,
                            model_wdh_list=wdh_models,
                            model_pas_mask=None,
                            basis_filename=basis_file_path,
                            load_from_basis=True)
        
        windfm.update_wind(wdh_models)

        model_fm = windfm.fm_parallelized()[0]



        return model_fm

