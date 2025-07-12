import os
import sys
import glob
from astropy.io import fits
import numpy as np
import jax.numpy as jnp

from modeling.jax_models.wdh_modeling import gen_multiwdh_image
from utils.io.yaml_handling import read_config
from utils.io.fits_handling import save_fits
from utils.masks import control_region_mask
from utils.sci_image_utils import parang_sort, prep_image_frames_parangs
from dev.pyklip.instruments.Wind import GenericWDH
from dev.pyklip.fmlib.windfm import WindFM
import dev.pyklip.fm as fm
from utils.make_gpi_psf_for_disks import make_disk_mask
import utils.astro_unit_conversion as convert

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

        self.datadir = os.path.join(basedir, self.params_file["BAND_DIR"])

        self.file_prefix = self.params_file["FILE_PREFIX"]

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
        aligned_center = self.params_file["ALIGNED_CENTER"]
        self.aligned_center = aligned_center
        self.image_size = np.ceil(aligned_center[0]) * 2
    
    def prep_psflib_and_mask(self):
        # Make the mask
        x_off = self.params_file["MASK"]["DX"]
        y_off = self.params_file["MASK"]["DY"]
        aligned_center = self.aligned_center
        mask_center = aligned_center[0] + x_off, aligned_center[1] + y_off
        filelist = sorted(glob.glob(f'{self.datadir}/*parang*.fits'),
                          key=parang_sort)
        if len(filelist) == 0:
            raise ValueError(f"Could not find files in the dir: {self.datadir}")
        input_data, par_angs, wdh1_angs, wdh2_angs = prep_image_frames_parangs(filelist,
                                                                               path_wind_parquet,
                                                                                   )
        frames = []
        # refs = []
        derot_angs = []
        for name in filelist:
            dat_unit, hdr_unit = fits.getdata(name,header=True)
            derot_angs.append(hdr_unit['PARANG'])
            frames.append(dat_unit)
        input_data = np.asarray(frames)
        par_angs = np.asarray(derot_angs)
        input_centers = np.array([aligned_center for _ in range(len(filelist))])
        # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
        IWA = self.params_file['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23
        dataset = GenericWDH(input_data,
                                 input_centers,
                                 obj_parangs=par_angs,
                                 wdh1_parangs=wdh1_angs,
                                 wdh2_parangs=wdh2_angs,
                                 IWA=IWA,filenames=filelist)
        dataset.OWA = self.params_file["OWA"]
        if dataset.input.shape[1] != dataset.input.shape[2]:
            raise ValueError(""" Data slices are not square (dimx!=dimy), 
                            please make them square""")

        # mask2generatedisk = 1 - mask_disk_zeros
        lyot_sm_rad = 3 #lambda / D
        ctrl_rad = 24 #lambda / D

        apdiam = self.diam
        reselem = self.wl * 1e-6 / apdiam * 180 / np.pi * 3600 #arcseconds
        reselem_pix = reselem / self.pixscale #num pixels, int
        coron_reg = lyot_sm_rad * reselem_pix
        seeing_limited = ctrl_rad * reselem_pix * np.sqrt(2)
        mask2generatehalo = control_region_mask(dataset.input.shape[1:],
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
            dataset.input.shape[1],
            pa_init,
            inc_init,
            convert.au_to_pix(self.params_init["disk_model"]['r1_init'],
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

        ### a few lines to create a circular central mask to hide center regions with a lot
        ### of speckles. Currently not using it but it's there
        mask_coron_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
        mask_speckle_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
        x = np.arange(dataset.input.shape[1], dtype=float)[None,:] - aligned_center[0]
        y = np.arange(dataset.input.shape[2], dtype=float)[:,None] - aligned_center[1]
        rho2d = np.sqrt(x**2 + y**2)
        mask_coron_region[np.where(rho2d < coron_reg)] = 0.
        mask_speckle_region[np.where(rho2d < mask_speckles)] = 0.
        mask2minimize = mask2minimize*mask_speckle_region
        mask2minimize[np.where(mask2minimize < 0.5)] = 0
        mask2minimize[np.where(mask2minimize > 0.5)] = 1

        fits.writeto(f'{save_mask_part}_mask2minimize.fits',
                     mask2minimize,
                     overwrite='True')
        

        def initialize_windfm(self):
            x = np.arange(self.image_size) - self.aligned_center[0]
            y = np.arange(self.image_size) - self.aligned_center[1]

            xx, yy = np.meshgrid(x, y)
            initial_wdh_images = gen_multiwdh_image(xx, yy, self.params_init)
            wdh_image_init = np.asarray(jnp.sum(initial_wdh_images, axis=0))
            wdh_images_np = np.asarray(initial_wdh_images)
            model_init_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
            save_fits(model_init_saveto, wdh_image_init)
            windobj = WindFM(dataset.input.shape,
                         self.numbasis,
                         self.dataset,
                         model_wdh_list=wdh_images_np,
                         basis_filename=os.path.join(
                             self.klipdir, self.file_prefix + '_klbasis.h5'),
                         save_basis=True,
                         aligned_center=aligned_center)
            maxnumbasis = dataset.input.shape[0]
            fm.klip_dataset(dataset,
                            windobj,
                            numbasis=self.numbasis,
                            maxnumbasis=maxnumbasis,
                            annuli=self.annuli,
                            mode=self.mode,
                            subsections=1,
                            outputdir=self.klipdir,
                            fileprefix=self.file_prefix,
                            aligned_center=aligned_center,
                            mute_progression=True,
                            highpass=False,
                            minrot=self.move_here,
                            calibrate_flux=False,
                            numthreads=1,
                            time_collapse='median',
                            psf_library=None)

            sys.stdout = sys.__stdout__
            reduced_data = fits.getdata(os.path.join(self.klipdir,
                                                    self.file_prefix + '-klipped-KLmodes-all.fits'))[0]
            self.reduced_data = reduced_data
