import os
import re
import glob
from astropy.io import fits
import numpy as np
import jax.numpy as jnp

from modeling.jax_models.wdh_modeling import gen_multiwdh_image
from utils.io.yaml_handling import read_config
from utils.io.fits_handling import save_fits
from dev.pyklip.instruments.Instrument import GenericData
from utils.make_gpi_psf_for_disks import make_disk_mask
import utils.astro_unit_conversion as convert

class ParametricWDH:
    def __init__(self, config):
        self.params_file = read_config(config)
        self.params_init = self._get_initial_params()
        self._load_dirs()
        self._load_metadata()
        # self.klbasis = self._loadbasis()
    def _get_initial_params(self):
        # This gets applied to all models
        all_wdhs = {}
        all_wdhs["fwhm"] = self.params_file["wdh_model"]["fwhm_init"]

        # First WDH
        wdh1 = {}
        wdh1["beta"] = self.params_file["wdh_model"]["beta_init"]
        wdh1["a_r"] = self.params_file["wdh_model"]["a_r_init"]
        wdh1["sig"] = self.params_file["wdh_model"]["sig_init"]
        wdh1["PA"] = self.params_file["wdh_model"]["pa_init"]
        wdh1["dx"] = self.params_file["wdh_model"]["dx_init"]
        wdh1["Norm"] = self.params_file["wdh_model"]["Norm_init"]
        
        wdh2 = {}
        wdh2["beta"] = self.params_file["wdh_model"]["beta2_init"]
        wdh2["a_r"] = self.params_file["wdh_model"]["a_r2_init"]
        wdh2["sig"] = self.params_file["wdh_model"]["sig2_init"]
        wdh2["PA"] = self.params_file["wdh_model"]["pa2_init"]
        wdh2["dx"] = self.params_file["wdh_model"]["dx2_init"]
        wdh2["Norm"] = self.params_file["wdh_model"]["Norm2_init"]

        params_init = {"ps_global":all_wdhs,
                       "ps_indiv":[wdh1, wdh2]}
        
        return params_init
    
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
        aligned_center = self.params_file["ALIGNED_CENTER"]
        self.aligned_center = aligned_center
        self.image_size = round(aligned_center[0]) * 2
        self.mode = self.params_file["MODE"]
        self.wl = self.params_file["METADATA"]["WL"]
        self.diam = self.params_file["METADATA"]["PRIM_MIRROR_SZ"]
    
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
        dataset = GenericData(input_data,input_centers,parangs=par_angs,IWA=IWA,filenames=filelist)
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
        
        def gen_wdh_image_from_params(self, params):
            x = jnp.arange(self.image_size) - self.aligned_center[0]
            y = jnp.arange(self.image_size) - self.aligned_center[1]

            wdh_image = gen_multiwdh_image(x, y, params)

            return wdh_image
        
        def initialize_windfm(self):
            initial_wdh_image = gen_wdh_image_from_params(self.params_init)
            model_init_saveto = os.path.join(self.klipdir, f"{self.file_prefix}_FirstModel.fits")
            save_fits(model_init_saveto, initial_wdh_image)
            windobj = WindFM(dataset.input.shape,
                         numbasis,
                         dataset,
                         model_wdh1=model1_here_convolved,
                         model_wdh2=model2_here_convolved,
                         basis_filename=os.path.join(
                             klipdir, file_prefix + '_klbasis.h5'),
                         save_basis=True,
                         aligned_center=aligned_center)




            
        


def parang_sort(filename):
    """Extracts the float value between '2x2bin_' and '_parang'."""
    match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
    if match:
        return float(match.group(1))  # Convert extracted string to float
    return float('inf')  # Assign an arbitrary large value if no match is found

def control_region_mask(framesize, coronrad, seeinglimited):
    center_x, center_y = framesize[0] // 2, framesize[1] // 2
    mask_ctrl_reg = np.ones(framesize)
    x = np.arange(framesize[0], dtype=float)[None,:] - center_x
    y = np.arange(framesize[1], dtype=float)[:,None] - center_y
    rho2d = np.sqrt(x**2 + y**2)
    mask_ctrl_reg[np.where(rho2d > seeinglimited)] = 0.
    # mask_ctrl_reg[int(ALIGNED_CENTER[0] - owa / 2):int(ALIGNED_CENTER[0] + owa / 2),int(ALIGNED_CENTER[1] - owa / 2):int(ALIGNED_CENTER[1] + owa / 2)] = 0.
    # mask_ctrl_reg[mask_ctrl_reg == 0.] = np.nan
    mask_ctrl_reg[np.where(rho2d < coronrad)] = 0.
    # mask_ctrl_reg = 1 - mask_ctrl_reg
    # mask_ctrl_reg = rotate(mask_ctrl_reg,angle=-62, reshape=False, order=0)
    # mask_ctrl_reg = 1 - mask_ctrl_reg
    mask_ctrl_reg[mask_ctrl_reg > 0.5] = 1
    mask_ctrl_reg[mask_ctrl_reg < 0.5] = 0
    return mask_ctrl_reg
