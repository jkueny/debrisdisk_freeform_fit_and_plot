import os
import sys
# import copy
import argparse


basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_smlyot_20230309_10.yaml'  # name of the parameter file
# you can also call it with the python function argument -p


import glob
import re



# # because this error was coming up
os.environ['OPENBLAS_NUM_THREADS'] = '1'


from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import yaml


from dev.pyklip.instruments.Instrument import GenericData

from dev.pyklip.fmlib.jax_diskfm import JDFM
from dev.pyklip.fmlib.funcs_JDFM import update_disk
from dev.pyklip.fmlib.diskfm import DiskFM
import dev.pyklip.fm as fm
from dev.pyklip.j_klip import rotate_image


import utils.make_gpi_psf_for_disks as gpidiskpsf
import utils.astro_unit_conversion as convert
from utils.klip_basis import load_kl_basis, unpack_basis_data


import jax
import jax.numpy as jnp
import jax.profiler
from jax.scipy.signal import convolve2d

update_disk_jit = jax.jit(update_disk, static_argnames=["aligned_center",
                                                        "min_num_models",])

#### JDFM funcs section for debugging ####
def calculate_fm(delta_KL, original_KL, sci, model_sci):
    """
    Same function as calculate_fm() but faster when numbasis has only one element. It doesn't do the mutliplication with
    the triangular matrix.

    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (\Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
    the spectral dependence has already been folded into delta_KL_nospec (treat it as delta_KL).

    Note: if inputflux is None and delta_KL_nospec has three dimensions (ie delta_KL_nospec was calculated using
    pertrurb_nospec() or perturb_nospec_modelsBased()) then only klipped_oversub and klipped_selfsub are returned.
    Besides they will have an extra first spectral dimension.

    Args:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix). If inputflux is None, delta_KL_nospec = delta_KL
        orignal_KL: unpertrubed KL modes (array of size [numbasis, numpix])
        numbasis: array of (ONE ELEMENT ONLY) KL mode cutoffs
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        sci: array of size p representing the science data
        model_sci: array of size p corresponding to the PSF of the science frame
        input_spectrum: array of size wv with the assumed spectrum of the model

    If delta_KL_nospec does NOT include a spectral dimension or if inputflux is not None:
    Returns:
        fm_psf: array of shape (b,p) showing the forward modelled PSF
                Skipped if inputflux = None, and delta_KL_nospec has 3 dimensions.
        klipped_oversub: array of shape (b, p) showing the effect of oversubtraction as a function of KL modes
        klipped_selfsub: array of shape (b, p) showing the effect of selfsubtraction as a function of KL modes
        Note: psf_FM = model_sci - klipped_oversub - klipped_selfsub to get the FM psf as a function of K Lmodes
              (shape of b,p)

    If inputflux = None and if delta_KL_nospec include a spectral dimension:
    Returns:
        klipped_oversub: Sum(<S|KL>KL) with klipped_oversub.shape = (size(numbasis),Npix)
        klipped_selfsub: Sum(<N|DKL>KL) + Sum(<N|KL>DKL) with klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)

    """
    max_basis = original_KL.shape[0]

    N_pix = np.size(sci)

    # remove means and nans from science image
    sci_mean_sub = sci - jnp.nanmean(sci)
    sci_meansub_nonan = jnp.nan_to_num(sci_mean_sub, nan=0.0)
    sci_mean_sub_rows = jnp.reshape(sci_meansub_nonan,(1,N_pix))


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci # should be subtracting off the mean?

    model_sci_meansub_nonans = jnp.nan_to_num(model_sci_mean_sub, nan=0.0)
    model_sci_mean_sub_rows = np.reshape(model_sci_meansub_nonans,(1,N_pix))


    # Forward model the PSF
    # 3 terms: 1 for oversubtracton (planet attenauted by speckle KL modes),
    # and 2 terms for self subtraction (planet signal leaks in KL modes which get projected onto speckles)
    #
    # Klipped = N-Sum(<N|KL>KL) + S-Sum(<S|KL>KL) - Sum(<N|DKL>KL) - Sum(<N|KL>DKL)
    # With  N = noise/speckles (science image)
    #       S = signal/planet model
    #       KL = KL modes
    #       DKL = perturbation of the KL modes/Delta_KL
    #

    oversubtraction_inner_products = jnp.dot(model_sci_mean_sub_rows, original_KL.T)
    selfsubtraction_1_inner_products = jnp.dot(sci_mean_sub_rows, delta_KL.T)
    # selfsubtraction_1_inner_products.shape = (max_basis,N_pix,max_basis)
    
    selfsubtraction_2_inner_products = jnp.dot(sci_mean_sub_rows, original_KL.T)


    thresh_oversub_inner_products = oversubtraction_inner_products.at[max_basis::].set(0)
    klipped_oversub = jnp.dot(thresh_oversub_inner_products, original_KL)

    thresh_selfsubtraction_1_inner_products = selfsubtraction_1_inner_products.at[0,max_basis::].set(0)

    thresh_selfsubtraction_2_inner_products = selfsubtraction_2_inner_products.at[0,max_basis::].set(0)
    klipped_selfsub = jnp.dot(thresh_selfsubtraction_1_inner_products, original_KL) + \
                        jnp.dot(thresh_selfsubtraction_2_inner_products, delta_KL)


    return model_sci[None,:] - klipped_oversub - klipped_selfsub, klipped_oversub, klipped_selfsub

# @jax.jit
def perturb_KLmodes(evals, evecs, original_KL, refs, models_ref):
    """
    Perturb the KL modes using a model of the PSF but with the spectrum included in the model. Quicker than the others

    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [pixels, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, pixels])
        refs: N_images x pixels array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images.
                    Each model should contain spectral informatoin
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes. Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    N_ref = refs.shape[0]
    # N_pix = original_KL.shape[1]

    refs_mean_sub = refs - jnp.nanmean(refs, axis=1, keepdims=True)

    refs_meansub_nonan = jnp.nan_to_num(refs_mean_sub, nan=0.0) 

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    # models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
    models_meansub_nonan = jnp.nan_to_num(models_mean_sub, nan=0.0)

    #print(evals.shape,evecs.shape,original_KL.shape,refs.shape,models_ref.shape)

    evals_tiled = jnp.tile(evals,(max_basis,1))
    evals_nan_diag = jnp.fill_diagonal(evals_tiled, jnp.nan, inplace=False)
    # print(evals_tiled)
    # sys.exit()
    evals_sqrt = jnp.sqrt(evals)
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_nan_diag.transpose()- evals_nan_diag)
    #print(evals)
    beta_tmp = beta_tmp.at[np.diag_indices(np.size(evals))].set(-0.5/evals)
    beta = evals_ratio*beta_tmp #no NaNs confirmed JKK 03/18/2025

    C_partial = models_meansub_nonan.dot(refs_meansub_nonan.transpose())
    C = C_partial+C_partial.transpose()
    #C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha_tmp = jnp.dot(evecs.transpose(), C)
    alpha = jnp.dot(alpha_tmp, evecs)

    delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)


    return delta_KL

def fm_from_eigen_single(sci_data, refs_data, model_disk_sci, model_disk_refs,
                         klmodes, evals, evecs, parang):
    """ 
    Compute the forward model for one disk model image.

    Note:
        - All inputs must be JAX arrays, tuples, or scalars with fixed shapes.
        - Any scalar or static parameters that are the same for every section
        can be passed as-is.

    Args:
        sci_data (JAX array): A single science PSF image in the sequence.
        Flattened, to a shape (1, N_pixels)

        refs_data (JAX array): The other PSF images in the sequence used as
        reference images for the PSF subtraction of the sci_data.
        Flattened, to a shape (N_refs, N_pixels).

        model_disk_sci (JAX array): A single disk model at the PA corresponding
        to the working sci_data image. Flattened to a shape (1, N_pixels).

        model_disk_refs (JAX array): Sequence of model disk images rotated to
        the PAs consistent with the refs_data set.
        Flattened to (N_refs, N_pixels).

        klmodes (JAX array): Basis vectors used for the PSF subtraction of the
        working science image, sci_data. Shape (N_modes, N_pixels)
        evals (_type_): _description_
        evecs (_type_): _description_
        parang (_type_): _description_
        IOWA (_type_): _description_
        aligned_center (_type_): _description_
        numbasis (_type_): _description_

    Returns:
        _type_: _description_
    """


    # Compute delta_KL (set to zero if mode=='RDI')
    # Ex. shape for delta_KL (2, 39112)
    delta_KL = perturb_KLmodes(evals, evecs, klmodes,
                                    refs_data, model_disk_refs,
                                    # return_perturb_covar=False,
                                    )
    # Calculate the post-KLIP PSF using your forward modeling routine.
    postklip_psf, _, _ = calculate_fm(delta_KL, klmodes,
                                      sci_data, model_disk_sci)
    
    # postklip_psf_squeezed = jnp.squeeze(postklip_psf)

    # jax.debug.print("print(postklip_psf.shape) -> {x}", x=postklip_psf.shape)

    postklip_psf_flipx = jnp.flip(postklip_psf, axis=1)
    # Save the rotated section.
    # derotated_output = rotate_image(postklip_psf_corrected,
    #                                 -parang,
    #                                 # flip_x=False
    #                                 )
    return postklip_psf_flipx

def fm_jaxed(aligned_images, model_disks, ref_models_stacked,
             ref_psfs_stacked, klmodes_stacked,
             evals_arr, evecs_stacked, 
            #  section_ind_arr, input_img_nums, input_img_shape,
             PAs):
    """Do the forward modeling procedure using JAX's vmap() framework.


    TODO Do we need the section indices array anymore? Right now,
    I don't think that we do. Investigate. It's been deleted.
    Also look at input_img_nums, numbasis, input_image_shape...


    Args:
        model_disks (JAX array): Flattened disk model images.
        Shape: (N_images, image_height * image_width)
        
        aligned_imgs (JAX array): Image data set, flattened.
        Shape: (N_images, image_height * image_width)
        
        klmodes_arr (dict): Karhunen-Loeve modes made from the
        science PSF images. Nested dict.
        
        evals_arr (dict): Eigenvalues from the KL-transform. Nested dict.
        
        evecs_arr (dict): Eigenvectors from the KL transform. Nested dict.
        section_ind_arr (dict): Array indices corresponding to the region
        of interest in every image. Nested dict.
        
        PAs (JAX array): Position angles of the disk in each of the aligned
        images. Shape (N_images)
        
        input_img_nums (dict): Values are the literal integer number in the 
        image sequence. Keys are the section IDs explained above. This could
        be used to index the nested dictionaries?
        
        input_img_shape (tuple): Shape of each unflattened input image in
        aligned_imgs. Ex. (224,224)
        
        IOWA (tuple): Inner and outer-working angle in pixels for each
        image in aligned_imgs. Basically inner- and outer- radius of
        the circular software mask w.r.t. the image center. Ex. (10,112)
        
        aligned_center (tuple): Center of each image in aligned_imgs. Should
        be the same for all images.
        
        numbasis (int): Number of KL modes for each image in aligned_imgs.
        Corresponds to the rows dimension of each set of KL modes since they
        are passed in flattened.

    Returns:
        _type_: _description_
    """    
    # Use vmap to vectorize fm_single over the first axis of all per-section arrays.
    # In this example, we assume that all per-section arrays have their first dimension equal to the number of sections.
    # For input_img_nums and PAs, ensure they are passed appropriately (e.g., one value per section).
    # Here, we set in_axes=0 for each per-section parameter.

    # Define a partial function that fixes the static parameters.
    # fm_single = partial(fm_from_eigen_single,
    #                     aligned_center=aligned_center,
    #                     )

    # fm_outputs = jax.vmap(fm_from_eigen_single,
    #                       in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    #                       )(aligned_images, ref_psfs_stacked,
    #                         model_disks,ref_models_stacked,
    #                         klmodes_stacked, evals_arr, evecs_stacked,
    #                         PAs)
    flat_postklip_PSFs = jax.vmap(fm_from_eigen_single
                          )(aligned_images, ref_psfs_stacked,
                            model_disks,ref_models_stacked,
                            klmodes_stacked, evals_arr, evecs_stacked,
                            PAs)
    # delta_KLs look reasonable JKK 03/18/2025
    # delta_KL_test1 = insert_section_into_full_image(delta_KLs[0][0], (224,224),
    #                                                 section_inds)
    # delta_KL_test2 = insert_section_into_full_image(delta_KLs[0][1], (224,224),
    #                                                 section_inds)
    # postklip_psf_test = insert_section_into_full_image(postklip_PSFs[0] , (224,224),
    #                                                 section_inds)
    # plt.imshow(np.asarray(postklip_psf_test),cmap="plasma")
    # plt.show()
    # exit()
    # jax.debug.print("print(fm_outputs.shape) -> {x}", x=fm_outputs.shape)
    # fm_outputs = fm_from_eigen_single(aligned_images, ref_psfs_stacked,
    #                         model_disks,ref_models_stacked,
    #                         klmodes_stacked, evals_arr, evecs_stacked,
    #                         PAs)
    # fm_outputs_squeezed = jnp.squeeze(fm_outputs)
    # fm_out = jnp.nanmean(fm_outputs_squeezed,axis=0)

    # jax.debug.print("print(fm_out.shape; after nanmean) -> {x}", x=fm_out.shape)

    return flat_postklip_PSFs

####

def mass_derotation(flat_postklip_psfs, PAs, image_dim, section_inds):
    print(f"image_dim -> {image_dim}")
    squeezed_postklip_psfs = jnp.squeeze(flat_postklip_psfs)
    postklip_psf_images = jax.vmap(insert_section_into_full_image,
                                   in_axes=(0, None, None))(squeezed_postklip_psfs,
                                                                   image_dim,
                                                                   section_inds)
    print(f"postklip_psf_images.shape -> {postklip_psf_images.shape}")
    derotated_postklip_psfs = jax.vmap(rotate_image)(postklip_psf_images,
                                                     PAs)
    corrected_postklip_psfs = jnp.flip(derotated_postklip_psfs, axis=1)
    return corrected_postklip_psfs

def initialize_freeform_model_reduced():
    """Initialize freeform model parameters for the unmasked region."""
    rng = jax.random.PRNGKey(42)
    # Instead of full image dimensions, only initialize num_free parameters.
    free_params = jax.random.normal(rng, (NUM_FREE,))

    return free_params

def reconstruct_full_image(free_params, full_shape):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    total_pixels = np.prod(full_shape)
    full_flat = jnp.zeros(total_pixels)
    full_flat = full_flat.at[MASK_INDICES].set(free_params)
    return full_flat.reshape(full_shape)

def parang_sort(filename):
        """Extracts the float value between '2x2bin_' and '_parang'."""
        match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
        if match:
            return float(match.group(1))  # Convert extracted string to float
        return float('inf')  # Assign an arbitrary large value if no match is found

def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"
    kernel_size = psf.shape[0] #should be square
    # Pad image to maintain size
    # padded_image = jnp.pad(input_model, [(kernel_size//2, kernel_size//2),
    #                                (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply convoluted convolution
    model_convolved = convolve2d(input_model, psf, mode="same")
    
    return model_convolved

def insert_section_into_full_image(flat_section, full_shape, section_inds):
    """
    Given:
      - flat_section: a 1D JAX array of length N_pixels_section (the forward-modeled section)
      - full_shape: tuple (height, width) for the full image (e.g. (224, 224))
      - section_inds: a JAX array or tuple of indices that select the section in the full image.
                     In your case, section_inds has shape (1, N_pixels_section), so we'll flatten it.
    
    This function flattens a blank full image, updates it at the given 1D indices with the values from flat_section,
    and then reshapes it back to full_shape.
    
    Returns:
      A full image (JAX array) of shape full_shape with the section inserted.
    """
    # Ensure section_inds is a 1D index array.
    section_inds = jnp.ravel(jnp.array(section_inds))
    # Create a blank full image flattened.
    total_pixels = np.prod(full_shape) # Ex. 50176
    full_image_flat = jnp.zeros(total_pixels)
    # Use .at to update the flattened array.
    full_image_flat = full_image_flat.at[section_inds].set(flat_section)
    # Reshape back to full_shape.
    full_image = full_image_flat.reshape(full_shape)
    return full_image

def do_single_fm(mod_pix_params, disk_image, psf, aligned_images,
                  ref_psfs_stacked, PAs, ref_PAs, fixed_refs, aligned_center,
                  section_inds_arr, klmodes_stacked, evals, evecs_stacked, 
                  image_dim):
    # Model reconstruction confirmed JKK 03/18/2025
    full_model_image = reconstruct_full_image(mod_pix_params, image_dim)
    # freeform_model = jax.nn.sigmoid(mod_pix_params)
    # freeform_model = jax.nn.sigmoid(full_model_image)

    freeform_image = convolve_model(full_model_image, psf)

    # confirmed image insertion works on global_models_prepped, individuals show as
    # expected.
    # confirmed ref_models_stacked as well. Zero pad images are at the end of the item
    # arrays. JKK 03/18/2025
    global_models_prepped, ref_models_stacked = update_disk_jit(model_disk=freeform_image,
                                                            PAs=PAs, ref_PAs=ref_PAs,
                                                            aligned_center=aligned_center,
                                                            section_inds=section_inds_arr,
                                                            min_num_models=fixed_refs,
                                                            )
    # # DEBUG: Grab a single disk model and view it
    # single_test_rotated = insert_section_into_full_image(ref_models_stacked[2][30], freeform_image.shape,
    #                                                      section_inds_arr)
    # plt.imshow(np.asarray(single_test_rotated), origin="lower")
    # plt.show()
    # sys.exit()
    # confirmed shape of model_images_prepped (84, 50176)
    flat_postklip_psfs = fm_jaxed(aligned_images,
                           global_models_prepped,
                           ref_models_stacked, ref_psfs_stacked,
                           klmodes_stacked, evals, evecs_stacked,
                           PAs)
    derotated_postklip_psfs = mass_derotation(flat_postklip_psfs,PAs,image_dim,section_inds)

    freeform_fm_full = np.nanmean(derotated_postklip_psfs, axis=0)

    print(f"freeform_fm_full.shape -> {freeform_fm_full.shape}")

    return freeform_fm_full


def prep_and_return_fm(target_image, model_init, psf, basis_data,):
    dimension = target_image.shape
    jax_target_image = jnp.array(target_image)

    # Initialize the initial image of random pixels
    image_params = model_init
    # image_params = initialize_freeform_model_reduced()



    basis_data_unpacked = unpack_basis_data(basis_data)

    # num_input_images = int(jax.device_get(basis_data["klparam_dict"]["nfiles"]))
    aligned_image_data = jnp.array(basis_data_unpacked["aligned_images"]) #shape ex. (84, 50176)
    global section_inds
    section_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)
    klmodes = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 2, 50176)
    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    # the eigenvectors have been zero-padded at the ends to removed ragged-ness....
    evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes)
    # evecs have been unpacked, stacked, and ready to be BATCHED!
    # input_img_nums = basis_data_unpacked["input_img_nums"]
    # These are the images used for the basis for every image in the dataset.
    ref_psfs = basis_data_unpacked["ref_psfs"] # zero-padded at the end to all have the same shape
    ref_PAs = basis_data_unpacked["ref_PAs"]
    fixed_refs = basis_data_unpacked["fixed_refs"]
    # ref_psfs shape (N_images, max_N_refs, N_pixels)
    # ref_psfs have been unpacked, stacked, and ready to be BATCHED!
    # position_angles = tuple(np.asarray(jax.device_get(basis_data["klparam_dict"]["PAs"])))
    position_angles = jnp.array((basis_data["klparam_dict"]["PAs"]))
    aligned_center = tuple(np.asarray(jax.device_get([basis_data["klparam_dict"]["aligned_center_x"],
                                basis_data["klparam_dict"]["aligned_center_y"]])))

    fm_out = do_single_fm(image_params, jax_target_image, psf,
                 aligned_image_data, ref_psfs,
                 position_angles, ref_PAs, fixed_refs,
                 aligned_center, section_inds,
                 klmodes, evals, evecs, dimension)
    
    return fm_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run diskFM MCMC')
    parser.add_argument('-p',
                        '--param_file',
                        required=False,
                        help='parameter file name')
    args = parser.parse_args()
    if args.param_file is None: #grab param file if no command line input, JKK
        str_yalm = f'initialization_files/{default_parameter_file}'
    else:
        str_yalm = args.param_file

    print("Read " + str_yalm + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)
    # Grab the info from the yaml file
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    KLIPDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'klip_fm_files')
    RESULTS_DIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'results_freeform')
    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']
    FIRST_TIME = params_mcmc_yaml["FIRST_TIME"]
    BASIS_FILE = f"{KLIPDIR}/{FILE_PREFIX}_klbasis.h5"

    
    # load wheremask2generatedisk
    WHEREMASK2GENERATEDISK = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits'))


    # load PSF
    psf = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_instrPSF.fits'))
    JAX_PSF = jnp.array(psf)
    JAX_PSF /= jnp.sum(JAX_PSF)

    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2

    # Read in the basis data
    fm_dict = load_kl_basis(BASIS_FILE)
    # fm_dict contains 
    # dict_keys(['aligned_images_dict', 'evals_dict', 'evecs_dict',
    # 'input_img_num_dict', 'klmodes_dict', 'section_ind_dict'])
    REDUCED_DATA = fits.getdata(os.path.join(KLIPDIR,
                                            FILE_PREFIX + '-klipped-KLmodes-all.fits'))[0]

    # mask the disk image
    TARGET_IMAGE = REDUCED_DATA * WHEREMASK2GENERATEDISK# * 1e4
    # print(INIT_MODEL_FLAT.shape)

    MASK = jnp.array(WHEREMASK2GENERATEDISK)  # convert to JAX array if needed
    MASK_INDICES = jnp.flatnonzero(MASK)  # 1D indices of nonzero (True) entries
    NUM_FREE = MASK_INDICES.shape[0]

    STARTING_DISK = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/lite_psflib/klip_fm_files/camsci2_z_20230309_10_FirstModel.fits")
    STARTING_DISK *= WHEREMASK2GENERATEDISK
    INIT_MODEL = jnp.array(STARTING_DISK)
    INIT_MODEL_FLAT = INIT_MODEL.reshape(INIT_MODEL.shape[0] * INIT_MODEL.shape[1])
    INIT_MODEL_INTEREST = INIT_MODEL_FLAT[MASK_INDICES]
    MODEL_RECONSTRUCTED = reconstruct_full_image(INIT_MODEL_INTEREST, TARGET_IMAGE.shape)
    # print(INIT_MODEL_INTEREST.shape)

    # plt.imshow(np.asarray(MODEL_RECONSTRUCTED),origin="lower")
    # plt.colorbar()
    # plt.show()
    # sys.exit()

    fm_full_image = prep_and_return_fm(target_image=TARGET_IMAGE,
                                       model_init=INIT_MODEL_INTEREST,
                                       psf=JAX_PSF, basis_data=fm_dict
                                )
    # fm_full_image = reconstruct_full_image(fm_init, TARGET_IMAGE.shape)
    # --- Visualization ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(np.asarray(TARGET_IMAGE), cmap='inferno', origin="lower")
    ax[0].set_title("Target Image (Ground Truth)")
    ax[0].axis("off")

    ax[1].imshow(np.array(fm_full_image), cmap='inferno', origin="lower")
    ax[1].set_title("Optimized Freeform Model")
    ax[1].axis("off")


    plt.show()