'''
Fit a pixel-by-pixel freeform model disk to a KLIP reduced image.

Using JAX.


Steps to develop:
1. Generate the KLIP image and save the basis for DiskFM.
2. Read in and save the instr. PSF to use for the FM.
3. Generate an image of random numbers as an intial guess and feed it to the
optimizer function.
4. ????
'''

import os
import sys
import argparse

basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_smlyot_20230309_10.yaml'  # name of the parameter file
# you can also call it with the python function argument -p


import time


# # because this error was coming up
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


from functools import partial
import numpy as np

import astropy.io.fits as fits



from modeling.disk_freeform import FreeFormDisk
from utils.io.save_results import save_ffdfit_outputs

from dev.pyklip.fmlib.funcs_JDFM import (
    update_disk, fm_from_eigen_adi, derotate_and_average
)

from utils.klip_basis import load_kl_basis, unpack_basis_data


import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax import lax
import optax
from optax.losses import huber_loss

# fm_from_eigen_adi_jit = jax.jit(
#     fm_from_eigen_adi,
#     donate_argnums=(1,2,3,4,5,6),
# )

def penalize_spatial_freq(model_ps, ref_model_ps, reg_lambda):
    '''
    We have an ideal scattered light disk model from a prior MCMC analysis.

    We can use this as a power spectrum reference to penalize high spatial frequencies.
    '''
    # Compute where freeform power exceeds ref model power
    excess_mask = model_ps > (ref_model_ps)
    excess_power = jnp.where(excess_mask, 
                             model_ps - ref_model_ps, #grab power at high freqs
                             0.0) #zero everything else

    # Return total excess as a scalar penalty
    # jax.debug.print("print(reg_lambda) -> {x}", x=reg_lambda)
    penalty = jnp.mean(excess_power)
    return reg_lambda * penalty

def fft_power_spectrum(image):
    """
    Compute the 2D power spectrum of an image (shifted so DC is at center).
    """
    fft = jnp.fft.fftshift(jnp.fft.fft2(image))
    power = jnp.abs(fft) ** 2
    return power #TODO the sum of this should be the variance of the mean-subbed image (Parseval's theorem)

# jax.config.update("jax_enable_x64", True)

# update_disk_jit = jax.jit(update_disk, static_argnames=["min_num_models",])
# reconstruct_full_image_jax = jax.jit(reconstruct_full_image,
#                                      static_argnames=["total_pixels"],
#                                      )
# insert_section_jaxed = jax.jit(insert_section_into_full_image,
#                                static_argnames=["full_shape","section_inds"])
# fm_jaxed_jit = jax.jit(fm_jaxed, static_argnames=[""])

def make_annular_mask(dimensions, inner_radius, outer_radius, center=None):
    """
    Create a binary annular mask for a 2D array.
    
    The pixels whose distance from the center is between inner_radius and outer_radius
    (inclusive) are set to 1; all other pixels are set to 0.
    
    Args:
        dimensions (tuple): (height, width) of the output mask.
        inner_radius (float): The inner radius (in pixels).
        outer_radius (float): The outer radius (in pixels).
        center (tuple, optional): (row, col) coordinates for the center.
                                  If None, defaults to the center of the array.
    
    Returns:
        jnp.ndarray: A binary mask with shape `dimensions` (1's inside the annulus, 0's outside).
    """
    H, W = dimensions
    if center is None:
        center = (H / 2 - 0.5, W / 2 - 0.5)
    
    # Create coordinate grid
    y, x = np.indices((H, W))
    # Compute the radial distance from the center for each pixel.
    # Note: center is given as (row, col) and x corresponds to column indices.
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create the binary mask: 1 inside the annulus, 0 elsewhere.
    mask = np.where((r >= inner_radius) & (r <= outer_radius), 1, 0)
    return mask

def record_pyklip_params(n_klmodes, iwa, owa, minrot, centering):
    params = {}

    params["n_KLmodes"] = n_klmodes
    params["IWA"] = iwa
    params["OWA"] = owa
    params["minrot"] = minrot
    params["aligned_center"] = centering

    return params


def reconstruct_full_image(free_params, total_pixels, mask_indices):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    full_shape = (int(np.sqrt(total_pixels)), int(np.sqrt(total_pixels)))
    full_flat = jnp.zeros(total_pixels)
    full_flat = full_flat.at[mask_indices].set(free_params)
    return full_flat.reshape(full_shape)

# def initialize_freeform_model_reduced():
#     """Initialize freeform model parameters for the unmasked region."""
#     rng = jax.random.PRNGKey(42)
#     # Instead of full image dimensions, only initialize num_free parameters.
#     free_params = jax.random.normal(rng, (NUM_FREE,))

#     return free_params


def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"
    kernel_size = psf.shape[0] #should be square
    # Pad image to maintain size
    # padded_image = jnp.pad(input_model, [(kernel_size//2, kernel_size//2),
    #                                (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply convoluted convolution
    model_convolved = fftconvolve(input_model, psf, mode="same")
    
    return model_convolved

def fm_scan_func(_, input_pt, full_sample_refs, full_sample_models):
    flat_model_here = input_pt["models"]
    # ref_psf_inds = input_pt["inds"]
    # ref_psfs_here = input_pt["refs"]
    aligned_image = input_pt["images"]
    # flat_model_refs_here = carry_models[ref_psf_inds,:]
    klmodes = input_pt["modes"]
    evals = input_pt["evals"]
    evecs = input_pt["evecs"]
    ref_inds = input_pt["ref_inds"]

    flat_postklip_psf_i = fm_from_eigen_adi(
            aligned_image,
            flat_model_here,
            klmodes,
            evals,
            evecs,
            ref_inds,
            full_sample_refs,
            full_sample_models,
        )
    
    return _, jnp.array(flat_postklip_psf_i)



@partial(jax.jit, static_argnames=["total_pixels", "isRDI", "reg_lambda"])
def loss_function(mod_pix_params, disk_image, psf, noise_map, ref_model_ps,
                  aligned_images, PAs, disk_mask_inds, iowa_sec_inds_arr, ref_psf_inds,
                  klmodes_stacked, evals, evecs_stacked,
                  total_pixels, isRDI, reg_lambda):
    """ measure the huber loss for a given disk freeform disk model.

    Steps executed:
     - create 2D disk model
     - penalize high spatial frequency using the first guess disk model
     - 2D convolve by the PSF
     - For N_images in the dataset, make N copies of model using the passed-
     in PAs to rotate them to the header parang values.
     - do the forward modeling using the image, reference images, and bank
     of rotated disk models.
     - compute the loss using the residuals of the KLIP image and FM divided
     by the variance map.

    Args (jnp.Array unless otherwise noted):
     - mod_pix_params: free pixel params in the 2D disk ROI.
     - disk_image: 2D KLIP-reduced image
     - psf: 2D empirical instrument PSF, L1 normalized
     - noise_map
     - ref_model_ps: power spectrum of the reference disk model for reg.
     - aligned_images: dataset of centered images, flattened (N_images, n_pixels)
     - PAs: PARANG header values for each image.
     - disk_mask_inds: indices for the disk ROI, flattened (1, m_pixels)
     - iowa_sec_inds_arr: inner-outer-working angle ROI indices, flattened (1, n_pixels)
     - ref_psf_inds: aligned_images indices for refs assoc. w/ each image (N_images, max_num_refs)
     - klmodes_stacked: flattened KL modes array (N_images, num_KL_modes, n_pixels)
     - evals: the eigenvalues for KLIP (N_images, num_KL_modes)
     - evecs_stacked: the eigenvectors (N_images, max_num_refs, num_KL_modes)
     - total_pixels (int): pixel count of the full-size image. Ex. 224**2 = 50176
     - isRDI (bool): toggle RDI mode
     - reg_lambda: regularization factor for high spatial frequency penalty

    Returns:
        mean huber loss for each pixel param.
    """

    # DM commands scaled to [0,1] fits cubes do like 10 secs of wall clock time
    # Spatil freq. such that speckles end up at 10 lamb/D
    # So the wind is the rate of change of the phase 2pi v k thing maybe over D
    isRDI = bool(isRDI)

    # Ensure that the model pixel values range [0,1]
    full_model_image = reconstruct_full_image(mod_pix_params, total_pixels, disk_mask_inds)
    full_model_norm = full_model_image / jnp.sum(full_model_image)

    # Compute the model power spectrum and use it to regularize high spatial freq.
    model_ps = fft_power_spectrum(full_model_norm)
    # we use the first_guess model as a reference
    hsf_penalty = penalize_spatial_freq(model_ps, ref_model_ps, reg_lambda=reg_lambda)

    freeform_image = convolve_model(full_model_image, psf)
    global_models_prepped = update_disk(model_disk=freeform_image,
                                        PAs=PAs,
                                        section_inds=iowa_sec_inds_arr,
                                        )
    # Make the pytreeeee for jax.lax.scan
    fm_calc_inputs = {
        "models": global_models_prepped,
        "images": aligned_images,
        "modes": klmodes_stacked,
        "evals": evals,
        "evecs": evecs_stacked,
        "ref_inds": ref_psf_inds,
    }

    # Loop over each image in the dataset to calculate the post-KLIP PSF using jax.lax.scan
    scan_func = partial(
        fm_scan_func,
        full_sample_refs=aligned_images,
        full_sample_models=global_models_prepped
    )
    _, flat_postklip_psfs = lax.scan(scan_func, None, fm_calc_inputs)

    # Reshape the postKLIP PSFs into 2D images and derotate them
    freeform_fm_full = derotate_and_average(flat_postklip_psfs, PAs, total_pixels, iowa_sec_inds_arr)
    freeform_fm_flat = jnp.reshape(freeform_fm_full, psf.shape[0] * psf.shape[1])

    # Grab just the disk ROI pixels
    freeform_fm_interest = freeform_fm_flat[disk_mask_inds]

    raw_loss = (freeform_fm_interest - disk_image) / noise_map**2
    # drive the Huber loss to zero residuals
    mean_huber = jnp.mean(huber_loss(raw_loss, 0.))

    return mean_huber + hsf_penalty #counts**2 units for both (kinda)



# --- JIT-Compiled Gradient Computation ---
# loss_and_grad = jax.jit(jax.value_and_grad(loss_function))
loss_and_grad = jax.value_and_grad(loss_function)


def optimize_model(target_image, model_init, ref_ps, noise_map, mask_indices,
                   psf, basis_data, total_pixels, num_steps, reg_lambda, lr=0.1):
    
    # dimension = img_dim
    jax_target_image = jnp.array(target_image).astype(jnp.float32)

    jax_noise_map = jnp.array(noise_map).astype(jnp.float32)

    # Initialize the initial image
    image_params = model_init.astype(jnp.float32)
    # image_params = initialize_freeform_model_reduced()

    # Set up optimizer, use adaptive stochastic grad descent (Adam)
    optimizer = optax.adam(lr)
    # initialize the internal state to track 1st and 2nd moments of the gradients
    opt_state =  optimizer.init(image_params) #this is all zeros initially

    loss_history = []

    basis_data_unpacked = unpack_basis_data(basis_data)

    # num_input_images = int(jax.device_get(basis_data["klparam_dict"]["nfiles"]))
    aligned_image_sections = jnp.array(basis_data_unpacked["aligned_images"]) #shape ex. (84, 39112)
    iowa_sec_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)


    klmodes_sections = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 6, 39112)
    # klmodes_sections = jnp.take(klmodes, section_inds[-1], axis=2, fill_value=0.)

    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    # the eigenvectors have been zero-padded at the ends to removed ragged-ness....
    evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes) ex. (84, 78, 6)
    # evecs have been unpacked, stacked, and ready to be BATCHED!
    # input_img_nums = basis_data_unpacked["input_img_nums"]
    # These are the images used for the basis for every image in the dataset.
    ref_psfs_sections = basis_data_unpacked["ref_psfs"] # zero-padded at the end to all have the same shape
    # ref_psfs_sections = jnp.take(ref_psfs, section_inds[-1], axis=2, fill_value=0.)


    PAs = jnp.array((basis_data["klparam_dict"]["PAs"]))
    # ref_PAs = basis_data_unpacked["ref_PAs"]

    if bool(basis_data_unpacked["klparams"]["isRDI"]):
        max_num_refs = klmodes_sections.shape[1]
        mode = 1
    elif not bool(basis_data_unpacked["klparams"]["isRDI"]):
        max_num_refs = basis_data_unpacked["fixed_refs"] #this is just a number smaller than N_images
        mode = 0
    # ref_psfs shape (N_images, max_N_refs, N_pixels) ex. (84, 78, 50176)
    # ref_psfs have been unpacked, stacked, and ready to be BATCHED!
    # position_angles = tuple(np.asarray(jax.device_get(basis_data["klparam_dict"]["PAs"])))
    # aligned_center = tuple(np.asarray(jax.device_get([basis_data["klparam_dict"]["aligned_center_x"],
    #                             basis_data["klparam_dict"]["aligned_center_y"]])))
    ref_psfs_inds = basis_data_unpacked["ref_inds"]
    

    time_now = time.time()
    # jax.profiler.start_trace("/tmp/tensorboard")
    # jax.config.update("jax_debug_nans", True)

    @jax.jit
    def step(image_params, opt_state):
        loss, grads = loss_and_grad(image_params, jax_target_image, psf, jax_noise_map,
                                    ref_ps, aligned_image_sections,
                                    PAs,
                                    mask_indices, iowa_sec_inds, ref_psfs_inds,
                                    klmodes_sections, evals, evecs, 
                                    total_pixels,
                                    mode, reg_lambda)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss

    for step_idx in range(num_steps):
        with jax.profiler.StepTraceAnnotation("train", step_num=step_idx):
            image_params, opt_state, loss = step(image_params, opt_state)
        
        loss_history.append(loss.item())

        # if step_idx % round(num_steps / 10) == 0:
        print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f} - {time.time() - time_now:.6f} sec elapsed")

    # jax.profiler.stop_trace()
    print(f"This run took {(time.time() - time_now):.6f} seconds.")
    
    # optimized_model = jax.nn.sigmoid(image_params)
    optimized_model = image_params
    return optimized_model, loss_history

def main(config, num_iterations, init_model, reg_lambda, first_time):
    # Grab the info from the yaml file
    if reg_lambda is None: #default
        reg_lambda = 1.

    # Initialize the freeform disk object
    ffd_obj = FreeFormDisk(config)

    # define needed variables
    klipdir = ffd_obj.klipdir
    resultsdir = ffd_obj.resultsdir
    file_prefix = ffd_obj.file_prefix # Ex. camsci1_i_20230309_10
    aligned_center = ffd_obj.aligned_center
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    mask2generatedisk = ffd_obj.prep_binary_masks()

    # Get the initial model

    model_firstguess = ffd_obj.get_initial_model(init_model)

    model_init_norm = model_firstguess / np.sum(model_firstguess)
    ps_ref_model = fft_power_spectrum(model_init_norm)

    # load PSF
    # psf = fits.getdata(os.path.join(klipdir, file_prefix + '_instrPSF.fits'))
    psf = ffd_obj.psf #psf gets normalized in the class
    jax_psf = jnp.array(psf)
    # jax_psf /= jnp.sum(jax_psf)

    # measure the size of images DIMENSION and make it global


    if ffd_obj.params_file["FIRST_TIME"] or bool(first_time):
        # initialize_diskfm and make diskobj global
        dataset = ffd_obj.prep_dataset()
        

        ffd_obj.initialize_diskfm(dataset, model_init=model_firstguess)

        print('First time initializing, check klip_fm_files directory and modify the yaml file first_time flag.')
        sys.exit(0)
    # Read in the basis data
    fm_dict = load_kl_basis(basis_path)
    # fm_dict contains 
    # dict_keys(['aligned_images_dict', 'evals_dict', 'evecs_dict',
    # 'input_img_num_dict', 'klmodes_dict', 'section_ind_dict'])
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))[0]
    reduced_data[reduced_data != reduced_data] = 0. #zero out the NaNs

    noise_map = fits.getdata(os.path.join(klipdir, f"{file_prefix}_noisemap.fits"))
    noise_map += 1. #get rid of any zeros
    noise_map_flat = noise_map.flatten()
    noise_map_flat[noise_map_flat != noise_map_flat] = 1.

    mask2generate_indices = jnp.flatnonzero(jnp.array(mask2generatedisk))[jnp.newaxis, :]
    
    total_pixels = np.prod(reduced_data.shape)



    disk_mask = np.array(mask2generatedisk)  # convert to JAX array if needed
    annular_mask = make_annular_mask(disk_mask.shape, 10, ffd_obj.owa)
    disk_mask *= annular_mask
    disk_mask[disk_mask != disk_mask] = 0.
    disk_mask_indices = jnp.flatnonzero(disk_mask)

    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[disk_mask_indices]
    # STARTING_DISK = fits.getdata("freeform_run.fits") #start from the last run
    # model_firstguess *= mask2generatedisk
    model_firstguess *= disk_mask
    init_model = jnp.array(model_firstguess)
    init_model_flat = init_model.reshape(init_model.shape[0] * init_model.shape[1])
    init_model_interest = init_model_flat[disk_mask_indices]

    noise_interest = noise_map_flat[disk_mask_indices]

    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
    optimized_model, loss_history = optimize_model(target_image=reduced_flat_interest,
                                                   model_init=init_model_interest, ref_ps=ps_ref_model,
                                                   noise_map=noise_interest,
                                                #    mask_indices=mask2generate_indices,
                                                   mask_indices=disk_mask_indices,
                                                   psf=jax_psf, basis_data=fm_dict,
                                                   total_pixels=total_pixels,
                                                   num_steps=num_iterations,
                                                   reg_lambda=reg_lambda,
                                                   )
    try:
        optimized_model.block_until_ready()
    except Exception as e:
        print(e)
    if trace_dest:
        jax.profiler.stop_trace()

    # Let's save what pyklip params were used with the outputs
    pyklip_params_dict = record_pyklip_params(ffd_obj.numbasis,
                                              ffd_obj.iwa,
                                              ffd_obj.owa,
                                              ffd_obj.minrot,
                                              ffd_obj.aligned_center
                                              )
    # optimized_model = np.asarray(reconstruct_full_image(optimized_model, total_pixels, mask2generate_indices))
    optimized_model = np.asarray(reconstruct_full_image(optimized_model, total_pixels, disk_mask_indices))
    # optimized_model = np.roll(optimized_model, (-1,-1))

    optimized_model_image = np.asarray(fftconvolve(optimized_model, psf, mode="same"))

    optimized_fm = ffd_obj.single_fm(np.asarray(optimized_model_image))

    residuals = np.asarray(reduced_data - optimized_fm)

    save_ffdfit_outputs(save_dir=resultsdir,
                        file_prefix=file_prefix,
                        model_opt=optimized_model,
                        model_image_opt=optimized_model_image,
                        forward_model_opt=optimized_fm,
                        residuals_image=residuals,
                        loss_history=loss_history,
                        hsf_regularization=reg_lambda,
                        pyklip_params=pyklip_params_dict
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run diskierFM autodiff')
    parser.add_argument('-p',
                        '--param-file',
                        required=False,
                        help='parameter file name')
    parser.add_argument(
                        '--iterations',
                        type=int,
                        required=True,
                        help='Num. iterations')
    parser.add_argument(
                        '--initial-model',
                        type=str,
                        required=True,
                        help='Path to starting model fits file')
    parser.add_argument(
                        '--reg',
                        type=float,
                        required=False,
                        help='Regularization factor')
    parser.add_argument(
                        '--first-time',
                        action="store_true",
                        required=False,
                        # default=False,
                        help='Startup procedure only')
    args = parser.parse_args()
    if args.param_file is None: #grab param file if no command line input, JKK
        str_yaml = f'initialization_files/{default_parameter_file}'
    else:
        str_yaml = args.param_file
        str_yaml_prefix = str_yaml.split("/")[-1]
        save_to_dir = str_yaml_prefix.split(".")[0]

    # print(args.reg)
    main(config=str_yaml,
        num_iterations=args.iterations,
        init_model=args.initial_model,
        reg_lambda=args.reg,
        first_time=args.first_time,
        )

    