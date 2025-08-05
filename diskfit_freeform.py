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
import multiprocessing
multiprocessing.set_start_method('forkserver')
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
from utils.io.save_results import save_ffdfit_outputs, get_next_run_dir

from dev.pyklip.fmlib.funcs_JDFM import (
    update_disk, fm_from_eigen_adi, derotate_and_average
)

from utils.klip_basis import load_kl_basis, unpack_basis_data

from utils.masks import make_annular_mask

from utils.improc_tools import reconstruct_full_image, fft_power_spectrum

from utils.diskfit_tools import convolve_model, record_pyklip_params, \
    penalize_spatial_freq

import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax import lax
import optax
from optax.losses import huber_loss



def fm_scan_func(_, input_pt, full_sample_refs, full_sample_models):
    flat_model_here = input_pt["models"]
    # ref_psf_inds = input_pt["inds"]
    # ref_psfs_here = input_pt["refs"]
    aligned_image = input_pt["images"]
    # flat_model_refs_here = carry_models[ref_psf_inds,:]
    klmodes = input_pt["modes"]
    evals = input_pt["evals"]
    evecs = input_pt["evecs"]
    reference_images_selector_vec = input_pt["reference_images_selector_vec"]

    flat_postklip_psf_i = fm_from_eigen_adi(
            aligned_image,
            flat_model_here,
            klmodes,
            evals,
            evecs,
            reference_images_selector_vec,
            full_sample_refs,
            full_sample_models,
        )
    
    return _, jnp.array(flat_postklip_psf_i)


@partial(jax.jit, static_argnames=["total_pixels", "isRDI", "reg_lambda"])
def loss_function(mod_pix_params, disk_image, psf, noise_map, ref_model_ps,
                  aligned_images, PAs, disk_mask_inds, iowa_sec_inds_arr, all_reference_images_selectors,
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
     - all_reference_images_selectors: N_images x N_images vectors that are True/1 where a source image is included in the final dataset and false otherwise
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


    full_model_image = reconstruct_full_image(mod_pix_params, total_pixels, disk_mask_inds)
    # Ensure total intensity is 1.0
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
    # Make the pytree for jax.lax.scan
    fm_calc_inputs = {
        "models": global_models_prepped,
        "images": aligned_images,
        "modes": klmodes_stacked,
        "evals": evals,
        "evecs": evecs_stacked,
        "reference_images_selector_vec": all_reference_images_selectors,
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
    loss = mean_huber + hsf_penalty #counts**2 units for both (kinda)
    aux_data = freeform_fm_full, full_model_norm
    return loss, aux_data



# --- JIT-Compiled Gradient Computation ---
# loss_and_grad = jax.jit(jax.value_and_grad(loss_function))
loss_and_grad = jax.value_and_grad(loss_function, has_aux=True)


def optimize_model(target_image, model_init, ref_ps, noise_map, mask_indices,
                   psf, basis_data, total_pixels, num_steps, reg_lambda, run_dir, reduced_data, lr=0.1):
    
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
    n_images = aligned_image_sections.shape[0]
    iowa_sec_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)


    klmodes_sections = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 6, 39112)
    # klmodes_sections = jnp.take(klmodes, section_inds[-1], axis=2, fill_value=0.)

    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    n_modes = evals.shape[1]
    subset_evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes) ex. (84, 78, 6)
    evecs = np.zeros((n_images, n_images, n_modes))
    ref_psfs_inds = basis_data_unpacked["ref_inds"]
    # doing this in mutable-array-land on CPU is faster
    # so we pre-fill the per-frame eigenbases in a consistent shape using
    # the indices from which the ref PSFs were drawn
    for idx, img_ref_indices in enumerate(ref_psfs_inds):
        evecs[idx, img_ref_indices] = subset_evecs[idx]
    # evecs have been unpacked, stacked, and ready to be BATCHED!
    # input_img_nums = basis_data_unpacked["input_img_nums"]
    # These are the images used for the basis for every image in the dataset.
    # ref_psfs_sections = basis_data_unpacked["ref_psfs"] # zero-padded at the end to all have the same shape
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

    all_reference_images_selectors = np.zeros((aligned_image_sections.shape[0], aligned_image_sections.shape[0]), dtype=bool)
    for i in range(aligned_image_sections.shape[0]):
        for j in ref_psfs_inds[i]:
            all_reference_images_selectors[i, j] = True
    all_reference_images_selectors = jax.device_put(all_reference_images_selectors.astype(float))

    run_start_ts = time.time()
    # jax.profiler.start_trace("/tmp/tensorboard")
    # jax.config.update("jax_debug_nans", True)

    @jax.jit
    def step(image_params, opt_state):
        (loss, aux_data), grads = loss_and_grad(image_params, jax_target_image, psf, jax_noise_map,
                                    ref_ps, aligned_image_sections,
                                    PAs,
                                    mask_indices, iowa_sec_inds, all_reference_images_selectors,
                                    klmodes_sections, evals, evecs, 
                                    total_pixels,
                                    mode, reg_lambda)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss, aux_data

    first_step = time.time()
    measure_warmup = True
    plot_idx = 0
    for step_idx in range(num_steps):
        with jax.profiler.StepTraceAnnotation("train", step_num=step_idx):
            image_params, opt_state, loss, aux_data = step(image_params, opt_state)
        freeform_fm_full, full_model_norm = aux_data
        loss_history.append(loss.item())

        # if step_idx % round(num_steps / 10) == 0:
        if measure_warmup:
            first_step = time.time() - first_step
            dt = time.time() - run_start_ts
            measure_warmup = False
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f} - {dt:.6f} sec elapsed - ? sec / step")
        else:
            dt = time.time() - run_start_ts - first_step
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f} - {dt:.6f} sec elapsed - {dt / (step_idx+1):.6f} sec / step")
        if step_idx % 10 == 0:
            out_filename = f"{run_dir}/training_{plot_idx:05}.png"
            print('Saving', out_filename, '...', end=' ')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(ncols=3, figsize=(12, 3))
            plt.colorbar(axs[0].imshow(reduced_data, origin='lower'))
            axs[0].set(title='Reduced data')
            plt.colorbar(axs[1].imshow(freeform_fm_full, origin='lower'))
            axs[1].set(title='Freeform FM')
            plt.colorbar(axs[2].imshow(np.log10(full_model_norm), cmap='magma', origin='lower'))
            axs[2].set(title=r'$\log_{10}$(Model)')
            fig.savefig(out_filename)
            plt.close(fig)
            plot_idx += 1
            print('Done.')

    print(f"This run took {(time.time() - run_start_ts):.6f} seconds.")

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
    run_dir = get_next_run_dir(resultsdir)

    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
        print(f"Tracing to {trace_dest}")
    optimized_model, loss_history = optimize_model(target_image=reduced_flat_interest,
                                                   model_init=init_model_interest, ref_ps=ps_ref_model,
                                                   noise_map=noise_interest,
                                                #    mask_indices=mask2generate_indices,
                                                   mask_indices=disk_mask_indices,
                                                   psf=jax_psf, basis_data=fm_dict,
                                                   total_pixels=total_pixels,
                                                   num_steps=num_iterations,
                                                   reg_lambda=reg_lambda,
                                                   run_dir=run_dir,
                                                   reduced_data=reduced_data,
                                                   )
    try:
        optimized_model.block_until_ready()
    except Exception as e:
        print(e)
    if trace_dest:
        jax.profiler.stop_trace()
        print("Ended profiler trace")
    print("Saving pyklip params...")
    
    # Let's save what pyklip params were used with the outputs
    pyklip_params_dict = record_pyklip_params(ffd_obj.numbasis,
                                            ffd_obj.iwa,
                                            ffd_obj.owa,
                                            ffd_obj.minrot,
                                            ffd_obj.aligned_center
                                            )
    # optimized_model = np.asarray(reconstruct_full_image(optimized_model, total_pixels, mask2generate_indices))
    print("Reconstructing full image...")
    optimized_model = np.asarray(reconstruct_full_image(optimized_model, total_pixels, disk_mask_indices))
    # optimized_model = np.roll(optimized_model, (-1,-1))

    print("Convolving optimized model image...")
    optimized_model_image = np.asarray(fftconvolve(optimized_model, psf, mode="same"))

    print("Generating the optimized forward model image...")
    optimized_fm = ffd_obj.single_fm(np.asarray(optimized_model_image))

    print("Calculating residuals between reduced data and optimized forward model...")
    residuals = np.asarray(reduced_data - optimized_fm)

    save_ffdfit_outputs(run_dir=run_dir,
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
                        required=False,
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

    