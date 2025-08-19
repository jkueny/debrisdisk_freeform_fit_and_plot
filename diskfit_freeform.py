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

import sys
import argparse

default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file


import time

from functools import partial
import numpy as np

import astropy.io.fits as fits

from matplotlib import cm
magma_g = cm.magma.copy()
magma_g.set_bad('0.5')
viridis_g = cm.viridis.copy()
viridis_g.set_bad('0.5')

from modeling.disk_freeform import FreeFormDisk
from utils.io.save_results import save_ffdfit_outputs, get_next_run_dir

from dev.pyklip.fmlib.funcs_JDFM import (
    update_disk, fm_from_eigen_adi, fm_from_eigen_rdi, derotate_and_average
)

from utils.klip_basis import load_kl_basis, unpack_basis_data
from utils.regularization import penalize_residual_disk
from utils.masks import make_annular_mask

from utils.improc_tools import reconstruct_full_image, fft_power_spectrum, subtract_radial_profile, \
    get_radial_inds

from utils.diskfit_tools import convolve_model, record_pyklip_params, \
    penalize_spatial_freq

import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax import lax
from jaxopt import LBFGS
from optax.losses import huber_loss

# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_debug_nans", True)



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


def plot_training(out_filename, reduced_data, freeform_fm_full, full_model_image, updates, mask_indices, min_percent=1.0, max_percent=99.9):
    print('Saving', out_filename, '...', end=' ')
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    fig, axs = plt.subplots(ncols=4, figsize=(10, 3))
    fig.subplots_adjust(left=0.05, right=0.95)
    mask_tmp = np.zeros(reduced_data.size)
    mask_tmp[mask_indices] = 1.0
    mask_good = (mask_tmp == 1.0).reshape(reduced_data.shape)
    mask_bad = (mask_tmp == 0.0).reshape(reduced_data.shape)
    data_vmin, data_vmax = 0, np.percentile(reduced_data[mask_good], max_percent)
    data_space_norm = simple_norm(reduced_data, 'log', vmin=data_vmin, vmax=data_vmax)
    reduced_data_masked = np.array(reduced_data)
    reduced_data_masked[mask_bad] = np.nan
    plt.colorbar(axs[0].imshow(reduced_data_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    axs[0].set(title='Reduced data')
    freeform_fm_full_masked = np.array(freeform_fm_full)
    freeform_fm_full_masked[mask_bad] = np.nan
    plt.colorbar(axs[1].imshow(freeform_fm_full_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    full_model_image_masked = np.array(full_model_image)
    full_model_image_masked[mask_bad] = np.nan
    axs[1].set(title='Freeform FM')
    plt.colorbar(axs[2].imshow(full_model_image_masked, origin='lower', norm=simple_norm(full_model_image, 'log', min_percent=min_percent, max_percent=max_percent), cmap=magma_g))
    axs[2].set(title=r'Model')
    updates_vmax = np.max(np.abs(updates))
    plt.colorbar(axs[3].imshow(updates, origin='lower', vmax=updates_vmax, vmin=-updates_vmax, cmap='RdYlBu_r'))
    axs[3].set(title=r'Updates')
    fig.savefig(out_filename, dpi=128)
    plt.close(fig)
    print('Done.')


def loss_function(mod_pix_params, disk_image, psf, noise_map, ref_model_psd, disk_spine,
                  aligned_images, PAs, disk_mask_inds, iowa_sec_inds_arr, all_reference_images_selectors,
                  klmodes_stacked, evals, evecs_stacked,
                  radial_inds, aligned_center, isRDI, total_pixels, size, reg_lambda):
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
     - reg_lambda: regularization factor for high spatial frequency penalty

    Returns:
        mean huber loss for each pixel param.
    """
    full_model_image = reconstruct_full_image(mod_pix_params, total_pixels, size, disk_mask_inds)
    full_noise_image = reconstruct_full_image(noise_map, total_pixels, size, disk_mask_inds)
    # Ensure total intensity is 1.0
    full_model_norm = full_model_image / jnp.linalg.norm(full_model_image)
    full_model_norm_meansub = full_model_norm - jnp.mean(full_model_norm)

    # Compute the model power spectrum and use it to regularize high spatial freq.
    model_psd = fft_power_spectrum(full_model_norm_meansub)
    # model_psd_log = jnp.log10(model_psd + 1e-12)
    # model_psd_log_shifted = model_psd_log - model_psd_log.min()
    model_psd_log_shifted = model_psd
    # we use the first_guess model as a reference
    hsf_penalty = penalize_spatial_freq(model_psd_log_shifted, ref_model_psd, reg_lambda=reg_lambda)

    # jax.debug.print("hsf_penalty -> {x}", x=hsf_penalty)

    freeform_image = convolve_model(full_model_image, psf)
    freeform_image_profilesub, _ = subtract_radial_profile(freeform_image,
                                                           aligned_center,
                                                           full_noise_image,
                                                           radial_inds,
                                                           )
    global_models_prepped = update_disk(model_disk=freeform_image_profilesub,
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
    # jax.debug.breakpoint()
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

    raw_loss = (freeform_fm_interest - disk_image) / noise_map
    # drive the Huber loss to zero residuals
    mean_huber = jnp.mean(huber_loss(raw_loss))
    full_residuals_image = reconstruct_full_image(raw_loss, total_pixels, size, disk_mask_inds)
    residual_disk_loss = penalize_residual_disk(disk_spine, full_residuals_image, reg_lambda=reg_lambda)
    # jax.debug.print("residual_disk_loss -> {x}", x=residual_disk_loss)
    # jax.debug.print("hsf_penalty -> {x}", x=hsf_penalty)
    # jax.debug.print("mean_huber -> {x}", x=mean_huber)
    loss = mean_huber + hsf_penalty + residual_disk_loss #counts**2 units for both (kinda
    return loss



def optimize_model(
    target_image, model_init, ref_psd, disk_spine, noise_map, mask_indices, psf,
    basis_data, total_pixels, num_steps, reg_lambda, run_dir, reduced_data,
    radial_inds,
    aligned_center,
):
    target_image = jnp.array(target_image).astype(jnp.float32)
    noise_map = jnp.array(noise_map).astype(jnp.float32)

    basis_data_unpacked = unpack_basis_data(basis_data)
    aligned_image_sections = jnp.array(basis_data_unpacked["aligned_images"]) #shape ex. (84, 39112)
    n_images = aligned_image_sections.shape[0]
    iowa_sec_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)
    klmodes_sections = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 6, 39112)
    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    n_modes = evals.shape[1]
    subset_evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes) ex. (84, 78, 6)
    ref_psfs_inds = basis_data_unpacked["ref_inds"]

    # Initialize the initial image
    # image_params = jnp.median(target_image[mask_indices]) * model_init.astype(jnp.float32)
    image_params = model_init.astype(jnp.float32)

    # Set up LBFGS solver (replacing Adam optimizer)
    loss_history = []

    # doing this in mutable-array-land on CPU is faster
    # so we pre-fill the per-frame eigenbases in a consistent shape using
    # the indices from which the ref PSFs were drawn
    evecs = np.zeros((n_images, n_images, n_modes))
    for idx, img_ref_indices in enumerate(ref_psfs_inds):
        evecs[idx, img_ref_indices] = subset_evecs[idx]

    PAs = jnp.array((basis_data["klparam_dict"]["PAs"]))
    if bool(basis_data_unpacked["klparams"]["isRDI"]):
        max_num_refs = klmodes_sections.shape[1]
        mode = 1
    elif not bool(basis_data_unpacked["klparams"]["isRDI"]):
        max_num_refs = basis_data_unpacked["fixed_refs"] #this is just a number smaller than N_images
        mode = 0
    # ref_psfs shape (N_images, max_N_refs, N_pixels) ex. (84, 78, 50176)
    # ref_psfs have been unpacked, stacked, and ready to be BATCHED!
    # position_angles = tuple(np.asarray(jax.device_get(basis_data["klparam_dict"]["PAs"])))
    

    all_reference_images_selectors = np.zeros((aligned_image_sections.shape[0], aligned_image_sections.shape[0]), dtype=bool)
    for i in range(aligned_image_sections.shape[0]):
        for j in ref_psfs_inds[i]:
            all_reference_images_selectors[i, j] = True
    all_reference_images_selectors = jax.device_put(all_reference_images_selectors.astype(float))

    run_start_ts = time.time()

    total_pixels = int(total_pixels)
    H = W = int(np.sqrt(total_pixels))
    assert H * W == total_pixels, "KLIP image is not a perfect square?"
    size = H #this is the size of the full square image


    # Set up LBFGS solver
    solver = LBFGS(
        fun=loss_function,
        value_and_grad=False,
        maxiter=num_steps,
        has_aux=False,
        jit=False,
        implicit_diff=True,
    )

    # Run the LBFGS optimization
    res = solver.run(
        image_params,
        target_image, psf, noise_map, ref_psd, disk_spine,
        aligned_image_sections, PAs, mask_indices, iowa_sec_inds, all_reference_images_selectors,
        klmodes_sections, evals, evecs,
        radial_inds, aligned_center, mode, total_pixels, size, reg_lambda,
    )
    # Extract the optimized parameters and final loss
    print(res.state)
    optimized_model = res.params
    final_loss = res.state.value
    
    # For LBFGS, we can get some convergence info
    # converged = res.state.converged
    n_iterations = res.state.iter_num
    
    print(f"LBFGS optimization completed:")
    # print(f"  - Converged: {converged}")
    print(f"  - Iterations: {n_iterations}")
    print(f"  - Final loss: {final_loss:.6f}")
    print(f"This run took {(time.time() - run_start_ts):.6f} seconds.")

    # Since LBFGS doesn't provide step-by-step loss history like Adam,
    # we'll return a simple list with the final loss
    loss_history = [final_loss] if final_loss is not None else []

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
    # Render the reference model
    if init_model is None:
        reference_model = ffd_obj.render_reference_model()
        reference_model[reference_model != reference_model] = 0.
    else:
        reference_model = model_firstguess
    

    reference_model_psd, window_opt = ffd_obj.get_reference_model_psd(reference_model)

    # load PSF
    psf = ffd_obj.psf #psf gets normalized in the class
    jax_psf = jnp.array(psf)

    if ffd_obj.params_file["FIRST_TIME"] or bool(first_time):
        # initialize_diskfm and make diskobj global
        dataset = ffd_obj.prep_dataset()

        ffd_obj.initialize_diskfm(dataset, model_init=model_firstguess)

        disk_spine = ffd_obj.high_pass_reference_model(reference_model)
        print('First time initializing, check klip_fm_files directory and modify the yaml file first_time flag.')
        sys.exit(0)
    # Read in the basis data
    fm_dict = load_kl_basis(basis_path)
    # fm_dict contains 
    # dict_keys(['aligned_images_dict', 'evals_dict', 'evecs_dict',
    # 'input_img_num_dict', 'klmodes_dict', 'section_ind_dict'])
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))[0]
    reduced_data[reduced_data != reduced_data] = 0. #zero out the NaNs
    disk_spine = ffd_obj.high_pass_reference_model(reference_model)

    if ffd_obj.params_file["noise"]["use"]:
        noise_map = fits.getdata(os.path.join(klipdir, f"{file_prefix}_noisemap.fits"))
        noise_map += 1. #get rid of any zeros
        noise_map_flat = noise_map.flatten()
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.
    else:
        noise_map_flat = np.ones_like(reduced_data.flatten())
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.

    total_pixels = np.prod(reduced_data.shape)

    radial_inds = get_radial_inds(reduced_data.shape, aligned_center)

    # import matplotlib.pyplot as plt
    # plt.imshow(reference_model_psd, origin="lower")
    # plt.colorbar()
    # plt.show()
    # sys.exit(0)

    disk_mask = np.array(mask2generatedisk)  # convert to JAX array if needed
    annular_mask = make_annular_mask(disk_mask.shape, 10, ffd_obj.owa)
    disk_mask *= annular_mask
    disk_mask[disk_mask != disk_mask] = 0.
    disk_mask_indices = jnp.flatnonzero(disk_mask)

    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[disk_mask_indices]
    model_firstguess *= disk_mask
    init_model = jnp.array(model_firstguess)
    init_model_flat = init_model.reshape(init_model.shape[0] * init_model.shape[1])
    init_model_interest = init_model_flat[disk_mask_indices]

    noise_interest = noise_map_flat[disk_mask_indices]
    run_dir = get_next_run_dir(resultsdir)

    aligned_center = float(fm_dict["klparam_dict"]["aligned_center_x"]), float(fm_dict["klparam_dict"]["aligned_center_y"])
    image_shape = (jnp.round(aligned_center[0]) * 2, jnp.round(aligned_center[1]) * 2)
    image_size = int(np.sqrt(total_pixels))
    radial_inds = get_radial_inds(image_shape, aligned_center)
    # print(f"max(init_model_interest) -> {np.max(init_model_interest)}")

    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
        print(f"Tracing to {trace_dest}")
    optimized_model, loss_history = optimize_model(target_image=reduced_flat_interest,
                                                   model_init=init_model_interest, ref_psd=reference_model_psd,
                                                   disk_spine=disk_spine,
                                                   noise_map=noise_interest,
                                                #    mask_indices=mask2generate_indices,
                                                   aligned_center=aligned_center,
                                                   mask_indices=disk_mask_indices,
                                                   psf=jax_psf, basis_data=fm_dict,
                                                   total_pixels=total_pixels,
                                                   num_steps=num_iterations,
                                                   reg_lambda=reg_lambda,
                                                   run_dir=run_dir,
                                                   reduced_data=reduced_data,
                                                   radial_inds=radial_inds,
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
    optimized_model = np.asarray(reconstruct_full_image(optimized_model, total_pixels, image_size, disk_mask_indices))
    # optimized_model = np.roll(optimized_model, (-1,-1))

    print("Convolving optimized model image...")
    opt_image_no_rprofsub = fftconvolve(optimized_model, psf, mode="same")

    noise_reconstructed = reconstruct_full_image(jnp.array(noise_interest),
                                                 total_pixels,
                                                 image_size,
                                                 disk_mask_indices,
                                                )

    opt_model_image, med_prof_image = subtract_radial_profile(opt_image_no_rprofsub,
                                                              aligned_center,
                                                              noise_reconstructed,
                                                              radial_inds,
                                                              )

    optimized_model_image = np.asarray(opt_model_image)
    median_profile_image = np.asarray(med_prof_image)


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
                        median_profile_image=median_profile_image,
                        loss_history=loss_history,
                        hsf_regularization=reg_lambda,
                        pyklip_params=pyklip_params_dict,
                        optimizer_name="jaxopt.LBFGS"
                        )


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
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

    main(
        config=str_yaml,
        num_iterations=args.iterations,
        init_model=args.initial_model,
        reg_lambda=args.reg,
        first_time=args.first_time,
    )
