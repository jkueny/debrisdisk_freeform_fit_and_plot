"""Make the DiskFM procedure purely functional for JAX.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import jax.lax as lax

from dev.pyklip.j_klip import rotate_image

def mass_derotation(flat_postklip_psfs, PAs, total_pixels, section_inds):
    # print(f"image_dim -> {image_dim}")
    image_dim = (int(np.sqrt(total_pixels)), int(np.sqrt(total_pixels)))
    squeezed_postklip_psfs = jnp.squeeze(flat_postklip_psfs)
    postklip_psf_images = jax.vmap(insert_section_into_full_image,
                                   in_axes=(0, None, None))(squeezed_postklip_psfs,
                                                                   image_dim,
                                                                   section_inds)
    # print(f"postklip_psf_images.shape -> {postklip_psf_images.shape}")
    derotated_postklip_psfs = jax.vmap(rotate_image)(postklip_psf_images,
                                                     -PAs)
    # not sure how we get to needing to flip both axes... TODO investigate
    corrected_postklip_psfs = jnp.flip(derotated_postklip_psfs, axis=(1,2))
    return corrected_postklip_psfs



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

def _apply_section(img, inds):
    return img[inds].reshape(inds.shape[1])


def pad_array_to_fixed_first_dim(arr, fixed_first_dim, pad_value=0.0):
    """
    Pads a 2D array along axis 0 to have shape (fixed_first_dim, arr.shape[1]).
    Raises an error if arr already has more rows than fixed_first_dim.
    """
    current_first_dim = arr.shape[0]
    if current_first_dim > fixed_first_dim:
        raise ValueError(f"Array has {current_first_dim} rows, which exceeds the fixed limit of {fixed_first_dim}.")
    pad_rows = fixed_first_dim - current_first_dim
    if pad_rows > 0:
        pad_width = [(0, pad_rows), (0,0)]
        # print(pad_width)
        return jnp.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    else:
        return arr

def update_disk(model_disk, PAs, ref_PAs, section_inds, min_num_models, isRDI):
    """
    Takes a 2D model disk and produces two outputs:
    
      1. global_rotated: A JAX array of flattened disk models rotated by the global PAs,
         then sectioned using section_inds.
         Shape: (N_global, N_pixels_section), where N_pixels_section is the number
         of pixels in the region-of-interest.
         
      2. ref_rotated: A JAX array of flattened disk models rotated by each reference PA
         for each global image. For each global image, if the number of reference angles
         is less than min_num_models, the output is zero-padded along that axis.
         Final shape: (N_global, min_num_models, N_pixels_section)
         
    Args:
      model_disk: 2D JAX array of shape (height, width).
      PAs: JAX array of global position angles (in degrees) with shape (N_global,).
      ref_PAs: Tuple (or list) of 1D JAX arrays; ref_PAs[i] contains the reference angles
               for the i-th global image (ragged, not padded).
      aligned_center: Tuple (cx, cy) specifying the center of rotation.
      section_inds: Tuple (or list) of length N_global. Each element is a tuple (row_inds, col_inds)
                    that selects the region-of-interest from an image.
      min_num_models: Integer; fixed number of reference models to output per global image.
    
    Returns:
      global_rotated: JAX array of shape (N_global, N_pixels_section) containing the flattened,
                      sectioned, globally rotated disk models.
      ref_rotated: JAX array of shape (N_global, min_num_models, N_pixels_section) containing
                   the flattened, sectioned disk models rotated by each reference PA (padded as needed).
    """
    # Global rotations.
    N_global = PAs.shape[0]
    # Tile the model_disk to get one copy per global image.
    global_disks = jnp.tile(model_disk, reps=(N_global, 1, 1))
    # Rotate each copy by its corresponding global PA.
    global_rot = jax.vmap(rotate_image)(global_disks, PAs)

    # global_rot_flipx = jnp.flip(global_rot, axis=2)
    global_rot_flipx = global_rot

    # Flatten each sectioned image.
    global_rot_flat = global_rot_flipx.reshape((N_global, -1))
    # Apply sectioning to each global rotated disk.
    global_rot_section_flat = jax.vmap(_apply_section, in_axes=(0, None))(global_rot_flat, section_inds)
    
    if not isRDI:
        # Reference rotations.
        ref_rotated_list = []
        for i in range(N_global):
            # For global image i, ref_PAs[i] is a 1D array of reference angles.
            ref_angles = ref_PAs[i]
            # Rotate model_disk for each reference angle.
            # Note: We rotate the same model_disk for each reference PA.
            ref_rot = jax.vmap(lambda angle: rotate_image(model_disk, angle))(ref_angles)
            # ref_rot has shape (L, height, width), where L = len(ref_angles).
            # Now, apply the section for global image i.
            # Flatten each sectioned reference model.
            ref_rot_flat = ref_rot.reshape((ref_rot.shape[0], -1)) # (78,50176) or (N_refs, N_pixels)
            ref_rot_sec = jax.vmap(lambda img: img[section_inds])(ref_rot_flat)
            ref_rot_sec_flat = jnp.squeeze(ref_rot_sec)
            # Pad along axis 0 so that each global image has min_num_models models.
            ref_rot_sec_flat_padded = pad_array_to_fixed_first_dim(ref_rot_sec_flat, min_num_models, pad_value=0)
            ref_rotated_list.append(jnp.squeeze(ref_rot_sec_flat_padded))
        # Stack the results to obtain shape (N_global, min_num_models, N_pixels_section).
        ref_rotated = jnp.stack(ref_rotated_list)

        return global_rot_section_flat, ref_rotated
    elif isRDI:
        return global_rot_section_flat
    
def update_wind(model_wdhs, PAs, ref_inds, section_inds,
                mask_skip_models, isRDI):
    """
    Takes a 2D model disk and produces two outputs:
    
      1. global_rotated: A JAX array of flattened disk models rotated by the global PAs,
         then sectioned using section_inds.
         Shape: (N_global, N_pixels_section), where N_pixels_section is the number
         of pixels in the region-of-interest.
         
      2. ref_rotated: A JAX array of flattened disk models rotated by each reference PA
         for each global image. For each global image, if the number of reference angles
         is less than min_num_models, the output is zero-padded along that axis.
         Final shape: (N_global, min_num_models, N_pixels_section)
         
    Args:
      model_disk: 2D JAX array of shape (height, width).
      PAs: JAX array of global position angles (in degrees) with shape (N_global,).
      ref_PAs: JAX array of shape (N_global, min_num_models) that contains the PAs of the
      assoc. wind directions in the reference images used in the basis. Padded, not ragged.
      aligned_center: Tuple (cx, cy) specifying the center of rotation.
      section_inds: Tuple (or list) of length N_global. Each element is a tuple (row_inds, col_inds)
                    that selects the region-of-interest from an image.
      min_num_models: Integer; fixed number of reference models to output per global image.
    
    Returns:
      global_rotated: JAX array of shape (N_global, N_pixels_section) containing the flattened,
                      sectioned, globally rotated disk models.
      ref_rotated: JAX array of shape (N_global, min_num_models, N_pixels_section) containing
                   the flattened, sectioned disk models rotated by each reference PA (padded as needed).
    """
    N_models, H, W = model_wdhs.shape
    N_img, K_ref   = ref_inds.shape
    N_pix          = section_inds[0].size

    # ----------------------------------------------------------
    # 1) build *combined* WDH model for every science frame
    # ----------------------------------------------------------
    # mask for layers that exist in each frame
    PAs_clean   = jnp.where(mask_skip_models, PAs, 0.0)           # any angle OK where mask==0

    def _one_frame(pa_vec, mask_vec):
        # rotate each layer → (N_models,H,W)
        rot = jax.vmap(rotate_image)(model_wdhs, pa_vec)
        # zero where layer absent and sum
        return (rot * mask_vec[:, None, None]).sum(axis=0)   # (H,W)

    # (N_img,H,W)
    combined_all = jax.vmap(_one_frame)(PAs_clean.T, mask_skip_models.T)

    combined_flat = combined_all.reshape(N_img, H*W)

    # section & flatten once – this feeds both outputs
    global_flat = jax.vmap(_apply_section, in_axes=(0, None))(
        combined_flat, section_inds
    )                                                       # (N_img,N_pix)

    # ----------------------------------------------------------
    # 2) RDI?  we're done
    # ----------------------------------------------------------
    if isRDI:
        return global_flat, None

    # ----------------------------------------------------------
    # 3) build *reference cube* by indexing global_flat
    # ----------------------------------------------------------
    valid_mask   = ref_inds != -1           # (N_img,K_ref) bool
    refs_clean   = jnp.where(valid_mask, ref_inds, 0) # put 0 where padding

    # gather → (N_img,K_ref,N_pix)
    ref_cube = jnp.take(global_flat, refs_clean, axis=0)
    ref_cube = ref_cube * valid_mask[..., None]              # zero padded slots

    return global_flat, ref_cube

# @jax.jit
def calculate_fm(delta_KL, original_KL, sci, model_sci):
    """
    Same function as calculate_fm() but faster when numbasis has only one element. It doesn't do the mutliplication with
    the triangular matrix.

    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
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
    sci_mean_sub = sci - jnp.mean(sci)
    # sci_meansub_nonan = jnp.nan_to_num(sci_mean_sub, nan=0.0)
    sci_mean_sub_rows = jnp.reshape(sci_mean_sub,(1,N_pix))


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci # should be subtracting off the mean?

    # model_sci_meansub_nonans = jnp.nan_to_num(model_sci_mean_sub, nan=0.0)
    model_sci_mean_sub_rows = np.reshape(model_sci_mean_sub,(1,N_pix))


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
    evals_nan_diag = jnp.fill_diagonal(evals_tiled, 1., inplace=False)
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

# @jax.jit
def fm_from_eigen_adi(sci_data, refs_data, model_disk_sci, model_disk_refs,
                         klmodes, evals, evecs):
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

    postklip_psf_corrected = jnp.flip(postklip_psf, axis=1)
    # postklip_psf_corrected = postklip_psf
    # Save the rotated section.
    # derotated_output = rotate_image(postklip_psf_corrected,
    #                                 -parang,
    #                                 # flip_x=False
    #                                 )
    return postklip_psf_corrected

def fm_from_eigen_rdi(sci_data, model_disk_sci,
                         klmodes):
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
    delta_KL = 0. * klmodes

    # Calculate the post-KLIP PSF using your forward modeling routine.
    postklip_psf, _, _ = calculate_fm(delta_KL, klmodes,
                                      sci_data, model_disk_sci)

    postklip_psf_corrected = jnp.flip(postklip_psf, axis=1)
    # postklip_psf_corrected = postklip_psf
    # Save the rotated section.
    # derotated_output = rotate_image(postklip_psf_corrected,
    #                                 -parang,
    #                                 # flip_x=False
    #                                 )
    return postklip_psf_corrected
