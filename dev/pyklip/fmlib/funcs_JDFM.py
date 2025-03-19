"""Make the DiskFM procedure purely functional for JAX.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import jax.lax as lax
from functools import partial

from dev.pyklip.j_klip import rotate_image
from utils.klip_basis import build_batched_ref_images



def _get_section_indicies(input_shape, img_center, IOWA, flipx=False):
    """
    Gets the pixels (via numpy.where) that correspond to this section

    Args:
        input_shape: shape of the image [ysize, xsize] [pixels]
        img_center: [x,y] image center [pxiels]
        radstart: minimum radial distance of sector [pixels]
        radend: maximum radial distance of sector [pixels]
        phistart: minimum azimuthal coordinate of sector [radians]
        phiend: maximum azimuthal coordinate of sector [radians]
        padding: number of pixels to pad to the sector [pixels]
        parang: how much to rotate phi due to field rotation [IN DEGREES]
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.

    Returns:
        sector_ind: the pixel coordinates that corespond to this sector
    """
    IWA,OWA = IOWA

    # create a coordinate system.
    x, y = np.meshgrid(np.arange(input_shape[1] * 1.0), np.arange(input_shape[0] * 1.0))
    if flipx:
        x = img_center[0] - (x - img_center[0])        
    r = np.sqrt((x - img_center[0])**2 + (y - img_center[1])**2)

    radstart = np.asarray([IWA])
    radend = np.asarray([OWA])

    # normal case where there's no 2 pi wrap
    section_ind = np.where((r >= radstart) & (r < radend))


    return section_ind

def get_nan_indices(arr):
    """
    Return the indices of NaNs in arr.
    If no NaNs are found, returns an empty array with a statically-known shape.
    """
    has_nan = jnp.any(jnp.isnan(arr))
    
    def true_fn(_):
        # When NaNs exist, get indices.
        # jnp.nonzero returns a tuple; here we assume arr is 1D (adjust as needed)
        idx = jnp.nonzero(jnp.isnan(arr))[0]
        return idx

    def false_fn(_):
        # When no NaNs exist, return an empty array with shape (0,)
        return jnp.empty((0,), dtype=jnp.int32)

    return lax.cond(has_nan, true_fn, false_fn, operand=None)

def nanmedian(x):
    """
    Compute the median of non-NaN values in a JAX array x.
    
    Strategy:
      1. Flatten x.
      2. Build a mask for valid (non-NaN) entries.
      3. Replace NaNs with a large value (jnp.inf) so that they sort to the end.
      4. Sort the flattened array.
      5. Count the valid entries.
      6. Use lax.cond to select the median value from the valid portion.
    """
    flat = x.ravel()
    valid_mask = ~jnp.isnan(flat)
    # Replace NaNs with +infinity so that valid numbers sort first.
    flat_filled = jnp.where(valid_mask, flat, jnp.inf)
    sorted_flat = jnp.sort(flat_filled)
    valid_count = jnp.sum(valid_mask)

    # Use lax.cond to conditionally select the median.
    median_value = lax.cond(
        valid_count > 0,
        # True branch: extract the median using jnp.take and squeeze to force a scalar.
        lambda cnt: jnp.squeeze(jnp.take(sorted_flat, cnt // 2)),
        # False branch: if no valid elements exist, return 0.0 (or another fallback).
        lambda _: 0.0,
        operand=valid_count
    )
    return median_value

# @jax.jit
def replace_nan_with_median(image):
    """
    Returns a new image where all NaN pixels are replaced with the median of the non-NaN pixels.
    """
    med_val = nanmedian(image)
    return jnp.where(jnp.isnan(image), med_val, image)

def derotate_section(input_shape, sector, sector_ind, angle, IOWA, img_center, flipx=True):
    """
    Rotate sector in output image at desired ranges

    Args:
        input_shape: shape of input_image
        sector: data in the sector to save to output_img
        sector_ind: index into input img (corresponding to input_shape) for the original sector
        angle: angle that the sector needs to rotate (I forget the convention right now)

        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
        img_center: center of image in input image coordinate

        flipx: if true, flip the x coordinate to switch coordinate handiness

    """
    # convert angle to radians
    angle_rad = jnp.radians(angle)


    # create the coordinate system of the image to manipulate for the transform
    dims = input_shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=np.float32), jnp.arange(dims[0], dtype=np.float32))


    # flip x if needed to get East left of North
    if flipx is True:
        x = img_center[0] - (x - img_center[0])

    # do rotation. CW rotation formula to get a CCW of the image
    xp = (x-img_center[0])*jnp.cos(angle_rad) + (y-img_center[1])*jnp.sin(angle_rad) + img_center[0]
    yp = -(x-img_center[0])*jnp.sin(angle_rad) + (y-img_center[1])*jnp.cos(angle_rad) + img_center[1]

    rot_sector_pix = _get_section_indicies(input_shape, img_center, IOWA, flipx=True)


    # do NaN detection by defining any pixel in the new coordiante system (xp, yp) as a nan
    # if any one of the neighboring pixels in the original image is a nan
    # e.g. (xp, yp) = (120.1, 200.1) is nan if either (120, 200), (121, 200), (120, 201), (121, 201)
    # is a nan
    dims = input_shape
    blank_input = jnp.zeros(dims[1] * dims[0])
    set_as_sector_flat = blank_input.at[sector_ind].set(sector)
    set_as_sector_2D = jnp.reshape(set_as_sector_flat, [dims[0], dims[1]])

    # floor and ceil to handle sub-pixel values
    xp_floor = jnp.clip(jnp.floor(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
    xp_flat_floor = jnp.ravel(xp_floor)
    xp_ceil = jnp.clip(jnp.ceil(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
    xp_flat_ceil = jnp.ravel(xp_ceil)
    yp_floor = jnp.clip(jnp.floor(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
    yp_flat_floor = jnp.ravel(yp_floor)
    yp_ceil = jnp.clip(jnp.ceil(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
    yp_flat_ceil = jnp.ravel(yp_ceil)
    # rotnans = jnp.where(jnp.isnan(set_as_sector_2D[yp_flat_floor, xp_flat_floor]) | 
    #                    jnp.isnan(set_as_sector_2D[yp_flat_floor, xp_flat_ceil]) |
    #                    jnp.isnan(set_as_sector_2D[yp_flat_ceil, xp_flat_floor]) |
    #                    jnp.isnan(set_as_sector_2D[yp_flat_ceil, xp_flat_ceil]))
    
    # Compute the combined NaN mask from the four neighboring pixels.

    max_size = int(xp_flat_floor.shape[0])  # this should be a Python integer
    nan_mask = (
        jnp.isnan(set_as_sector_2D[yp_flat_floor, xp_flat_floor]) |
        jnp.isnan(set_as_sector_2D[yp_flat_floor, xp_flat_ceil])  |
        jnp.isnan(set_as_sector_2D[yp_flat_ceil,   xp_flat_floor]) |
        jnp.isnan(set_as_sector_2D[yp_flat_ceil,   xp_flat_ceil])
        )
    # Define a helper function that returns the indices (with fixed size) where nan_mask is True.
    def get_nan_indices(nan_mask, max_size):
        # Here we use jnp.nonzero with a static size.
        # If there are fewer than max_size indices, the remainder will be filled with fill_value (-1).
        indices = jnp.nonzero(nan_mask, size=max_size, fill_value=-1)[0]
        return indices

    # Define branch functions for lax.cond.
    def branch_with_nans(_):
        return get_nan_indices(nan_mask, max_size)

    def branch_without_nans(_):
        # Return an empty array with shape (max_size,) and a fill value (e.g. -1)
        return jnp.full((max_size,), -1, dtype=jnp.int32)

    # Use lax.cond to choose the correct branch.
    rotnans = lax.cond(jnp.any(nan_mask),
                    branch_with_nans,
                    branch_without_nans,
                    operand=None)
    # resample image based on new coordinates, set nan values as median
    # nanpix = jnp.isnan(set_as_sector_2D)
    # medval = np.median(blank_input[np.where(~np.isnan(blank_input))])
    # medval = jnp.median(set_as_sector_2D[~nanpix])
    # medval = nanmedian(set_as_sector_2D)
    input_copy = jnp.copy(set_as_sector_2D)
    input_copy_nonan = replace_nan_with_median(input_copy)
    rot_sector = map_coordinates(input_copy_nonan,
                                         [yp[rot_sector_pix], xp[rot_sector_pix]],
                                         order=0, cval=np.nan)
    # rot_sector = klip.bilinear_interpolate(input_copy, yp[rot_sector_pix], xp[rot_sector_pix])

    # mask nans
    rot_sector_nans = rot_sector.at[rotnans].set(np.nan)
    # sector_validpix = np.where(~np.isnan(rot_sector))
    sector_validpix = ~jnp.isnan(rot_sector_nans)

    # need to define only where the non nan pixels are, so we can store those in the output image
    blank_output = jnp.zeros([dims[0], dims[1]]) * np.nan
    blank_output_valid = blank_output.at[rot_sector_pix].set(rot_sector)
    blank_output_valid_reshape = jnp.reshape(blank_output_valid, (dims[0], dims[1]))
    rot_sector_validpix_2D = jnp.where(~jnp.isnan(blank_output_valid_reshape))
    # rot_sector_validpix_2d = np.isnan(blank_output)


    return rot_sector_validpix_2D

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

def update_disk(model_disk, PAs, ref_PAs, section_inds, min_num_models):
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
    global_disks = jnp.tile(model_disk, (N_global, 1, 1))
    # Rotate each copy by its corresponding global PA.
    global_rot = jax.vmap(rotate_image)(global_disks, PAs)

    global_rot_flipx = jnp.flip(global_rot, axis=2)
    # Helper: apply section indices.
    def apply_section(img, inds):
        return img[inds].reshape(inds.shape[1])
    # Flatten each sectioned image.
    global_rot_flat = global_rot_flipx.reshape((N_global, -1))
    # Apply sectioning to each global rotated disk.
    global_rot_section_flat = jax.vmap(apply_section, in_axes=(0, None))(global_rot_flat, section_inds)
    
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

# @jax.jit
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
    eps = 1e-6
    # N_pix = original_KL.shape[1]

    refs_mean_sub = refs - jnp.mean(refs, axis=1, keepdims=True)

    # refs_meansub_nonan = jnp.nan_to_num(refs_mean_sub, nan=0.0) 

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    # models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
    # models_meansub_nonan = jnp.nan_to_num(models_mean_sub, nan=0.0)

    #print(evals.shape,evecs.shape,original_KL.shape,refs.shape,models_ref.shape)

    evals_tiled = jnp.tile(evals,(max_basis,1))
    evals_nan_diag = jnp.fill_diagonal(evals_tiled, 1., inplace=False)
    # print(evals_tiled)
    # sys.exit()
    evals_sqrt = jnp.sqrt(evals) + eps
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./((evals_nan_diag.transpose()- evals_nan_diag) + eps)
    #print(evals)
    beta_tmp = beta_tmp.at[np.diag_indices(np.size(evals))].set(-0.5/evals)
    beta = evals_ratio*beta_tmp #no NaNs confirmed JKK 03/18/2025

    C_partial = models_mean_sub.dot(refs_mean_sub.transpose())
    C = C_partial+C_partial.transpose()
    #C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha_tmp = jnp.dot(evecs.transpose(), C)
    alpha = jnp.dot(alpha_tmp, evecs)

    delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)


    return delta_KL

# @jax.jit
def fm_from_eigen_single(sci_data, refs_data, model_disk_sci, model_disk_refs,
                         klmodes, evals, evecs,):
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
    # Save the rotated section.
    # derotated_output = rotate_image(postklip_psf_corrected,
    #                                 -parang,
    #                                 # flip_x=False
    #                                 )
    return postklip_psf_corrected

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
    postklip_psfs = jax.vmap(fm_from_eigen_single
                          )(aligned_images, ref_psfs_stacked,
                            model_disks,ref_models_stacked,
                            klmodes_stacked, evals_arr, evecs_stacked,
                            )
    # jax.debug.print("print(postklip_psfs.shape) -> {x}", x=postklip_psfs.shape)
    # fm_outputs = fm_from_eigen_single(aligned_images, ref_psfs_stacked,
    #                         model_disks,ref_models_stacked,
    #                         klmodes_stacked, evals_arr, evecs_stacked,
    #                         PAs)
    # fm_outputs_squeezed = jnp.squeeze(fm_outputs)
    # fm_out = jnp.nanmean(fm_outputs_squeezed,axis=0)

    # jax.debug.print("print(fm_out.shape; after nanmean) -> {x}", x=fm_out.shape)

    return postklip_psfs