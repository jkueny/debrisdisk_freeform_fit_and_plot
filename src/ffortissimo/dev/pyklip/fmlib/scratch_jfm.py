import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import jax.lax as lax
from functools import partial

from dev.pyklip.j_klip import rotate_image

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

@jax.jit
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
        output_img: the array to save the data to
        output_img_numstacked: array to increment region where we saved output to to bookkeep stacking. None for
                               skipping bookkeeping
        angle: angle that the sector needs to rotate (I forget the convention right now)

        The next 6 parameters define the sector geometry in input image coordinates
        radstart: radius from img_center of start of sector
        radend: radius from img_center of end of sector
        phistart: azimuthal start of sector
        phiend: azimuthal end of sector
        padding: amount of padding around each sector
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
        img_center: center of image in input image coordinate

        flipx: if true, flip the x coordinate to switch coordinate handiness
        new_center: if not none, center of output_img. If none, center stays the same
    """
    # convert angle to radians
    angle_rad = jnp.radians(angle)

    #incorporate padding
    # IWA,OWA = IOWA
    # radstart_padded = np.max([radstart-padding,IWA])
    # if OWA is not None:
    #     radend_padded = np.min([radend+padding,OWA])
    # else:
    #     radend_padded = radend+padding
    # phistart_padded = (phistart - padding/np.mean([radstart, radend])) % (2 * np.pi)
    # phiend_padded = (phiend + padding/np.mean([radstart, radend])) % (2 * np.pi)

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

def calculate_fm_singleNumbasis(delta_KL_nospec, original_KL, numbasis, sci, model_sci):
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
    if numbasis[0] is None:
        numbasis_index = [max_basis-1]
    else:
        numbasis_index = np.clip(numbasis - 1, 0, max_basis-1)

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


    delta_KL = delta_KL_nospec


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
    if np.size(delta_KL.shape) == 2:
        selfsubtraction_1_inner_products = jnp.dot(sci_mean_sub_rows, delta_KL.T)
        # selfsubtraction_1_inner_products.shape = (max_basis,N_pix,max_basis)
    else:
        Nlambda = delta_KL.shape[1]
        #Before delta_KL.shape = (max_basis,N_lambda or N_ref,N_pix)
        # delta_KL = jnp.rollaxis(delta_KL,1,0)
        delta_KL = jnp.moveaxis(delta_KL,1,0)
        #Now delta_KL.shape = (N_lambda or N_ref,max_basis,N_pix)
        # np.rollaxis(delta_KL,2,1).shape = (N_lambda or N_ref,N_pix,max_basis)
        # np.dot() takes the last dimension of first array and sum over the second to last dimension of second array
        # selfsubtraction_1_inner_products = jnp.dot(sci_mean_sub_rows, jnp.rollaxis(delta_KL,2,1))
        selfsubtraction_1_inner_products = jnp.dot(sci_mean_sub_rows, jnp.moveaxis(delta_KL,2,1))
    selfsubtraction_2_inner_products = jnp.dot(sci_mean_sub_rows, original_KL.T)


    thresh_oversub_inner_products = oversubtraction_inner_products.at[max_basis::].set(0)
    klipped_oversub = jnp.dot(thresh_oversub_inner_products, original_KL)

    thresh_selfsubtraction_1_inner_products = selfsubtraction_1_inner_products.at[0,max_basis::].set(0)

    thresh_selfsubtraction_2_inner_products = selfsubtraction_2_inner_products.at[0,max_basis::].set(0)
    klipped_selfsub = jnp.dot(thresh_selfsubtraction_1_inner_products, original_KL) + \
                        jnp.dot(thresh_selfsubtraction_2_inner_products, delta_KL)


    return model_sci[None,:] - klipped_oversub - klipped_selfsub, klipped_oversub, klipped_selfsub
     

def perturb_specIncluded(evals, evecs, original_KL, refs, models_ref, return_perturb_covar=False):
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
    jnp.fill_diagonal(evals_tiled,np.nan, inplace=False)
    evals_sqrt = jnp.sqrt(evals)
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
    #print(evals)
    beta_tmp = beta_tmp.at[np.diag_indices(np.size(evals))].set(-0.5/evals)
    beta = evals_ratio*beta_tmp

    C_partial = models_meansub_nonan.dot(refs_meansub_nonan.transpose())
    C = C_partial+C_partial.transpose()
    #C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha_tmp = jnp.dot(evecs.transpose(), C)
    alpha = jnp.dot(alpha_tmp, evecs)

    delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)

    if return_perturb_covar:
        return delta_KL, C
    else:
        return delta_KL

def fm_from_eigen_single(model_disks, klmodes, evals, evecs, aligned_imgs,
                         section_ind, parang, input_img_num, 
                         input_img_shape, ref_psfs_indicies,
                         IOWA, aligned_center,  numbasis):
    """
    Compute the forward model for one section.

    Note:
        - All inputs must be JAX arrays (or scalars) with fixed shapes.
        - Any scalar or static parameters that are the same for every section
        can be passed as-is.
    """
    # Extract science image and reference images based on section indices.
    sci = aligned_imgs[input_img_num, section_ind[0]]
    refs = aligned_imgs[ref_psfs_indicies, :]
    refs = refs[:, section_ind[0]]

    # Process the model disk stored in the class.
    model_sci = model_disks[input_img_num, section_ind[0]]
    model_sci_nonan = jnp.nan_to_num(model_sci, nan=0.0)
    model_refs_full = model_disks[ref_psfs_indicies, :]
    model_ref = model_refs_full[:, section_ind[0]]
    model_ref_nonan = jnp.nan_to_num(model_ref, nan=0.0)

    # Compute delta_KL (set to zero if mode=='RDI')
    delta_KL = perturb_specIncluded(evals, evecs, klmodes, refs,
                                                    model_ref_nonan,
                                                    return_perturb_covar=False)
    # Calculate the post-KLIP PSF using your forward modeling routine.
    postklip_psf, _, _ = calculate_fm_singleNumbasis(delta_KL, klmodes, numbasis,
                                                     sci, model_sci_nonan,
                                                     inputflux=None)
    # Save the rotated section.
    output_img = derotate_section(
                                input_img_shape,
                                postklip_psf,
                                section_ind,
                                parang,
                                IOWA,
                                aligned_center,
                                flipx=True)
    return output_img

def fm_vectorized(model_disks, aligned_imgs, klmodes_dict, evals_dict, evecs_dict, section_ind_dict,
                             PAs, input_img_nums, input_img_shape, IOWA, aligned_center,
                             numbasis):
    """Do the forward modeling procedure using JAX's vmap() framework.

    TODO Handle the nested dictionaries that are klmodes_dict, evals_dict,
    evecs_dict, section_ind_dict. This is because each of the images in the
    aligned_images has a corresponding set of KL modes, eigenvectors, and
    eigenvalues. Should these dictionaries be unpacked before passing them
    into this function? Or should that be done here before the vectorized
    forward modeling procedure? Note, all of these dictionaries have
    identical keys for their values. Strings in the format "idsec101i000",
    "idsec101i001", "idsec101i002" etc.


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
    # Define a partial function that fixes the static parameters.
    fm_single = partial(fm_from_eigen_single,
                        input_img_shape=input_img_shape,
                        IOWA=IOWA,
                        aligned_center=aligned_center,
                        numbasis=numbasis,
                        # model_disks=model_disks,
                        aligned_imgs=)
    # Use vmap to vectorize fm_single over the first axis of all per-section arrays.
    # In this example, we assume that all per-section arrays have their first dimension equal to the number of sections.
    # For input_img_nums and PAs, ensure they are passed appropriately (e.g., one value per section).
    # Here, we set in_axes=0 for each per-section parameter.
    fm_outputs = jax.vmap(fm_single,
                            in_axes=(None, None, 0, 0, 0, 0, 0, 0) 
                            )(model_disks, aligned_imgs, klmodes_arr, evals_arr, evecs_arr,
                              section_ind_arr, PAs, input_img_nums,)
    
    fm_out = jnp.nanmean(fm_outputs,axis=1)

    return fm_out