import jax
import jax.numpy as jnp
from functools import partial
import dev.pyklip.j_fm as jfm
from dev.pyklip.j_klip import rotate_image

# Example helper function: processes one section/image.
# All per-section parameters are passed as arguments.
def fm_from_eigen_single(klmodes, evals, evecs, input_img_shape,
                         input_img_num, ref_psfs_indicies, section_ind,
                         radstart, radend, phistart, phiend,
                         padding, IOWA, ref_center, parang,
                         numbasis, fmout, flipx, mode, model_disks, aligned_imgs):
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
    delta_KL = jnp.where(mode == 'RDI',
                         klmodes * 0.0,
                         jfm.perturb_specIncluded(evals, evecs, klmodes, refs,
                                                   model_ref_nonan,
                                                   return_perturb_covar=False))
    # Calculate the post-KLIP PSF using your forward modeling routine.
    postklip_psf, _, _ = jfm.calculate_fm(delta_KL, klmodes, numbasis,
                                          sci, model_sci_nonan,
                                          inputflux=None)
    # Save the rotated section.
    output_img = jfm._save_rotated_section(
        input_img_shape,
        postklip_psf,
        section_ind,
        fmout,  # Assuming fmout is preallocated for each section
        parang,
        radstart,
        radend,
        phistart,
        phiend,
        padding,
        IOWA,
        ref_center,
        flipx=flipx)
    return output_img

# Suppose inside your fm_from_eigen() method you have dictionaries
# keyed by section. The idea is to collect the per-section values into arrays.
# For example:
#
#    dict_keys = self.dict_keys
#    klmodes_arr = jnp.stack([jnp.array(self.klmodes_dict[k]) for k in dict_keys])
#    evals_arr   = jnp.stack([jnp.array(self.evals_dict[k]) for k in dict_keys])
#    evecs_arr   = jnp.stack([jnp.array(self.evecs_dict[k]) for k in dict_keys])
#    section_ind_arr = jnp.stack([jnp.array(self.section_ind_dict[k]) for k in dict_keys])
#    radstart_arr = jnp.stack([jnp.array(self.radstart_dict[k]) for k in dict_keys])
#    ... (and so on for the other per-section parameters)
#
# Other variables that are common to all sections (like input_img_shape, padding,
# IOWA, ref_center, flipx, mode, numbasis) can be passed directly (or marked as static).
#
# Also assume that self.model_disks and aligned_imgs are available JAX arrays.
#
# Then, you can define a vectorized version with vmap. For instance:

def fm_from_eigen_vectorized(klmodes_arr, evals_arr, evecs_arr, section_ind_arr,
                             radstart_arr, radend_arr, phistart_arr, phiend_arr,
                             input_img_nums,  # an array of image indices (one per section)
                             input_img_shape, padding, IOWA, ref_center,
                             PAs, numbasis, fmout, flipx, mode,
                             model_disks, aligned_imgs):
    # Define a partial function that fixes the static parameters.
    fm_single = partial(fm_from_eigen_single,
                        input_img_shape=input_img_shape,
                        padding=padding,
                        IOWA=IOWA,
                        ref_center=ref_center,
                        numbasis=numbasis,
                        fmout=fmout,
                        flipx=flipx,
                        mode=mode,
                        model_disks=model_disks,
                        aligned_imgs=aligned_imgs)
    # Use vmap to vectorize fm_single over the first axis of all per-section arrays.
    # In this example, we assume that all per-section arrays have their first dimension equal to the number of sections.
    # For input_img_nums and PAs, ensure they are passed appropriately (e.g., one value per section).
    # Here, we set in_axes=0 for each per-section parameter.
    output_imgs = jax.vmap(fm_single,
                           in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0,  # per-section arrays
                                    None, None, None, None,    # static parameters
                                    0, 0)  # if PAs is per-section, or mark as static
                           )(klmodes_arr, evals_arr, evecs_arr, section_ind_arr,
                             radstart_arr, radend_arr, phistart_arr, phiend_arr,
                             input_img_nums,  # This could also be static if all sections use the same image index
                             PAs)
    return output_imgs

# In your fm_from_eigen() method, instead of looping over self.dict_keys,
# you would do something like:
#
#    # Convert dictionary entries to arrays:
#    klmodes_arr = jnp.stack([jnp.array(self.klmodes_dict[k]) for k in self.dict_keys])
#    evals_arr   = jnp.stack([jnp.array(self.evals_dict[k]) for k in self.dict_keys])
#    evecs_arr   = jnp.stack([jnp.array(self.evecs_dict[k]) for k in self.dict_keys])
#    section_ind_arr = jnp.stack([jnp.array(self.section_ind_dict[k]) for k in self.dict_keys])
#    radstart_arr = jnp.stack([jnp.array(self.radstart_dict[k]) for k in self.dict_keys])
#    radend_arr = jnp.stack([jnp.array(self.radend_dict[k]) for k in self.dict_keys])
#    phistart_arr = jnp.stack([jnp.array(self.phistart_dict[k]) for k in self.dict_keys])
#    phiend_arr = jnp.stack([jnp.array(self.phiend_dict[k]) for k in self.dict_keys])
#
#    # Also gather image indices (if each section corresponds to a specific image)
#    input_img_nums = jnp.array([self.input_img_num_dict[k] for k in self.dict_keys])
#
#    # Then call the vectorized function:
#    output_imgs = fm_from_eigen_vectorized(klmodes_arr, evals_arr, evecs_arr, section_ind_arr,
#                         radstart_arr, radend_arr, phistart_arr, phiend_arr,
#                         input_img_nums,
#                         input_img_shape=[self.inputs_shape[1], self.inputs_shape[2]],
#                         padding=0.0, IOWA=(self.IWA, self.OWA),
#                         ref_center=self.aligned_center,
#                         PAs=self.PAs,  # if PAs is an array with appropriate shape
#                         numbasis=self.numbasis,
#                         fmout=your_preallocated_fmout,  # as allocated by self.alloc_fmout()
#                         flipx=True, mode=mode,
#                         model_disks=self.model_disks,
#                         aligned_imgs=your_aligned_images_array)
#
# This replaces the for-loop over self.dict_keys with a single, vectorized call.
# Be sure that all arrays passed to vmap have a static (compile-time) shape in their first dimension.
