import h5py
import jax.numpy as jnp
import numpy as np


##############################################################################
###### routines to save and load h5 in dictionnaries
##############################################################################

def pad_array_to_fixed_first_dim(arr, fixed_first_dim, pad_value=0):
    """
    Pads a 2D array along axis 0 to have shape (fixed_first_dim, arr.shape[1]).
    Raises an error if the array is larger than fixed_first_dim.
    """
    current_first_dim = arr.shape[0]
    if current_first_dim > fixed_first_dim:
        raise ValueError(f"Array has {current_first_dim} rows, which exceeds the fixed limit of {fixed_first_dim}.")
    pad_rows = fixed_first_dim - current_first_dim
    if pad_rows > 0:
        if len(arr.shape) > 1:
            # The tuple is for padding arrays both before AND after the current values...!
            # Pad (before, after)
            pad_width = [(0, pad_rows), (0, 0)]
        else:
            pad_width = (0, pad_rows) # we only want to add values at the end.
        return jnp.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    else:
        return arr

def stack_with_padding_and_collect_pad_info(d, fixed_first_dim):
    """
    Given a dictionary d whose values are numeric arrays of shape (n_refs, n_modes)
    (with n_refs varying but n_modes constant), pads each array along axis 0 so that
    it becomes shape (fixed_first_dim, n_modes), and then stacks them along a new first axis.
    
    Returns a JAX array of shape (N_keys, fixed_first_dim, n_modes)
    and a pad_info dict mapping each key to its padding amount.
    """
    keys = sorted(d.keys())
    padded_arrays = []
    pad_info = {}
    for k in keys:
        arr = jnp.array(d[k])
        n_ref, n_modes = arr.shape
        pad0 = fixed_first_dim - n_ref
        padded_arr = pad_array_to_fixed_first_dim(arr, fixed_first_dim, pad_value=0)
        padded_arrays.append(padded_arr)
        pad_info[k] = {'pad_shape': (pad0, 0)}
    return jnp.stack(padded_arrays), pad_info

def build_batched_ref_images(aligned_images, ref_psfs_indicies_dict, fixed_refs, image_shape):
    """
    For each key in ref_psfs_indicies_dict, gathers the corresponding reference images
    from aligned_images (assumed to have shape (N_images, height, width)),
    flattens them, and pads the result along axis 0 to shape (fixed_refs, height*width).
    
    Returns a JAX array of shape (N_keys, fixed_refs, image_pixels).
    """
    keys = sorted(ref_psfs_indicies_dict.keys())
    batched_refs = []
    batched_inds = []
    # height, width = image_shape
    image_pixels = image_shape
    for k in keys:
        # Get reference indices for this key.
        ref_inds = np.asarray(ref_psfs_indicies_dict[k]).astype(int)
        # ref_inds shape (n_refs,)
        n_refs = len(ref_inds)
        if n_refs > fixed_refs:
            raise ValueError(f"For key {k}: number of references {n_refs} exceeds fixed limit {fixed_refs}.")
        # Gather reference images.
        refs = aligned_images[ref_inds]  # shape (n_refs, height, width)
        refs_flat = refs.reshape((refs.shape[0], image_pixels)) #shape (n_refs, n_pixels)
        refs_padded = pad_array_to_fixed_first_dim(refs_flat, fixed_refs, pad_value=0)
        inds_padded = pad_array_to_fixed_first_dim(ref_inds, fixed_refs, pad_value=-1)
        batched_inds.append(inds_padded)
        batched_refs.append(refs_padded)
    return jnp.stack(batched_refs), jnp.stack(batched_inds)

def build_batched_ref_PAs(global_PAs, ref_psfs_indicies_dict):
    """
    For each key in ref_psfs_indicies_dict, extract from klparam_dict["PAs"]
    the position angles corresponding to the reference indices.
    
    Returns a tuple of 1D JAX arrays (one per key), which may have varying lengths.
    No zero-padding is performed.
    """
    # global_PAs = jnp.array(klparam_dict["PAs"])
    PAs_arr = jnp.array(global_PAs)
    batched_PAs = []
    for k in ref_psfs_indicies_dict.keys():
        ref_inds = np.asarray(ref_psfs_indicies_dict[k]).astype(int)
        # Extract the position angles corresponding to these indices.
        image_PAs = PAs_arr[ref_inds]
        batched_PAs.append(image_PAs)
    return tuple(batched_PAs)

def apply_section_inds_to_image(aligned_image, section_inds):
    """
    Given a full aligned image (2D) and section_inds (a tuple of indices, e.g.
    (row_indices, col_indices)) for the region of interest,
    returns a flattened array of the selected pixels.
    
    Assumes all images yield the same number of pixels.
    """
    # Use the tuple of indices to index into the image.
    return aligned_image[section_inds]

def prepare_ref_angle_tensors(ref_PAs, min_num_models):
    """
    Turn a ragged list/tuple of 1-D JAX arrays into two dense tensors:
       ref_ang_pad : (N_global, min_num_models)   [float32]
       valid_mask  : (N_global, min_num_models)   [bool]
    The padding value itself is irrelevant because valid_mask tells JAX
    which entries to use.
    """
    N_global = len(ref_PAs)
    ref_ang_pad = np.zeros((N_global, min_num_models), dtype=np.float32)
    valid_mask  = np.zeros((N_global, min_num_models), dtype=np.bool_)
    for i, arr in enumerate(ref_PAs):
        L = len(arr)
        if L > min_num_models:
            raise ValueError(
                f"ref_PAs[{i}] has length {L} > min_num_models={min_num_models}"
            )
        ref_ang_pad[i, :L] = np.asarray(arr, dtype=np.float32)
        valid_mask [i, :L] = True
    return jnp.asarray(ref_ang_pad), jnp.asarray(valid_mask)

def unpack_basis_data(basis_data):
    """
    Unpacks the nested basis_data dictionary loaded from the H5 file into a set of JAX arrays,
    with automatic padding of variable-length arrays for the eigenvectors and the reference images.
    Also, the full aligned images are sectioned using section_inds.
    
    The function infers:
      - fixed_refs: the maximum number of references from evecs_dict.
      - image_shape: inferred from the first image in aligned_images_dict["wl1000"].
      - image_pixels: the number of pixels in the sectioned image (assumed same for all images).
    
    Returns a dictionary with the following keys:
      "aligned_images": JAX array of shape (N_keys, N_pixels_section) -- flattened, sectioned images.
      "evals": stacked JAX array from evals_dict.
      "klmodes": stacked JAX array from klmodes_dict.
      "evecs": JAX array of shape (N_keys, fixed_refs, n_modes)
      "input_img_nums": JAX int array of shape (N_keys,)
      "section_inds": tuple of section index values (as provided in basis_data).
      "ref_psfs": JAX array of shape (N_keys, fixed_refs, N_pixels_section)
      "ref_PAs": a tuple of 1D JAX arrays (padded) with the reference PAs.
      "klparams": original klparam_dict.
    """
    out = {}
    # Process aligned images.
    full_aligned_images = basis_data["aligned_images_dict"]["wl1000"]
    full_aligned_images = jnp.array(full_aligned_images)  # shape (N_images, H, W)
    
    # Get section_inds from basis_data (assumed to be a dict with one entry per image)
    keys = sorted(basis_data["section_ind_dict"].keys())
    # Here we assume each value is a tuple of two arrays (row_inds, col_inds)
    section_inds_list = [basis_data["section_ind_dict"][k] for k in keys]
    out["section_inds"] = tuple(section_inds_list)
    
    # For each image, select only the region of interest.
    # This returns a flattened array per image.
    reduced_images_list = []
    for i, inds in enumerate(section_inds_list):
        # Convert inds to tuple if needed.
        if not isinstance(inds, tuple):
            # If the section indices are stored in another format, convert as needed.
            inds = tuple(inds)
        reduced = apply_section_inds_to_image(full_aligned_images[i], inds)
        reduced_images_list.append(reduced)
    reduced_aligned_images = jnp.stack(reduced_images_list)  # shape (N_keys, N_pixels_section)
    out["aligned_images"] = reduced_aligned_images
    
    # Infer the number of pixels in the section.
    N_pixels_section = reduced_aligned_images.shape[1]
    
    # Determine fixed_refs from evecs_dict.
    n_refs_list = [np.array(basis_data["evecs_dict"][k]).shape[0] for k in basis_data["evecs_dict"].keys()]
    fixed_refs = int(max(n_refs_list))
    out["fixed_refs"] = fixed_refs
    
    # For evals_dict and klmodes_dict, assume consistent shapes; stack them.
    def stack_from_dict(d):
        keys = sorted(d.keys())
        arrays = [jnp.array(d[k]) for k in keys]
        return jnp.stack(arrays)
    
    out["evals"] = stack_from_dict(basis_data["evals_dict"])
    out["klmodes"] = stack_from_dict(basis_data["klmodes_dict"])
    
    # For evecs_dict: pad each array to (fixed_refs, n_modes) and stack.
    evecs_stacked, _ = stack_with_padding_and_collect_pad_info(basis_data["evecs_dict"], fixed_refs)
    out["evecs"] = evecs_stacked
    
    # For input_img_num_dict: convert values to integers.
    keys_nums = sorted(basis_data["input_img_num_dict"].keys())
    input_img_nums = [int(basis_data["input_img_num_dict"][k]) for k in keys_nums]
    out["input_img_nums"] = jnp.array(input_img_nums, dtype=jnp.int32)
    
    # For ref_psfs_indicies_dict: build batched reference images from the reduced (sectioned) images.
    ref_psfs, ref_inds = build_batched_ref_images(reduced_aligned_images, basis_data["ref_psfs_indicies_dict"], fixed_refs, N_pixels_section)
    out["ref_psfs"] = ref_psfs  # shape (N_keys, fixed_refs, N_pixels_section)
    
    # For ref_PAs: build ragged reference position angles (no padding).
    ref_PAs = build_batched_ref_PAs(basis_data["klparam_dict"]["PAs"],
                                    basis_data["ref_psfs_indicies_dict"])
    out["ref_PAs"] = ref_PAs  # tuple of 1D JAX arrays (ragged)

    out["ref_inds"] = ref_inds
    if "wdhPAs_dict" in basis_data.keys():
        wdhPAs = jnp.array(basis_data["wdhPAs_dict"]["PAs"])
        out["wdhPAs"] = wdhPAs
        out["PAmask"] = jnp.array(basis_data["wdhPAs_dict"]["PAmask"])
    

    
    # klparam_dict remains unchanged.
    out["klparams"] = basis_data["klparam_dict"]
    
    # print(f"Inferred fixed_refs: {fixed_refs}")
    # print(f"Inferred sectioned image pixel count: {N_pixels_section}")
    # pa_lengths = [arr.shape[0] for arr in ref_PAs]
    # print(f"Reference PAs lengths per key: {pa_lengths}")
    
    return out

def recursive_convert_to_jax(obj):
    """
    Recursively convert numeric numpy arrays (or nested lists/tuples) in obj
    to JAX arrays. Leaves strings and other non-numeric objects unchanged.
    """
    if isinstance(obj, dict):
        return {k: recursive_convert_to_jax(v) for k, v in obj.items()}
        # return (recursive_convert_to_jax(obj[k]) for k in obj.keys())
    elif isinstance(obj, list):
        return [recursive_convert_to_jax(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_convert_to_jax(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        # Only convert if the array is numeric; leave arrays of strings as is.
        if obj.dtype.kind in ('U', 'S'):
            return obj
        else:
            return jnp.array(obj)
    else:
        return obj

def load_kl_basis(h5_path):
    """
    Reads an h5 file and returns a dictionary with selected keys,
    converting the arrays of interest to JAX arrays.

    Args:
        h5_path (str): Path to the h5 file containing the forward model klbasis.

    Returns:
        dict: A dictionary with the following keys:
            - "aligned_images_dict": converted to a JAX array.
            - "evals_dict": converted to a JAX array.
            - "evecs_dict": converted to a JAX array.
            - "input_img_num_dict": converted to a JAX array.
            - "klmodes_dict": converted to a JAX array.
            - "klparams_dict": left as is (non-JAX array).
            - "section_ind_dict": left as is (e.g. a list or array of strings).
    """
    # Read the full dictionary from the h5 file.
    klbasis = _load_dict_from_hdf5(h5_path)

    output = {}

    # Keys to convert to JAX arrays.
    convert_keys = [
        "aligned_images_dict",
        "evals_dict",
        "evecs_dict",
        "input_img_num_dict",
        "klmodes_dict",
        "section_ind_dict",
        "ref_psfs_indicies_dict",
    ]
    # Special keys: leave as non-JAX arrays.
    special_keys = [
        "klparam_dict",
        "wdhPAs_dict"
    ]

    for key in klbasis:
        if key in convert_keys:
            output[key] = recursive_convert_to_jax(klbasis[key])
        elif key in special_keys:
            # Leave these keys unchanged (or process them separately if desired)
            output[key] = klbasis[key]
        else:
            # Optionally process or ignore other keys
            continue
            # output[key] = klbasis[key]
    
    return output

def _save_dict_to_hdf5(dic, filename):
    """
    Saving a nested dictionnary into a h5 file

    Args:
        dic: the dictionnary to file
        filename: the filename of the h5 where it will be saved

    Returns:
        None

    """
    with h5py.File(filename, "w") as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)


def _load_dict_from_hdf5(filename):
    """
    Load a dictionnary from a h5 file

    Args:
        filename: the filename of the h5

    Returns:
        the dictionnary exctracted

    """

    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively explore the dictionnary for saving it

    Args:
        h5file: the file in which we save, opened with h5py.File
        path: the separator to aggregate the keys. Should not be set to a value that
            is likely to be in the dictionnary keys already
        dic: the dictionnary to deconstruct

    Returns
        None

    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/',
                                                     item)
        else:
            raise ValueError("Cannot save {0} type in h5 (key = {1})".format(
                type(item), path + key))


def _recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively explore the dictionnary for loading it

    Args:
        h5file: the file from which we load, opened with h5py.File
        path: the separator to aggregate the keys. Should be the same one used in
        _recursively_save_dict_contents_to_group
    Returns
        the rebuilt dictionnary
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

if __name__ == "__main__":
    from klip_basis import load_kl_basis  # adjust the import as needed
    h5_path = "/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/norm_lite_psflib/klip_fm_files/camsci2_z_20230309_10_klbasis.h5"
    basis_data = load_kl_basis(h5_path)

    # print(basis_data["ref_psfs_indicies_dict"])
    unpacked_data = unpack_basis_data(basis_data)  # previous version for comparison

    ####
    # import matplotlib.pyplot as plt
    # test_zero_padded_flat = np.asarray(unpacked_data["ref_psfs"][0][0])
    # test_zero_padded_image = np.reshape(test_zero_padded_flat, (224, 224))
    # plt.imshow(test_zero_padded_image, cmap="viridis")
    # plt.show()
    ####

    # For demonstration, print out the inferred shapes:
    # print(f"Reference PAs: {unpacked_data["ref_PAs"]}")
    # print("Aligned images shape:", unpacked_data["aligned_images"].shape)  # e.g. (84, 224, 224)
    # print("Evecs shape:", unpacked_data["evecs"].shape)                    # e.g. (84, fixed_refs, n_modes)
    # print("Ref PSFs shape:", unpacked_data["ref_psfs"].shape)                # e.g. (84, fixed_refs, 224*224)
    # print("Input image numbers:", unpacked_data["input_img_nums"])
    print("Section indices:", len(unpacked_data["section_inds"]))