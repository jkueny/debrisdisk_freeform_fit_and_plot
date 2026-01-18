import numpy as np
import numba

@numba.njit
def manual_dot_product(A, B):
    # Assuming A and B are 2D arrays with compatible dimensions for dot product
    result = np.empty((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum_product = 0.0
            for k in range(A.shape[1]):
                sum_product += A[i, k] * B[k, j]
            result[i, j] = sum_product
    return result

@numba.njit
def create_evals_tiled(evals, max_basis):
    # Initialize an empty array of the same shape you'd get with np.tile()
    evals_tiled = np.empty((max_basis, evals.shape[0]))

    # Manually fill in the array to mimic tiling
    for i in range(max_basis):
        for j in range(evals.shape[0]):
            evals_tiled[i, j] = evals[j]

    return evals_tiled

@numba.njit
def perturb_specIncluded(evals, evecs, original_KL, refs_meansub, modelrefs_meansub):
    """
    Perturb the KL modes using a model of the PSF but with the spectrum included in the model. Quicker than the others

    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [p, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, p])
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images.
                    Each model should contain spectral informatoin
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes. Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    # N_ref = refs.shape[0]
    # N_pix = original_KL.shape[1]

    # refs_mean_sub = refs# - np.nanmean(refs, axis=1)[:, None]
    # refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    # models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    #print(evals.shape,evecs.shape,original_KL.shape,refs.shape,models_ref.shape)

    evals_tiled = create_evals_tiled(evals, max_basis)
    np.fill_diagonal(evals_tiled,np.nan)
    evals_sqrt = np.sqrt(evals)
    evalse_inv_sqrt = 1./evals_sqrt
    # Manually calculate evals_ratio using the custom dot product function
    evalse_inv_sqrt_reshaped = evalse_inv_sqrt.reshape((-1, 1))  # Reshape for manual dot product
    evals_sqrt_reshaped = evals_sqrt.reshape((1, -1))  # Reshape for manual dot product
    evals_ratio = manual_dot_product(evalse_inv_sqrt_reshaped, evals_sqrt_reshaped)
    # evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
    #print(evals)
    # beta_tmp[np.diag_indices(np.size(evals))] = -0.5/evals
    for i in range(evals.shape[0]):  # Manual diagonal modification
        beta_tmp[i, i] = -0.5 / evals[i]
    beta = evals_ratio*beta_tmp

    # C_partial = modelrefs_meansub.dot(refs_meansub.transpose())
    C_partial = manual_dot_product(modelrefs_meansub,refs_meansub.transpose())
    C = C_partial+C_partial.transpose()
    #C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha = (evecs.transpose()).dot(C).dot(evecs)
    alpha_tmp = manual_dot_product(evecs.transpose(),C)
    alpha = manual_dot_product(alpha_tmp,evecs)

    # delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(modelrefs_meansub)
    delta_KL_a = manual_dot_product((beta*alpha),original_KL)
    delta_KL_b = manual_dot_product(evalse_inv_sqrt.reshape((len(evals), 1))*evecs.transpose(),modelrefs_meansub)
    delta_KL = delta_KL_a + delta_KL_b

    return delta_KL