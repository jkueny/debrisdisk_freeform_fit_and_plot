import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import legendre, jn

import numba

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def orthogonalize_bessel_basis(x, nmodes, npts):
    # n = len(x)
    x = np.linspace(x[0],x[-1],npts)
    basis_matrix = np.zeros((npts, nmodes))
    
    # Step 1: Populate the basis matrix with the first `nmodes` Bessel functions
    for i in range(nmodes):
        basis_matrix[:, i] = jn(i, x)
    
    # Step 2: Apply Gram-Schmidt orthogonalization
    ortho_basis = np.zeros_like(basis_matrix)
    for i in range(nmodes):
        # Start with the current Bessel function
        vec = basis_matrix[:, i]
        for j in range(i):
            # Subtract the projection onto the previous orthogonalized vectors
            vec -= np.dot(ortho_basis[:, j], basis_matrix[:, i]) * ortho_basis[:, j]
        # Normalize the vector
        ortho_basis[:, i] = vec / np.linalg.norm(vec)
        # max_val = np.max(np.abs(ortho_basis[:, i]))
        # ortho_basis[:, i] /= max_val
    
    return ortho_basis


def calculate_hg_spf(g1,g2,alpha1,scattangs):
    '''
    g1: First HG scattering parameter (forward-scattering)
    g2: Second HG scattering parameter (back-scattering)
    alpha1: Weighting factor for second scattering parameter
    scattangs: numpy array of scattering angles

    returns: SPF normalized to 1 at the disk ansae (90 deg scattang). Not mean-subtracted.
    '''
    scattering_angles = scattangs
    spf = []
    #Constant for HG function
    k = 1. / (4 * np.pi)
    for a in scattering_angles:
        cos_phi = np.cos(np.radians(a))
        #Henyey Greenstein function
        g1_2 = g1 * g1
        g2_2 = g2 * g2
        hg1 = k * alpha1 * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
        hg2 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2 -
                                                (2 * g2 * cos_phi))**1.5

        hg = hg1 + hg2
        spf.append(hg)
    #Henyey Greenstein function at 90
    hg1_90 = k * alpha1 * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2)**1.5

    hg_90 = hg1_90 + hg2_90
    spf = np.asarray(spf)
    spf /= hg_90
    print(f"The mean of the best-fit HG SPF: {np.mean(spf)}")
    return spf

def fourier_series(x, *a):
    """
    Fourier series model. 'a' contains the coefficients of the series:
    a[0] is the constant term, a[1] and a[2] are the coefficients of the first sine and cosine terms, respectively, and so on.
    'period' is the known period of the data.
    """
    x_rad = np.deg2rad(x)
    y = a[0]
    for i in range(1, len(a), 2):
        n = i // 2 + 1
        y += a[i] * np.cos(n * x_rad) + a[i+1] * np.sin(n * x_rad)
    return y

def orthogonalize_bessel_basis(x, nmodes, npts):
    # n = len(x)
    x = np.linspace(x[0],x[-1],npts)
    basis_matrix = np.zeros((npts, nmodes))
    
    # Step 1: Populate the basis matrix with the first `nmodes` Bessel functions
    for i in range(nmodes):
        basis_matrix[:, i] = jn(i, x)
    
    # Step 2: Apply Gram-Schmidt orthogonalization
    ortho_basis = np.zeros_like(basis_matrix)
    for i in range(nmodes):
        # Start with the current Bessel function
        vec = basis_matrix[:, i]
        for j in range(i):
            # Subtract the projection onto the previous orthogonalized vectors
            vec -= np.dot(ortho_basis[:, j], basis_matrix[:, i]) * ortho_basis[:, j]
        # Normalize the vector
        ortho_basis[:, i] = vec / np.linalg.norm(vec)
        # max_val = np.max(np.abs(ortho_basis[:, i]))
        # ortho_basis[:, i] /= max_val
    
    return ortho_basis

# @numba.njit
def legendre_reconstruction(x, *a):
    nmodes = len(a)
    n = len(x)
    x_fit = np.linspace(-1, 1, n)
    # basis_matrix = np.zeros((n, nmodes))
    
    # for i in range(nmodes):
    #     basis_matrix[:, i] = legendre(i)(x_fit)
    
    # # coefficients = np.linalg.lstsq(basis_matrix, spf_meansub, rcond=None)[0]
    
    reconstruction = np.zeros_like(x,dtype=np.float64)
    for i in range(nmodes):
        reconstruction += a[i] * legendre(i)(x_fit)
    
    return reconstruction

@numba.njit
def bessel_reconstruction(x, ortho_basis, *a):
    nmodes = len(a) - 1
    # n = len(x)
    reconstruction = np.ones_like(x) * a[0] #apply the DC offset
    for i in range(nmodes):
        reconstruction += a[i+1] * ortho_basis[:,i]
    
    return reconstruction

def fit_fourier_to_hg_spf(savedir,g1,g2=0.0,alpha1=0.0,n=10):
    scattering_angles = np.arange(13,167,1)
    spf = []
    #Constant for HG function
    k = 1. / (4 * np.pi)
    for a in scattering_angles:
        cos_phi = np.cos(np.radians(a))
        #Henyey Greenstein function
        g1_2 = g1 * g1
        g2_2 = g2 * g2
        hg1 = k * alpha1 * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
        hg2 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2 -
                                                (2 * g2 * cos_phi))**1.5

        hg = hg1 + hg2
        spf.append(hg)
    #Henyey Greenstein function at 90
    hg1_90 = k * alpha1 * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2)**1.5

    hg_90 = hg1_90 + hg2_90
    spf /= hg_90
    spf_meansub = spf - spf.mean()

    # Initial guess for coefficients (a0, a1, b1, a2, b2, ..., an, bn)
    initial_guess = np.zeros(2 * n + 1)

    # # Fit model to data
    coeffs, _ = curve_fit(fourier_series, scattering_angles, spf_meansub, p0=initial_guess)
    coeffs[0] = coeffs[0] + spf.mean()
    coeffs = np.round(coeffs,decimals=2)
    # Fit the model
    # params, params_covariance = curve_fit(lambda x, *a: fourier_series(x, *a, period=T), scattering_angles, spf_meansub, p0=initial_guess)


    # # Generate fitted curve
    x_fit = np.linspace(min(scattering_angles), max(scattering_angles), 1000)
    y_fit = fourier_series(x_fit, *coeffs)

    # # Generate fitted curve
    # y_fit = fourier_series(x_fit, *params, period=T)

    # Visualization
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(scattering_angles, spf, 'b.', label='Data')
    plt.plot(x_fit, np.abs(y_fit), 'r-', label='Fitted Model')
    plt.title(f'SPF and Fitted Fourier Series Model, order: {n}')
    plt.xlabel('Scattering angle [deg]')
    plt.ylabel('SPF')
    plt.grid()
    # plt.yscale('log')
    plt.legend()

    # Fitted Fourier coefficients
    plt.subplot(2, 1, 2)
    # coeffs = coeffs[1:]  # Exclude the constant term for the coefficient plot
    plt.bar(range(1, len(coeffs)+1), np.abs(coeffs), tick_label=[f"{i}" for i in range(1, len(coeffs)+1)])
    plt.title('Magnitude of Fitted Fourier Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{savedir}/fourierfit_spf_hg.png')
    # plt.show()
    return coeffs #initial coeffs for SPF fit

def fit_legendre_to_hg_spf(savedir,spftofit,scattangstofit,nmodes):
    nmodes += 1 #add 1 b/c we ignore the DC offset (Leg0)
    spf = np.asarray(spftofit)
    spf_meansub = spf #- spf.mean()
    scattering_angles = np.asarray(scattangstofit)
    n = len(scattering_angles)
    x = np.linspace(-1, 1, n)
    basis_matrix = np.zeros((n, nmodes))
    
    for i in range(nmodes):
        basis_matrix[:, i] = legendre(i)(x)
    
    coefficients = np.linalg.lstsq(basis_matrix, spf_meansub, rcond=None)[0]
    dc_off = coefficients[0]
    x_fit = np.linspace(-1, 1, len(scattering_angles)*10)
    model_scattangs = np.linspace(min(scattering_angles), max(scattering_angles), len(scattering_angles)*10)
    reconstructed_spf = np.zeros_like(x_fit)
    for i in range(nmodes):
        reconstructed_spf += coefficients[i] * legendre(i)(x_fit)

    # Visualization
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(scattering_angles, spf, 'b', label='Data', alpha=0.5,
             marker="o", markevery=20)
    plt.plot(model_scattangs, reconstructed_spf, 'r-', label='Fitted Model')
    plt.title(f'SPF and Fitted Model, Legendre basis, modes: {nmodes}')
    plt.xlabel('Scattering angle [deg]')
    plt.ylabel('SPF')
    plt.grid()
    # plt.yscale('log')
    plt.legend()

    # Fitted Fourier coefficients
    plt.subplot(2, 1, 2)
    # print(coeffs)
    # coeffs = coeffs[1:]  # Exclude the constant term for the coefficient plot
    plt.bar(range(len(coefficients)), np.abs(coefficients), tick_label=[f"{i}" for i in range(len(coefficients))])
    plt.title('Magnitude of Fitted Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{savedir}/legendrefit_spf_hg.png')
    plt.close()
    
    return coefficients

def fit_bessel_to_hg_spf(savedir, xrangemax, spftofit, scattangstofit, nmodes, dcoffset, orthogonalize=False):
    '''
    Orthogonal basis of Bessel functions of the first kind.
    '''
    spf = np.asarray(spftofit)
    spf_meansub = spf - dcoffset
    scattering_angles = np.asarray(scattangstofit)
    n = len(scattering_angles)
    x = np.linspace(0, xrangemax, n)
    # x = np.linspace(0, 20, n)
    # basis_matrix = np.zeros((n, 10))
    regparam = 0.0
    if orthogonalize:
        basis_matrix = orthogonalize_bessel_basis(x=x,nmodes=nmodes, npts=n)
    else: 
        basis_matrix = np.zeros((n, nmodes))
        for i in range(nmodes):
            basis_matrix[:, i] = jn(i,x)
    reg_matrix = regparam * np.eye(nmodes)  # Regularization term
    augmented_A = np.vstack([basis_matrix, reg_matrix])
    augmented_b = np.hstack([spf_meansub, np.zeros(nmodes)])  # Zero target for regularization term
    # coefficients = np.linalg.lstsq(basis_matrix, spf_meansub, rcond=None)[0]
    coefficients, res, rank, s = np.linalg.lstsq(augmented_A, augmented_b, rcond=None)
    print(f"Fitted {nmodes} Bessel components to the HG SPF with squared sum residuals {res}")
    print(f"Normalized by degrees of freedom: {res / (len(x) - nmodes)}")
    coefficients_all = np.append(dcoffset,coefficients)
    reconstructed_spf = bessel_reconstruction(x,basis_matrix,*coefficients_all)
    
    # x_fit = np.linspace(0, 10, n)
    model_scattangs = np.linspace(min(scattering_angles), max(scattering_angles), n)
    # reconstructed_spf = np.zeros_like(x)
    # for i in range(nmodes):
    #     # reconstructed_spf += coefficients[i] * jn(i,x_fit)
    #     print(coefficients[i])
    #     reconstructed_spf += coefficients[i] * basis_matrix[:,i]
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(scattering_angles, spf, 'b', label='Data',
             marker="o",markevery=20,alpha=0.5)
    plt.plot(model_scattangs, reconstructed_spf, 'r-', label='Fitted Model')
    plt.title(f'SPF and Fitted Model, Bessel basis, modes: {nmodes}')
    plt.xlabel('Scattering angle [deg]')
    plt.ylabel('SPF')
    plt.grid()
    # plt.yscale('log')
    plt.legend()

    # Fitted Fourier coefficients
    plt.subplot(2, 1, 2)
    # print(coeffs)
    # coeffs = coeffs[1:]  # Exclude the constant term for the coefficient plot
    plt.bar(range(len(coefficients)+1), np.abs(coefficients_all), tick_label=[f"{i}" for i in range(len(coefficients)+1)])
    plt.title('Magnitude of Fitted Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{savedir}/besselfit_spf_hg_meansub.png')
    plt.close()
    # Visualization 2: Verify orthogonality with a heatmap

    dot_product_matrix = np.dot(basis_matrix.T, basis_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(dot_product_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar(label="Dot Product Value")
    plt.title("Orthogonality Check: Dot Product of Basis Components")
    plt.xlabel("Basis Index")
    plt.ylabel("Basis Index")
    plt.xticks(range(nmodes))
    plt.yticks(range(nmodes))
    plt.grid(False)
    plt.savefig(f'{savedir}/Grammat_ortho_basis_check.png')
    plt.close()
    
    # return np.log(coefficients + cshift)
    return coefficients, basis_matrix