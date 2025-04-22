# pylint: disable=C0103

####### This is the MCMC plotting code for HR 4796 data #######
import sys
import os

basedir = f"{os.environ['HOME']}/projects"  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).
# basedir = '/Users/jmazoyer/Dropbox/Work/python/python_data/disk_mcmc/spie_paper/hr4796 like/'

# default_parameter_file = 'FakeHr4796faint_MCMC_RDI.yaml'
# default_parameter_file = 'FakeHr4796bright_MCMC_ADI.yaml'

# default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'
# default_parameter_file = 'HR4796_i_camsci1_20230309_10.yaml'
# default_parameter_file = 'HR4796_r_camsci1_20230312_13.yaml'
# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'
default_parameter_file = 'HR4796_RDI_z_20240328_29.yaml'


import warnings
import argparse

from datetime import datetime

# import math as mt
import numpy as np
from scipy.ndimage import rotate

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines

import yaml

import corner
from emcee import backends, autocorr

from numba.core.errors import NumbaWarning


# from anadisk_model.anadisk_sum_mask import phase_function_spline, generate_disk

from utils.disk_models import fastgen_disk_dxdy_custom, fastmodgen_disk_custom

from utils.spf_models import calculate_hg_spf, \
                        fit_fourier_to_hg_spf, fit_legendre_to_hg_spf, \
                        legendre_reconstruction, bessel_reconstruction, \
                        fit_bessel_to_hg_spf

# from kowalsky import kowalsky

import modelfit_physical_to_freeform
from modelfit_physical_to_freeform import arr_free_params

# plt.switch_backend('agg')

# There is a conflict when I import
# matplotlib with pyklip if I don't use this line

def plot_ml_custom_spf(coeffs,scplotrange, scattangs, dcterm, errupper, errlower):
    scattering_angles = scattangs
    # print(coeffs)
    if basis == "bessel":
        bestfit_coefficients = np.append(np.asarray(dcterm),np.asarray(coeffs))
        # bestfit_coefficients = coeffs
        spf = bessel_reconstruction(x_tofit,select_basis,*bestfit_coefficients)
        # Sampling coefficients
        n_samples = 5000
        sampled_coeffs = sample_pd_asymmetric(bestfit_coefficients, errlower, errupper, n_samples)
        # Generate model curves for all sampled coefficients
        model_curves = []
        for scfs in sampled_coeffs:
            # Compute percentiles for uncertainty bands
            model_curves.append(bessel_reconstruction(x_tofit, select_basis, *scfs))
        model_curves = np.asarray(model_curves)
        err_lower_bound = np.percentile(model_curves, 16, axis=0)
        median_curve = np.percentile(model_curves, 50, axis=0)
        err_upper_bound = np.percentile(model_curves, 84, axis=0)
    elif basis == "legendre":
        bestfit_coefficients = coeffs
        spf = legendre_reconstruction(scattering_angles, *bestfit_coefficients)
        err_upper_bound = legendre_reconstruction(scattering_angles, *errupper)
        err_lower_bound = legendre_reconstruction(scattering_angles, *errlower)
    print(f"Best coeffs: {bestfit_coefficients}")
    print(f"Penalty incurred: {-lambda_reg * np.sum(bestfit_coefficients**2)}")
    spf_90 = spf[np.argmin(np.abs(scattering_angles - 90))]
    low_err_90 = err_lower_bound[np.argmin(np.abs(scattering_angles - 90))]
    upp_err_90 = err_upper_bound[np.argmin(np.abs(scattering_angles - 90))]
    # spf /= spf_90
    spf -= (spf_90 - 1)
    err_lower_bound -= (low_err_90 - 1)
    err_upper_bound -= (upp_err_90 - 1)
    angs_of_interest = np.where((scattering_angles < scplotrange[1]) & (scattering_angles > scplotrange[0]))
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    # plt.plot(scattering_angles, spf, 'r-', label='Best fit SPF via Legendre basis')
    plt.plot(scattering_angles[angs_of_interest], spf[angs_of_interest], 'r-', label=f'Best fit SPF via {basis} basis')
    plt.fill_between(scattering_angles[angs_of_interest], err_lower_bound[angs_of_interest], err_upper_bound[angs_of_interest],
                     alpha=0.4, color="red")
    plt.plot(scattering_angles[angs_of_interest], spf_tofit[angs_of_interest], "b-", label="Best-fit HG SPF")
    plt.title(f'SPF and Fitted Model, order: {len(coeffs)}')
    plt.xlabel('Scattering angle [deg]')
    plt.ylabel('SPF')
    plt.grid()
    # plt.yscale('log')
    plt.legend()

    # Fitted Fourier coefficients
    plt.subplot(2, 1, 2)
    # print(coeffs)
    # coeffs = bestfit_coefficients[1:]  # Exclude the constant term for the coefficient plot
    plt.bar(range(len(bestfit_coefficients)), np.abs(bestfit_coefficients), 
            tick_label=[f"{i}" for i in range(len(bestfit_coefficients))])
    plt.title('Magnitude of Fitted Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{mcmcresultdir}/bestfit_spf_custom.png')
    plt.close()


def chains_to_params(chain, flatten=False):

    chain_param = chain * 0.

    n_iter = chain.shape[0]
    nwalkers = chain.shape[1]
    n_dim_mcmc = chain.shape[2]

    for i in range(n_iter):
        for j in range(nwalkers):
            _, chain_param[i, j, :] = modelfit_physical_to_freeform.from_theta_to_params(
                chain[i, j, :])

    if flatten:
        return_chain = np.zeros((n_iter * nwalkers, n_dim_mcmc))
        for i in range(n_dim_mcmc):
            return_chain[:, i] = chain_param[:, :, i].flatten()
    else:
        return_chain = chain_param

    return return_chain


########################################################
def crop_center_odd(img, crop):
    img[img != img] = 0.
    y, x = img.shape
    startx = (x - 1) // 2 - crop // 2
    starty = (y - 1) // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop]


########################################################
def offset_2_RA_dec(dx, dy, inclination, principal_angle, distance_star):
    """ right ascension and declination of the ellipse centre with respect to the star
        location from the offset in AU in the disk plane define by the max disk code

    Args:
        dx: offsetx of the star in AU in the disk plane define by the max disk code
            au, + -> NW offset disk plane Minor Axis
        dy: offsety of the star in AU in the disk plane define by the max disk code
            au, + -> SW offset disk plane Major Axis
        inclination: inclination in degrees
        principal_angle: prinipal angle in degrees

    Returns:
        [right ascension, declination]
    """

    dx_disk_mas = convert.au_to_mas(dx * np.cos(np.radians(inclination)),
                                    distance_star)
    dy_disk_mas = convert.au_to_mas(-dy, distance_star)

    dx_sky = np.cos(np.radians(principal_angle)) * dx_disk_mas - np.sin(
        np.radians(principal_angle)) * dy_disk_mas
    dy_sky = np.sin(np.radians(principal_angle)) * dx_disk_mas + np.cos(
        np.radians(principal_angle)) * dy_disk_mas

    dAlpha = -dx_sky
    dDelta = dy_sky

    return dAlpha, dDelta


########################################################
def make_chain_plot(params_mcmc_yaml):
    """ make_chain_plot reading the .h5 file from zeus

    It is structured:
    Group: "mcmc"/
    Dsets: "accepted", "chain", "log_prob"

    Query chains like:

    chains = hf["mcmc"]["chains"]

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    disk_model = params_mcmc_yaml["DISK_MODEL"]

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    iter = reader.iteration
    if iter < burnin - 1:
        burnin = 0
        params_mcmc_yaml['BURNIN'] = 0

    chain = reader.get_chain(discard=0, thin=thin)
    chain_flat = reader.get_chain(flat=True)
    blobs_flat = reader.get_blobs(flat=True)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)
    wheremin = np.where(
        log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    theta_ml = chain_flat[wheremin0, :]
    # print(log_prob_samples_flat)
    tau = reader.get_autocorr_time(tol=0)
    if burnin > reader.iteration - 1:
        raise ValueError(
            "the burnin cannot be larger than the # of iterations")
    print("")
    print("")
    print(name_h5)
    print(f"# of iteration in the backend chain initially: {reader.iteration}")
    print("Max Tau times 50: {0}".format(50 * np.max(tau)))
    print("")

    print("Maximum Likelyhood: {0}".format(np.nanmax(log_prob_samples_flat)))

    print("burn-in: {0}".format(burnin))
    print("chain shape: {0}".format(chain.shape))

    n_dim_mcmc = chain.shape[2]
    nwalkers = chain.shape[1]

    if basis == "legendre" or basis == "bessel":
        n_dim_geo = int(len(free_params))
        _, axspf = plt.subplots(n_dim_mcmc - n_dim_geo,
                        sharex=True,
                        figsize=(6.4 * quality_plot, 4.8 * quality_plot))
        coeff_idx = 1
        for i in range(n_dim_geo, n_dim_mcmc):
                # print(i)
                axspf[i - n_dim_geo].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)
                axspf[i - n_dim_geo].tick_params(axis='y', labelsize=4 * quality_plot)
                axspf[i - n_dim_geo].set_ylabel(f'a{coeff_idx}', fontsize=5 * quality_plot)

                for j in range(nwalkers):
                    axspf[i - n_dim_geo].plot(np.abs(chain[:, j, i]), linewidth=quality_plot)
                coeff_idx += 1

        axspf[n_dim_mcmc - n_dim_geo - 1].tick_params(axis='x', labelsize=6 * quality_plot)
        axspf[n_dim_mcmc - n_dim_geo - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

        plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains_coeffs.jpg'))
        plt.close()
    if disk_model == "original":
        for i in range(n_dim_geo):
            if names[i] == 'R1':
                print(f'R1: {np.exp(theta_ml[i])}')
            elif names[i] == 'R2':
                print(f'R2: {np.exp(theta_ml[i])}')
            elif names[i] == 'Norm':
                print(f'Norm: {np.exp(theta_ml[i])}')
            elif names[i] == 'rscale':
                print(f'rscale: {np.exp(theta_ml[i])}')
            else:
                print(f'{names[i]}: {theta_ml[i]}')
    elif disk_model == "modified":
        for i in range(n_dim_geo):
            if names[i] == 'RC':
                print(f'RC: {np.exp(theta_ml[i])}')
            elif names[i] == 'Norm':
                print(f'Norm: {np.exp(theta_ml[i])}')
            elif names[i] == 'rscale':
                print(f'rscale: {np.exp(theta_ml[i])}')
            else:
                print(f'{names[i]}: {theta_ml[i]}')
    print("From make_chain_plot(), theta_ml...")
    print(theta_ml)


    modelfit_physical_to_freeform.SPF_MODEL = params_mcmc_yaml[
        'SPF_MODEL']  #Type of description for the SPF
    chain = chains_to_params(chain)

    _, axarr = plt.subplots(n_dim_geo,
                            sharex=True,
                            figsize=(6.4 * quality_plot, 4.8 * quality_plot))
    for i in range(n_dim_geo):
        axarr[i].set_ylabel(labels[names[i]], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)

        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)

        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[n_dim_geo - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[n_dim_geo - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains.jpg'))
    plt.close()

def sample_pd_asymmetric(coeffs, lower_sigma, upper_sigma, nsamples):
    samples = []
    for coeff, lower, upper in zip(coeffs, lower_sigma, upper_sigma):
        # Determine sign of the coefficient
        sign = np.sign(coeff)
        abs_coeff = abs(coeff)
        abs_lower = np.abs(lower)
        abs_upper = np.abs(upper)
        
        # Generate samples for the absolute value
        sample = np.random.normal(abs_coeff, abs_lower, nsamples)  # Default lower spread
        mask = np.random.rand(nsamples) > 0.5  # Flip a coin
        sample[mask] = np.random.normal(abs_coeff, abs_upper, np.sum(mask))  # Replace with upper spread
        
        # Restore the sign
        sample *= sign
        samples.append(sample)
    return np.array(samples, dtype=float).T


########################################################
def make_corner_plot(params_mcmc_yaml):
    """ make corner plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    band_name = params_mcmc_yaml['BAND_NAME']
    modelfit_physical_to_freeform.SPF_MODEL = params_mcmc_yaml[
        'SPF_MODEL']  #Type of description for the SPF

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = chains_to_params(chain, flatten=True)
    n_dim_mcmc = chain_flat.shape[1]
    if spf_model == "adp":
        n_dim_geo = int(len(free_params)) + 1
    else:
        n_dim_geo = int(len(free_params))
    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros(
            (len(chain4thatparam[wherenotnan]), n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:, i] = chain_flat[wherenotnan, i]
        chain_flat = chainflatnonan

    rcParams['axes.labelsize'] = 19
    rcParams['axes.titlesize'] = 14

    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13

    ### cumulative percentiles
    ### value at 50% is the center of the Normal law
    ### value at 50% - value at 15.9% is -1 sigma
    ### value at 84.1%% - value at 50% is 1 sigma
    if sigma == 1:
        quants = (0.159, 0.5, 0.841)
    if sigma == 2:
        quants = (0.023, 0.5, 0.977)
    if sigma == 3:
        quants = (0.001, 0.5, 0.999)

    #### Check truths = bests parameters

    shouldweplotalldatapoints = False

    labels_hash = [labels[names[i]] for i in range(n_dim_geo)]
    fig = corner.corner(chain_flat[:,:n_dim_geo],
                        labels=labels_hash,
                        quantiles=quants,
                        show_titles=True,
                        plot_datapoints=shouldweplotalldatapoints,
                        verbose=False)

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(band_name,
                       xy=(0.55, 0.99),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("{0:,} iterations (+ {1:,} burn-in)".format(
        reader.iteration - burnin, burnin),
                       xy=(0.55, 0.95),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("with {0:,} walkers: {1:,} models".format(
        nwalkers, reader.iteration * nwalkers),
                       xy=(0.55, 0.91),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs.pdf'))
    plt.close()

    if basis == "bessel" or basis == "legendre":
        labels_coeffs = [f'a{j}' for j in range(1,n_dim_mcmc - n_dim_geo + 1)]
        fig_spf = corner.corner(chain_flat[:,n_dim_geo:n_dim_mcmc],
                            labels=labels_coeffs,
                            quantiles=quants,
                            show_titles=True,
                            plot_datapoints=shouldweplotalldatapoints,
                            verbose=False)
    
        fig_spf.subplots_adjust(hspace=0)
        fig_spf.subplots_adjust(wspace=0)

        fig_spf.gca().annotate(band_name,
                        xy=(0.55, 0.99),
                        xycoords="figure fraction",
                        xytext=(-20, -10),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=44)

        fig_spf.gca().annotate("{0:,} iterations (+ {1:,} burn-in)".format(
            reader.iteration - burnin, burnin),
                        xy=(0.55, 0.95),
                        xycoords="figure fraction",
                        xytext=(-20, -10),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=44)

        fig_spf.gca().annotate("with {0:,} walkers: {1:,} models".format(
            nwalkers, reader.iteration * nwalkers),
                        xy=(0.55, 0.91),
                        xycoords="figure fraction",
                        xytext=(-20, -10),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=44)

        plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs_coeffs.pdf'))
        plt.close()


########################################################
def create_header(params_mcmc_yaml):
    """ measure all the important parameters and exctract their error bars
        and print them and save them in a hdr file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        header for all the fits
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    comments = params_mcmc_yaml['COMMENTS']
    names = params_mcmc_yaml['NAMES']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']

    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    modelfit_physical_to_freeform.SPF_MODEL = params_mcmc_yaml[
        'SPF_MODEL']  #Type of description for the SPF
    modelfit_physical_to_freeform.DISK_MODEL = params_mcmc_yaml[
        'DISK_MODEL']  #Type of description for the SPF

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = chains_to_params(chain, flatten=True)

    n_dim_mcmc = chain_flat.shape[1]

    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros(
            (len(chain4thatparam[wherenotnan]), n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:, i] = chain_flat[wherenotnan, i]
        chain_flat = chainflatnonan
        log_prob_samples_flat = log_prob_samples_flat[wherenotnan]

    samples_dict = dict()
    comments_dict = comments
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(names[:n_dim_mcmc]):
        samples_dict[key] = chain_flat[:, i]

    for i, key in enumerate(names[n_dim_mcmc:]):
        samples_dict[key] = chain_flat[:, i] * 0.

    # measure of 2 other parameters:  eccentricity and argument
    # of the perihelie
    # for modeli in range(chain_flat.shape[0]):
    #     r1_here = samples_dict['R1'][modeli]
    #     dx_here = samples_dict['dx'][modeli]
    #     dy_here = samples_dict['dy'][modeli]
    #     a = r1_here
    #     c = np.sqrt(dx_here**2 + dy_here**2)
    #     eccentricity = c / a
    #     samples_dict['ecc'][modeli] = eccentricity
    #     samples_dict['Argpe'][modeli] = np.degrees(np.arctan2(
    #         dx_here, dy_here))

    #     samples_dict['R1mas'][modeli] = convert.au_to_mas(
    #         r1_here, distance_star)

        # dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here,
        #                                  distance_star)

        # samples_dict['RA'][modeli] = dAlpha
        # samples_dict['Decl'][modeli] = dDelta

        # semimajoraxis = convert.au_to_mas(r1_here, distance_star)
        # ecc = np.sin(np.radians(inc_here))
        # semiminoraxis = semimajoraxis*np.sqrt(1- ecc**2)

        # samples_dict['Smaj'][modeli] = semimajoraxis
        # samples_dict['ecc'][modeli] = ecc
        # samples_dict['Smin'][modeli] = semiminoraxis

        # true_a, true_ecc, argperi, inc, longnode = kowalsky(
        #     semimajoraxis, ecc, pa_here, dAlpha, dDelta)

        # samples_dict['Rkowa'][modeli] = true_a
        # samples_dict['ekowa'][modeli] = true_ecc
        # samples_dict['ikowa'][modeli] = inc
        # samples_dict['Omega'][modeli] = longnode
        # samples_dict['Argpe'][modeAli] = argperi

    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    if sigma == 1:
        quants = [15.9, 50., 84.1]
    if sigma == 2:
        quants = [2.3, 50., 97.77]
    if sigma == 3:
        quants = [0.1, 50., 99.9]

    for key in samples_dict.keys():
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(4)

        percent = np.percentile(samples_dict[key], quants)

        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]

    # MLval_mcmc_val_mcmc_err_dict['RAp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['RA'], PIXSCALE_INS)
    # MLval_mcmc_val_mcmc_err_dict['Declp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['Decl'], PIXSCALE_INS)

    # MLval_mcmc_val_mcmc_err_dict['R2mas'] = convert.au_to_mas(
    #     MLval_mcmc_val_mcmc_err_dict['R2'], distance_star)

    # print(" ")
    # for key in MLval_mcmc_val_mcmc_err_dict.keys():
    #     print(key +
    #           '_ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f}'.format(
    #               MLval_mcmc_val_mcmc_err_dict[key][0],
    #               MLval_mcmc_val_mcmc_err_dict[key][1],
    #               MLval_mcmc_val_mcmc_err_dict[key][2],
    #               MLval_mcmc_val_mcmc_err_dict[key][3]) + comments_dict[key])
    # print(" ")

    print(" ")
    if (modelfit_physical_to_freeform.SPF_MODEL
            == 'hg_1g') or (modelfit_physical_to_freeform.SPF_MODEL
                            == 'hg_2g') or (modelfit_physical_to_freeform.SPF_MODEL == 'hg_3g'):
        just_these_params = ['g1', 'g2', 'Alph1']
        for key in just_these_params:
            print(key + ' MCMC {0:.3f}, -/+1sig: {1:.3f}/+{2:.3f}'.format(
                MLval_mcmc_val_mcmc_err_dict[key][1],
                MLval_mcmc_val_mcmc_err_dict[key][2],
                MLval_mcmc_val_mcmc_err_dict[key][3]))
        print(" ")

    hdr = fits.Header()
    hdr['COMMENT'] = 'Best model of the MCMC reduction'
    hdr['COMMENT'] = 'PARAM_ML are the parameters producing the best LH'
    hdr['COMMENT'] = 'PARAM_MM are the parameters at the 50% percentile in the MCMC'
    hdr['COMMENT'] = 'PARAM_M and PARAM_P are the -/+ sigma error bars (16%, 84%)'
    hdr['KL_FILE'] = name_h5
    hdr['FITSDATE'] = str(datetime.now())
    hdr['BURNIN'] = burnin
    hdr['THIN'] = thin

    hdr['TOT_ITER'] = reader.iteration

    hdr['n_walker'] = nwalkers
    hdr['n_param'] = n_dim_mcmc

    hdr['MAX_LH'] = (np.max(log_prob_samples_flat),
                     'Max likelyhood, obtained for the ML parameters')

    for key in samples_dict.keys():
        hdr[key + '_ML'] = (MLval_mcmc_val_mcmc_err_dict[key][0],
                            comments_dict[key])
        hdr[key + '_MC'] = MLval_mcmc_val_mcmc_err_dict[key][1]
        hdr[key + '_M'] = MLval_mcmc_val_mcmc_err_dict[key][2]
        hdr[key + '_P'] = MLval_mcmc_val_mcmc_err_dict[key][3]

    return hdr


########################################################
def best_model_plot(params_mcmc_yaml, ff_image):
    """ Make the best models plot and save fits of
        BestModel
        BestModel_Conv
        BestModel_FM
        BestModel_Res

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        hdr: the header obtained from create_header

    Returns:
        None
    """
    # mcmcresultdir = save_to
    # I am going to plot the model, I need to define some of the
    # global variables to do so

    # global ALIGNED_CENTER, PIXSCALE_INS, DISTANCE_STAR, WHEREMASK2GENERATEDISK, DIMENSION, SPF_MODEL
    freeform_image = fits.getdata(ff_image)
    pixscale = params_mcmc_yaml['PIXSCALE_INS']
    modelfit_physical_to_freeform.DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    modelfit_physical_to_freeform.PIXSCALE_INS = pixscale

    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    band_name = params_mcmc_yaml['BAND_NAME']
    name_h5 = file_prefix + '_backend_file_mcmc'

    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    sm_scatt_ang = params_mcmc_yaml['DISK_MIN_SC_ANG']
    lg_scatt_ang = params_mcmc_yaml['DISK_MAX_SC_ANG']
    noise_scaling = params_mcmc_yaml['NOISE_MULTIPLICATION_FACTOR']
    sigma = params_mcmc_yaml['sigma']
    counts_to_Jy = params_mcmc_yaml["CONV_FACTOR"]
    conv_factor_SB = counts_to_Jy * 1e3 / pixscale**2

    modelfit_physical_to_freeform.ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']
    modelfit_physical_to_freeform.SPF_MODEL = params_mcmc_yaml[
        'SPF_MODEL']  #Type of description for the SPF

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    iwa = params_mcmc_yaml['IWA']
    distance_star = params_mcmc_yaml["DISTANCE_STAR"]

    dimension = round(aligned_center[0]) * 2

    max_fov = dimension / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    n_pts = int(np.floor(dimension / 1))
    xsize = max_fov * distance_star  #maximum radial distance in AU from the center to the edge

    # print(f'max_fov: {max_fov}; xsize: {xsize}')

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=n_pts)
    z = np.linspace(-xsize, xsize, num=n_pts)

    modelfit_physical_to_freeform.Y_MODEL = y
    modelfit_physical_to_freeform.Z_MODEL = z
    modelfit_physical_to_freeform.n_pts = n_pts

    # Read in the HG SPF params for comparison plots later
    g1 = params_mcmc_yaml["g1_init"]
    g2 = params_mcmc_yaml["g2_init"]
    alpha1 = params_mcmc_yaml["alpha1_init"]
    g3 = params_mcmc_yaml["g3_init"]
    alpha2 = params_mcmc_yaml["alpha2_init"]

    # Read in the current realistic SPF file for comparison later
    phasefunction_file = f"{basedir}/{params_mcmc_yaml['SPF_FILE']}/phase_function.txt"
    plaw_aexp = params_mcmc_yaml["aexp_init"]

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain = reader.get_chain(discard=0)
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    wheremin = np.where(
        log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    n_dim_mcmc = chain.shape[2]
    n_dim_geo = int(len(free_params))
    params_ml = chain_flat[wheremin0, :]
    # theta_ml = params_ml[]

    # dc_offset = coeffs_init[0]
    # plot_ml_spf(dc_offset,coeffs_ml,(sm_scatt_ang,lg_scatt_ang))
    if basis == "legendre" or basis == "bessel":
        if sigma == 1:
            quants = (0.159, 0.841)
        if sigma == 2:
            quants = (0.023, 0.977)
        if sigma == 3:
            quants = (0.001, 0.999)
        coeffs_ml = params_ml[n_dim_geo:]
        # dc_offset = coeffs_ml[0]
        dc_offset = dc
        upper_coeffs_errs = []
        lower_coeffs_errs = []
        for i in range(len(coeffs_ml)):
            errs_on_coeffs = corner.quantile(chain_flat[:,n_dim_geo + i],
                                             quants)
            lower_coeffs_errs.append(errs_on_coeffs[0])
            upper_coeffs_errs.append(errs_on_coeffs[1])
        print(f"Upper coeff error: {upper_coeffs_errs}")
        print(f"Lower coeff error: {lower_coeffs_errs}")
        np.save(f"{mcmcresultdir}/coeffs_ml",
                np.asarray(coeffs_ml))
        np.save(f"{mcmcresultdir}/upper_err_coeffs_ml",
                np.asarray(upper_coeffs_errs))
        np.save(f"{mcmcresultdir}/lower_err_coeffs_ml",
                np.asarray(lower_coeffs_errs))
        plot_ml_custom_spf(coeffs_ml,
                           (sm_scatt_ang,lg_scatt_ang),
                           sc_angs,
                           dc_offset,
                           errupper=upper_coeffs_errs,
                           errlower=lower_coeffs_errs)
        modelfit_physical_to_freeform.COEFFS_INIT = coeffs_init_all
        # plot_all_ml_spfs(g1=g1, g2=g2, alpha1=alpha1,
        #                  coeffs=coeffs_ml,scplotrange=(sm_scatt_ang,lg_scatt_ang),
        #                  dcterm=dc_offset, errlower=lower_coeffs_errs, errupper=upper_coeffs_errs,
        #                  plaw_index=plaw_aexp, adpspf_file=phasefunction_file)
        


    mask2generatedisk = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))

    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    modelfit_physical_to_freeform.MASK2GENERATEDISK = (mask2generatedisk !=
                                           mask2generatedisk)
    instrument = params_mcmc_yaml['INSTRUMENT']

    # load the data
    reduced_data = fits.getdata(
        os.path.join(klipdir, file_prefix + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode
    modelfit_physical_to_freeform.DIMENSION = reduced_data.shape[1]
    # modelfit_physical_to_freeform.ORTHO_BASIS = select_basis

    # load the noise
    noise = fits.getdata(os.path.join(klipdir,
                                      file_prefix + '_noisemap.fits')) / noise_scaling
    # params_ml = np.append(theta_init[:-4],theta_ml)
    # params_ml = np.append(params_ml,theta_ml)
    # print(params_ml)
    disk_ml, _ = modelfit_physical_to_freeform.call_gen_disk(params_ml)
    print(f"Total disk flux: {np.sum(disk_ml) * conv_factor_SB / 1e3} mJy")

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel.fits'),
                 disk_ml,
                #  header=hdr,
                 overwrite=True)

    # find the position of the pericenter in the model
    # argpe = hdr['ARGPE_MC']
    # pa = hdr['PA_MC']

    # model_rot = np.clip(
    #     rotate(disk_ml, argpe + pa, mode='wrap', reshape=False), 0., None)

    # argpe_direction = model_rot[int(modelfit_physical_to_freeform.ALIGNED_CENTER[0]):,
    #                             int(modelfit_physical_to_freeform.ALIGNED_CENTER[1])]
    # radius_argpe = np.where(argpe_direction == np.nanmax(argpe_direction))[0]

    # x_peri_true = radius_argpe * np.cos(
    #     np.radians(argpe + pa + 90))  # distance to star, in pixel
    # y_peri_true = radius_argpe * np.sin(
    #     np.radians(argpe + pa + 90))  # distance to star, in pixel

    #convolve by the PSF
    # disk_ml_convolved = convolve(disk_ml, psf, boundary='wrap')
    disk_ml_convolved = convolve(disk_ml, psf, mode='same')

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Conv.fits'),
                 disk_ml_convolved,
                #  header=hdr,
                 overwrite=True)
    


    mask_speckle_region = np.ones((reduced_data.shape[0], reduced_data.shape[1]))
    x = np.arange(reduced_data.shape[0], dtype=float)[None,:] - aligned_center[0]
    y = np.arange(reduced_data.shape[1], dtype=float)[:,None] - aligned_center[1]
    rho2d = np.sqrt(x**2 + y**2)
    mask_speckle_region[np.where(rho2d < iwa)] = 0.
    disk_ml_convolved *= mask_speckle_region

    #Measure the residuals
    residuals = freeform_image - disk_ml_convolved
    snr_residuals = (freeform_image - disk_ml_convolved) / noise


    # dim_crop_image = int(4 * params_mcmc_yaml['OWA'] // 2) + 1
    dim_crop_image = round(1.75*params_mcmc_yaml['OWA']) + 1

    disk_ml_crop = crop_center_odd(disk_ml, dim_crop_image)
    # disk_ml_crop *= conv_factor_SB
    freeform_image_crop = crop_center_odd(freeform_image, dim_crop_image)
    disk_ml_convolved_crop = crop_center_odd(disk_ml_convolved, dim_crop_image)
    # disk_ml_convolved_crop *= conv_factor_SB
    reduced_data_crop = crop_center_odd(reduced_data, dim_crop_image)
    # reduced_data_crop *= conv_factor_SB
    residuals_crop = crop_center_odd(residuals, dim_crop_image)
    # residuals_crop *= conv_factor_SB
    snr_residuals_crop = crop_center_odd(snr_residuals, dim_crop_image)

    #Set the colormap
    vmin = params_mcmc_yaml["VSCALING_MIN"] * np.min(disk_ml_convolved)
    vmax = params_mcmc_yaml["VSCALING_MAX"] * np.max(disk_ml_convolved)

    fig, ax = plt.subplots(2,2, figsize=(8,8))


    #The model
    cax00 = ax[0,0].imshow(disk_ml_crop,
                     origin='lower',
                    #  vmin=vmin,
                    #  vmax=vmax,
                     cmap='viridis')

    ax[0,0].set_title("Best Model")
    cbar = fig.colorbar(cax00, fraction=0.046, pad=0.04)
    plt.axis('off')

    #The residuals
    cax01 = ax[0,1].imshow(residuals_crop,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     cmap='viridis')

    ax[0,1].set_title("Residuals")
    cbar = fig.colorbar(cax01, fraction=0.046, pad=0.04)
    plt.axis('off')

    #The model convolved
    cax10 = ax[1,0].imshow(disk_ml_convolved_crop,
                     origin='lower',
                    #  vmin=vmin,
                    #  vmax=vmax,
                     cmap='viridis')

    ax[1,0].set_title("Best Model Convolved")
    cbar = fig.colorbar(cax10, fraction=0.046, pad=0.04)
    plt.axis('off')

    #The freeform image
    cax11 = ax[1,1].imshow(freeform_image_crop,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     cmap='viridis')
    ax[1,1].set_title("Optimized Freeform Image")

    # make the colobar ticks integer only for gpi
    cbar = fig.colorbar(cax11, fraction=0.046, pad=0.04)
    plt.axis('off')

    # #The SNR of the residuals
    # ax1 = fig.add_subplot(236)
    # cax = plt.imshow(snr_residuals_crop,
    #                  origin='lower',
    #                  vmin=-2,
    #                  vmax=5,
    #                  cmap='viridis')
    # ax1.set_title("SNR Residuals", fontsize=caracsize, pad=caracsize / 3.)
    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1, 2, 3, 4, 5], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    # cbar.ax.set_yticklabels(['-1', '0', '1', '2', '3', '4', '5'])
    # plt.axis('off')

    

    

    fig.suptitle(band_name + ': Best Model and Residuals',
                 y=0.985)

    fig.tight_layout()

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.jpg'))
    plt.close()



if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', NumbaWarning)
    parser = argparse.ArgumentParser(description='run diskFM MCMC')
    parser.add_argument('-p',
                        '--param_file',
                        required=False,
                        help='parameter file name')
    parser.add_argument("--image",
                        help="Path to freeform fits file to use.",
                        required=True)
    # parser.add_argument("--noise",
    #                     help="Path to noise map fits file to use.",
    #                     required=True)
    args = parser.parse_args()

    if args.param_file is None: #grab param file if no command line input, JKK
        str_yaml = f'initialization_files/{default_parameter_file}'
    else:
        str_yaml = args.param_file
        str_yaml_prefix = str_yaml.split("/")[-1]
        save_to_dir = str_yaml_prefix.split(".")[0]

    with open(os.path.join(str_yaml),
              'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)

    params_mcmc_yaml['BAND_NAME'] = params_mcmc_yaml[
        'BAND_NAME'] + ' (KL#: ' + str(params_mcmc_yaml['KLMODE_NUMBER']) + ')'
    print(params_mcmc_yaml['BAND_NAME'])
    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    modelfit_physical_to_freeform.SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF
    modelfit_physical_to_freeform.DISK_MODEL = params_mcmc_yaml['DISK_MODEL']
    modelfit_physical_to_freeform.FREE_PARAMS = modelfit_physical_to_freeform.arr_free_params(params_mcmc_yaml)
    modelfit_physical_to_freeform.THETA_INIT = modelfit_physical_to_freeform.from_param_to_theta_init(params_mcmc_yaml)
    klipdir = os.path.join(DATADIR, 'klip_fm_files')
    mcmcresultdir = f'{save_to_dir}/results_MCMC'

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    rprofsub = params_mcmc_yaml['RPROFSUB']
    sm_angle = params_mcmc_yaml['MIN_SCATT_ANG']
    lg_angle = params_mcmc_yaml['MAX_SCATT_ANG']
    n_modes = params_mcmc_yaml['N_BASIS']
    basis = params_mcmc_yaml["SPF_BASIS"]
    spf_model = params_mcmc_yaml['SPF_MODEL']
    disk_model = params_mcmc_yaml["DISK_MODEL"]
    lambda_reg = params_mcmc_yaml["LAMBDA_REG"]

    psf = fits.getdata(os.path.join(klipdir, file_prefix + '_instrPSF.fits'))
    psf /= np.sum(psf)


    if not os.path.isfile(os.path.join(mcmcresultdir, name_h5 + '.h5')):
        raise ValueError("the mcmc h5 file does not exist")
    
    free_params = arr_free_params(params_mcmc_yaml)
    modelfit_physical_to_freeform.FREE_PARAMS = free_params
    # scattering_angles_deg = np.arange(sm_angle,lg_angle,1)
    sc_angs = np.linspace(sm_angle,lg_angle,(lg_angle - sm_angle) * 10)
    x_tofit = np.linspace(0,10,len(sc_angs))
    modelfit_physical_to_freeform.X_TOFIT = x_tofit
    sc_angs_rad = np.deg2rad(sc_angs)
    modelfit_physical_to_freeform.SC_ANGS_RAD = sc_angs_rad
    theta_init = modelfit_physical_to_freeform.from_param_to_theta_init(params_mcmc_yaml)
    spf_tofit = calculate_hg_spf(g1=theta_init[-3],
                                 g2=theta_init[-2],
                                 alpha1=theta_init[-1],
                                #  scattangs=scattering_angles_deg)
                                 scattangs=sc_angs)
    modelfit_physical_to_freeform.THETA_INIT = theta_init[:-3]
    # COEFFS_INIT = fit_fourier_to_hg_spf(klipdir,g1=THETA_INIT[-4],g2=THETA_INIT[-3],alpha1=THETA_INIT[-2])
    xrange_max = params_mcmc_yaml["XRANGE_MAX"]
    if spf_model == "hg_1g" or spf_model == "hg_2g":
        if basis == "legendre":
            coeffs_init = fit_legendre_to_hg_spf(klipdir,spf_tofit,sc_angs,nmodes=n_modes)
            dc = coeffs_init[0]
            coeffs_init_all = coeffs_init
        elif basis == "bessel":
            coeffs_init_leg = fit_legendre_to_hg_spf(klipdir,spf_tofit,sc_angs,nmodes=n_modes)
            dc = coeffs_init_leg[0]
            print(f"Legendre 0 fit on the HG SPF: {dc}")
            coeffs_init, select_basis = fit_bessel_to_hg_spf(klipdir,
                                                             xrange_max,
                                                             spf_tofit,
                                                             sc_angs,
                                                             nmodes=n_modes,
                                                             dcoffset=dc)
            print(f"Init bestfit Bessel coeffs w/o DC: {coeffs_init}")
            modelfit_physical_to_freeform.SELECT_BASIS = select_basis
        elif basis == "ortho_bessel":
            xrange_max = params_mcmc_yaml["XRANGE_MAX"]
            coeffs_init_leg = fit_legendre_to_hg_spf(klipdir,spf_tofit,sc_angs,nmodes=n_modes)
            dc = coeffs_init_leg[0]
            print(f"Legendre 0 fit on the HG SPF: {dc}")
            coeffs_init, select_basis = fit_bessel_to_hg_spf(klipdir,
                                                             xrange_max,
                                                             spf_tofit,
                                                             sc_angs,
                                                             nmodes=n_modes,
                                                             dcoffset=dc,
                                                             orthogonalize=True)
            print(f"Init bestfit Bessel coeffs w/o DC: {coeffs_init}")
            modelfit_physical_to_freeform.SELECT_BASIS = select_basis
            coeffs_init_all = np.append(dc,coeffs_init)
            basis = "bessel"
        modelfit_physical_to_freeform.COEFFS_INIT = coeffs_init_all
    
    modelfit_physical_to_freeform.XRANGE_MAX = xrange_max
    modelfit_physical_to_freeform.N_MODES = n_modes
    modelfit_physical_to_freeform.BASIS = basis
    modelfit_physical_to_freeform.DC = dc


    # Plot the chain values
    make_chain_plot(params_mcmc_yaml)

    # compare SPF with injected
    # compare_injected_spfs_plot(params_mcmc_yaml)

    # # Plot the PDFs
    make_corner_plot(params_mcmc_yaml)

    # measure the best likelyhood model and excract MCMC errors
    hdr = create_header(params_mcmc_yaml)

    # save the fits, plot the model and residuals
    best_model_plot(params_mcmc_yaml,  args.image)

    # print some of the best parameter values to put in excel/latex easily(not super clean)
    # print_geometry_parameter(params_mcmc_yaml, hdr)
