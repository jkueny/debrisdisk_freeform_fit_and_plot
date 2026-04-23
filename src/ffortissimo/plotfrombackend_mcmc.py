# pylint: disable=C0103

####### MCMC plotting for wind-driven halo (WDH) runs from windfit_mcmc #######
import os
import sys
from datetime import datetime

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import yaml

import corner
from emcee import backends, autocorr
from numba.core.errors import NumbaWarning
import warnings

def get_basedir():
    basedir = os.environ.get('DATA_DIR',f'{os.environ["HOME"]}/data')
    # print(f"Basedir: {basedir}")
    return basedir

from ffortissimo.dev.pyklip.fmlib.windfm import WindFM
import ffortissimo.windfit_mcmc as wfm

basedir = get_basedir()

default_parameter_file = 'wdh_HR4796_z_20230309_10.yaml'
# default_parameter_file = 'wdh_HR4796_i_20230309_10.yaml'
# default_parameter_file = 'wdh_HR4796_r_20230312_13.yaml'

# Populated by bootstrap_windfit_plot_runtime before plotting.
klipdir = None
mcmcresultdir = None


def _initialization_yaml_path(str_yalm):
    """Resolve YAML path for `python -m ffortissimo.plotfrombackend_mcmc` from repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_init = os.path.join(here, '..', '..', 'initialization_files', str_yalm)
    if os.path.isfile(repo_init):
        return repo_init
    cwd_init = os.path.join(os.getcwd(), 'initialization_files', str_yalm)
    if os.path.isfile(cwd_init):
        return cwd_init
    return repo_init


def bootstrap_windfit_plot_runtime(params_mcmc_yaml, datadir):
    """Match paths and windfit_mcmc globals used by call_gen_disk / logl."""
    global klipdir, mcmcresultdir
    klipdir = os.path.join(datadir, 'wind_fm_files')
    mcmcresultdir = os.path.join(datadir, 'windfit_MCMC')

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    wdh_cfg = params_mcmc_yaml.get('wdh_model', params_mcmc_yaml)

    wfm.N_WDH_COMPONENTS = wfm._n_wdh_components(params_mcmc_yaml)
    wfm.SHARED_COMPONENT_FLAGS = wfm._shared_component_flags(params_mcmc_yaml)
    wfm.COMPONENT_INIT = [
        wfm._component_init_from_yaml(params_mcmc_yaml, idx)
        for idx in range(1, wfm.N_WDH_COMPONENTS + 1)
    ]
    wfm.GAMMA_FIXED = float(wfm._cfg_first_match(wdh_cfg, ['gamma_fixed'], default=1.0))
    wfm.FREE_PARAMS = wfm.arr_free_params(params_mcmc_yaml)
    wfm.THETA_INIT = wfm.from_param_to_theta_init(params_mcmc_yaml)
    wfm.ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']
    wfm.DIMENSION = int(round(wfm.ALIGNED_CENTER[0]) * 2)

    mask_path = os.path.join(klipdir, file_prefix + '_mask2generatehalo.fits')
    wfm.WHEREMASK2GENERATEHALO = (fits.getdata(mask_path) == 0)

    wfm.RPROFSUB = bool(params_mcmc_yaml.get('RPROFSUB', False))
    if wfm.RPROFSUB:
        wfm.RADIAL_INDS = np.asarray(
            wfm.get_radial_inds((wfm.DIMENSION, wfm.DIMENSION), wfm.ALIGNED_CENTER)
        )
        radial_np = np.asarray(wfm.RADIAL_INDS, dtype=np.int32)
        max_r = int(radial_np.max()) + 1
        wfm.RADII = np.arange(max_r)
    else:
        wfm.RADIAL_INDS = None
        wfm.RADII = None


def _backend_param_tokens(n_dim_mcmc, params_mcmc_yaml):
    """Infer token order for an existing backend, including pre-PA chains."""
    current = list(wfm.FREE_PARAMS)
    if n_dim_mcmc == len(current):
        return current
    no_pa = [tok for tok in current if not tok.startswith('PA_')]
    if n_dim_mcmc == len(no_pa):
        return no_pa
    names = list(params_mcmc_yaml.get('NAMES', []))
    if len(names) >= n_dim_mcmc:
        return names[:n_dim_mcmc]
    return [f'theta_{i + 1}' for i in range(n_dim_mcmc)]


def _axis_labels_for_tokens(tokens, params_mcmc_yaml):
    labels_map = params_mcmc_yaml.get('LABELS', {})
    return [labels_map.get(tok, tok) for tok in tokens]


def _compact_axis_label(token, fallback_label):
    """Compact parameter labels for dense chain/corner plots."""
    try:
        p_name, idx_str = token.rsplit('_', 1)
    except ValueError:
        return fallback_label

    compact_base = {
        'sigma_up': 'sig_up',
        'sigma_down': 'sig_dn',
        'Norm': 'N',
    }.get(p_name, p_name)
    return f'{compact_base}_{idx_str}'


def _theta_init_for_tokens(tokens):
    component_lookup = {idx + 1: comp for idx, comp in enumerate(wfm.COMPONENT_INIT)}
    theta = []
    for token in tokens:
        try:
            p_name, idx_str = token.rsplit('_', 1)
            theta.append(float(component_lookup[int(idx_str)][p_name]))
        except Exception:
            theta.append(np.nan)
    return np.asarray(theta, dtype=float)


def crop_center_odd(img, crop):
    img[img != img] = 0.
    y, x = img.shape
    startx = (x - 1) // 2 - crop // 2
    starty = (y - 1) // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop]


_FITS_COMMENT_CHARS = 68


def _fits_short_comment(text):
    """Keep COMMENT field within typical FITS card length."""
    if not text:
        return ''
    s = ' '.join(str(text).split())
    if len(s) <= _FITS_COMMENT_CHARS:
        return s
    return s[: _FITS_COMMENT_CHARS - 3] + '...'


def _fits_par_prefix(i_one_based):
    """Return PARnn (5 chars); PARnn_XX keywords are 8 chars (FITS HIERARCH not needed)."""
    if not (1 <= i_one_based <= 99):
        raise ValueError(f'FITS PARnn index out of range: {i_one_based}')
    return f'PAR{i_one_based:02d}'


########################################################
def make_chain_plot(params_mcmc_yaml):
    """Plot walker histories from the emcee HDF5 backend (native theta space)."""
    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    if reader.iteration < burnin - 1:
        burnin = 0
        params_mcmc_yaml['BURNIN'] = 0

    chain = reader.get_chain(discard=0, thin=thin)
    chain_flat = reader.get_chain(flat=True)
    log_prob_samples_flat = reader.get_log_prob(
        discard=burnin, flat=True, thin=thin
    )
    wheremin = np.where(log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    theta_ml = chain_flat[wheremin0, :]

    tau = autocorr.integrated_time(chain[:, :, :], tol=5)
    if burnin > reader.iteration - 1:
        raise ValueError('the burnin cannot be larger than the # of iterations')

    print('')
    print('')
    print(name_h5)
    print('# of iteration in the backend chain initially: {0}'.format(reader.iteration))
    print('Max Tau times 50: {0}'.format(50 * np.min(tau)))
    print('')

    print('Maximum Likelyhood: {0}'.format(np.nanmax(log_prob_samples_flat)))
    print('burn-in: {0}'.format(burnin))
    print('chain shape: {0}'.format(chain.shape))

    n_dim_mcmc = chain.shape[2]
    nwalkers = chain.shape[1]
    tokens = _backend_param_tokens(n_dim_mcmc, params_mcmc_yaml)
    base_labels = _axis_labels_for_tokens(tokens, params_mcmc_yaml)
    axis_labels = [
        _compact_axis_label(tok, lab) for tok, lab in zip(tokens, base_labels)
    ]

    print('Best-fit model params (theta at max log-prob)...')
    for i, tok in enumerate(tokens):
        print(f'{tok}: {theta_ml[i]}')

    if n_dim_mcmc != len(axis_labels):
        raise ValueError(
            f'LABELS count ({len(axis_labels)}) != chain dim ({n_dim_mcmc}). '
            'Add LABELS keys for every entry in backend parameter tokens (see YAML).'
        )

    _, axarr = plt.subplots(
        n_dim_mcmc,
        sharex=True,
        figsize=(6.4 * quality_plot, 4.8 * quality_plot),
    )

    for i in range(n_dim_mcmc):
        axarr[i].set_ylabel(axis_labels[i], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)
        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)
        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[n_dim_mcmc - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[n_dim_mcmc - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains.jpg'))
    plt.close()


########################################################
def make_corner_plot(params_mcmc_yaml):
    """Corner plot of the thinned chain (native theta space)."""
    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'
    band_name = params_mcmc_yaml['BAND_NAME']

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    n_dim_mcmc = chain_flat.shape[1]
    tokens = _backend_param_tokens(n_dim_mcmc, params_mcmc_yaml)
    base_labels = _axis_labels_for_tokens(tokens, params_mcmc_yaml)
    axis_labels = [
        _compact_axis_label(tok, lab) for tok, lab in zip(tokens, base_labels)
    ]

    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros((len(chain4thatparam[wherenotnan]), n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:, i] = chain_flat[wherenotnan, i]
        chain_flat = chainflatnonan

    if n_dim_mcmc != len(axis_labels):
        raise ValueError(
            f'LABELS count ({len(axis_labels)}) != chain dim ({n_dim_mcmc}). '
            'Add LABELS keys for every entry in backend parameter tokens (see YAML).'
        )

    rcParams['axes.labelsize'] = 19
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13

    if sigma == 1:
        quants = (0.159, 0.5, 0.841)
    elif sigma == 2:
        quants = (0.023, 0.5, 0.977)
    else:
        quants = (0.001, 0.5, 0.999)

    shouldweplotalldatapoints = 'Fake' in file_prefix

    fig = corner.corner(
        chain_flat,
        labels=axis_labels,
        quantiles=quants,
        show_titles=True,
        plot_datapoints=shouldweplotalldatapoints,
        verbose=False,
    )

    truth = _theta_init_for_tokens(tokens)
    if shouldweplotalldatapoints and np.all(np.isfinite(truth)):
        green_line = mlines.Line2D(
            [], [], color='red', label='Initial theta (YAML)'
        )
        plt.legend(
            handles=[green_line],
            loc='center right',
            bbox_to_anchor=(0.5, 8),
            fontsize=30,
        )
        axes = np.array(fig.axes).reshape((n_dim_mcmc, n_dim_mcmc))
        for i in range(n_dim_mcmc):
            axes[i, i].axvline(truth[i], color='r')
        for yi in range(n_dim_mcmc):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truth[xi], color='r')
                ax.axhline(truth[yi], color='r')

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(
        band_name,
        xy=(0.55, 0.99),
        xycoords='figure fraction',
        xytext=(-20, -10),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=44,
    )
    fig.gca().annotate(
        '{0:,} iterations (+ {1:,} burn-in)'.format(reader.iteration - burnin, burnin),
        xy=(0.55, 0.95),
        xycoords='figure fraction',
        xytext=(-20, -10),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=44,
    )
    fig.gca().annotate(
        'with {0:,} walkers: {1:,} models'.format(nwalkers, reader.iteration * nwalkers),
        xy=(0.55, 0.91),
        xycoords='figure fraction',
        xytext=(-20, -10),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=44,
    )

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs.pdf'))
    plt.close()


########################################################
def create_header(params_mcmc_yaml):
    """FITS header metadata from posteriors (WDH free parameters only)."""
    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    comments_all = params_mcmc_yaml.get('COMMENTS', {})
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    log_prob_samples_flat = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    n_dim_mcmc = chain_flat.shape[1]

    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros((len(chain4thatparam[wherenotnan]), n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:, i] = chain_flat[wherenotnan, i]
        chain_flat = chainflatnonan
        log_prob_samples_flat = log_prob_samples_flat[wherenotnan]

    free_tokens = _backend_param_tokens(n_dim_mcmc, params_mcmc_yaml)

    samples_dict = {free_tokens[i]: chain_flat[:, i] for i in range(n_dim_mcmc)}

    MLval_mcmc_val_mcmc_err_dict = {}
    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    if sigma == 1:
        quants = [15.9, 50.0, 84.1]
    elif sigma == 2:
        quants = [2.3, 50.0, 97.77]
    else:
        quants = [0.1, 50.0, 99.9]

    for key in samples_dict:
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(4)
        percent = np.percentile(samples_dict[key], quants)
        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]

    hdr = fits.Header()
    hdr['COMMENT'] = 'WDH MCMC best model; PARnn_* see COMMENT map below'
    hdr['COMMENT'] = 'PARnn_ML max log-prob row; PARnn_MC posterior median'
    hdr['COMMENT'] = 'PARnn_EM / PARnn_EP lower/upper error (percentile-MC)'
    hdr['KL_FILE'] = name_h5
    hdr['FITSDATE'] = str(datetime.now())
    hdr['BURNIN'] = burnin
    hdr['THIN'] = thin
    hdr['TOT_ITER'] = reader.iteration
    hdr['n_walker'] = nwalkers
    hdr['n_param'] = n_dim_mcmc
    hdr['MAX_LH'] = (
        float(np.max(log_prob_samples_flat)),
        _fits_short_comment('Max log-prob at ML row'),
    )

    for i, key in enumerate(free_tokens, start=1):
        pfx = _fits_par_prefix(i)
        cmt = _fits_short_comment(
            comments_all.get(key, f'MCMC parameter {key}')
        )
        v = MLval_mcmc_val_mcmc_err_dict[key]
        hdr[f'{pfx}_ML'] = (float(v[0]), cmt)
        hdr[f'{pfx}_MC'] = float(v[1])
        hdr[f'{pfx}_EM'] = float(v[2])
        hdr[f'{pfx}_EP'] = float(v[3])
        hdr['COMMENT'] = _fits_short_comment(f'{pfx} maps to FREE_PARAMS {key}')

    return hdr


########################################################
def best_model_plot(params_mcmc_yaml, hdr):
    """Intrinsic WDH sum, pre-FM input to WindFM, KL FM map, residuals, and SNR."""
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    band_name = params_mcmc_yaml['BAND_NAME']
    name_h5 = file_prefix + '_backend_file_mcmc'

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(
        discard=burnin, flat=True, thin=thin
    )
    wheremin = np.where(log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    theta_ml = chain_flat[wheremin0, :]
    tokens = _backend_param_tokens(chain_flat.shape[1], params_mcmc_yaml)

    reduced_data = fits.getdata(
        os.path.join(klipdir, file_prefix + '-klipped-KLmodes-all.fits')
    )[0]

    noise = fits.getdata(os.path.join(klipdir, file_prefix + '_noisemap.fits'))
    noise = noise.reshape(reduced_data.shape)
    variance_for_stats = noise**2
    total_noise_spatial = np.sqrt(np.nansum(variance_for_stats))
    noise[noise == 0.0] = np.nan

    free_params_backup = list(wfm.FREE_PARAMS)
    try:
        wfm.FREE_PARAMS = list(tokens)
        model_list = wfm.call_gen_disk(theta_ml)
    finally:
        wfm.FREE_PARAMS = free_params_backup
    intrinsic_sum = np.nansum(np.asarray(model_list), axis=0)

    if wfm.RPROFSUB:
        combined = np.nansum(np.asarray(model_list), axis=0)
        combined_sub = wfm.subtract_radial_profile_np(
            combined, wfm.RADIAL_INDS, wfm.RADII
        )
        model_list_fm = [np.asarray(combined_sub, dtype=np.float32)]
    else:
        model_list_fm = model_list


    fits.writeto(
        os.path.join(mcmcresultdir, name_h5 + '_BestModel.fits'),
        intrinsic_sum,
        header=hdr,
        overwrite=True,
    )

    pre_fm = np.nansum(np.asarray(model_list_fm, dtype=np.float64), axis=0)

    fits.writeto(
        os.path.join(mcmcresultdir, name_h5 + '_BestModel_PreFM.fits'),
        pre_fm,
        header=hdr,
        overwrite=True,
    )

    windobj = WindFM(
        None,
        None,
        None,
        model_wdh_list=model_list_fm,
        model_pas_mask=None,
        basis_filename=os.path.join(klipdir, file_prefix + '_klbasis.h5'),
        load_from_basis=True,
    )
    windobj.update_wind(model_list_fm)
    model_fm = windobj.fm_parallelized()[0]
    model_fm[model_fm != model_fm] = 0.0

    fits.writeto(
        os.path.join(mcmcresultdir, name_h5 + '_BestModel_FM.fits'),
        model_fm,
        header=hdr,
        overwrite=True,
    )

    residuals = reduced_data - model_fm
    snr_residuals = (reduced_data - model_fm) / noise

    fits.writeto(
        os.path.join(mcmcresultdir, name_h5 + '_BestModel_Res.fits'),
        residuals,
        header=hdr,
        overwrite=True,
    )

    vmin = params_mcmc_yaml['VSCALING_MIN'] * np.nanmin(reduced_data)
    vmax = params_mcmc_yaml['VSCALING_MAX'] * np.nanmax(reduced_data)
    dim_crop_image = round(1.75 * params_mcmc_yaml['OWA']) + 1

    intrinsic_crop = crop_center_odd(intrinsic_sum, dim_crop_image)
    pre_fm_crop = crop_center_odd(pre_fm, dim_crop_image)
    fm_crop = crop_center_odd(model_fm, dim_crop_image)
    reduced_data_crop = crop_center_odd(reduced_data, dim_crop_image)
    residuals_crop = crop_center_odd(residuals, dim_crop_image)
    snr_residuals_crop = crop_center_odd(snr_residuals, dim_crop_image)

    caracsize = 40 * quality_plot / 2.0
    fig = plt.figure(figsize=(6.4 * 2 * quality_plot, 4.8 * 2 * quality_plot))

    ax1 = fig.add_subplot(235)
    cax = plt.imshow(
        reduced_data_crop + 0.1,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap="viridis",
    )
    ax1.set_title('KLIP reduced data', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    plt.axis('off')

    ax1 = fig.add_subplot(233)
    cax = plt.imshow(
        residuals_crop,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap="viridis",
    )
    ax1.set_title('Residuals', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    plt.axis('off')

    ax1 = fig.add_subplot(236)
    cax = plt.imshow(
        snr_residuals_crop, origin='lower', vmin=-2, vmax=5, cmap='seismic'
    )
    ax1.set_title('SNR residuals', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(
        cax, ticks=[-1, 0, 1, 2, 3, 4, 5], fraction=0.046, pad=0.04
    )
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    cbar.ax.set_yticklabels(['-1', '0', '1', '2', '3', '4', '5'])
    plt.axis('off')

    ax1 = fig.add_subplot(231)
    vmax_model = int(np.round(np.max(intrinsic_crop) / 2.0))
    cax = plt.imshow(
        intrinsic_crop, origin='lower', vmin=0, vmax=vmax_model, cmap='bone'
    )
    ax1.set_title('Best Model', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    pos_star = plt.Circle(
        (dim_crop_image / 2, dim_crop_image / 2), 2, color='r', alpha=0.8
    )
    ax1.add_artist(pos_star)
    plt.axis('off')

    vmax_pre_fm = int(np.round(np.max(pre_fm_crop) / 3.0))
    vmin_pre_fm = int(np.round(np.min(pre_fm_crop) / 1.5))
    ax1 = fig.add_subplot(234)
    cax = plt.imshow(
        pre_fm_crop,
        origin='lower',
        vmin=vmin_pre_fm,
        vmax=vmax_pre_fm,
        cmap="magma",
    )
    ax1.set_title('Best Model (pre-FM)', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    plt.axis('off')

    ax1 = fig.add_subplot(232)
    cax = plt.imshow(
        fm_crop,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap="magma",
    )
    ax1.set_title('Best Model (FM)', fontsize=caracsize, pad=caracsize / 3.0)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.0)
    plt.axis('off')

    fig.subplots_adjust(hspace=-0.4, wspace=0.2)
    fig.suptitle(
        band_name + ': best WDH model and residuals',
        fontsize=5 / 4.0 * caracsize,
        y=0.985,
    )
    fig.tight_layout()
    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.jpg'))
    plt.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.simplefilter('ignore', NumbaWarning)

    str_yalm = default_parameter_file if len(sys.argv) == 1 else sys.argv[1]
    yaml_path = _initialization_yaml_path(str_yalm)
    with open(yaml_path, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)

    if str(params_mcmc_yaml.get('DISK_MODEL', '')).lower() != 'wdh':
        raise NotImplementedError(
            'plotfrombackend_mcmc is WDH-only; use DISK_MODEL: wdh or extend this script.'
        )

    params_mcmc_yaml['BAND_NAME'] = (
        params_mcmc_yaml['BAND_NAME']
        + ' (KL#: '
        + str(params_mcmc_yaml['KLMODE_NUMBER'])
        + ')'
    )
    print(params_mcmc_yaml['BAND_NAME'])

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    bootstrap_windfit_plot_runtime(params_mcmc_yaml, DATADIR)

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'
    if not os.path.isfile(os.path.join(mcmcresultdir, name_h5 + '.h5')):
        raise FileNotFoundError(
            f'Missing backend: {os.path.join(mcmcresultdir, name_h5 + ".h5")}'
        )

    make_chain_plot(params_mcmc_yaml)
    make_corner_plot(params_mcmc_yaml)
    hdr = create_header(params_mcmc_yaml)
    best_model_plot(params_mcmc_yaml, hdr)
