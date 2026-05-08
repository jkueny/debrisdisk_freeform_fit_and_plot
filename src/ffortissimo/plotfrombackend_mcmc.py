# pylint: disable=C0103

####### MCMC plotting for wind-driven halo (WDH) runs from windfit_mcmc #######
import os
import sys
import math
import re
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
from scipy.signal.windows import tukey
import warnings

def get_basedir():
    basedir = os.environ.get('DATA_DIR',f'{os.environ["HOME"]}/data')
    # print(f"Basedir: {basedir}")
    return basedir

from ffortissimo.dev.pyklip.fmlib.windfm import WindFM
import ffortissimo.windfit_mcmc as wfm

basedir = get_basedir()

# default_parameter_file = 'wdh_HR4796_z_20230309_10.yaml'
default_parameter_file = 'wdh_HR4796_i_20230309_10.yaml'
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

    n_components = wfm._n_wdh_components(params_mcmc_yaml)
    canonical_params = ('h0', 'sigma_up', 'sigma_down', 'PA', 'Norm')
    canonical = [
        f'{p_name}_{idx}'
        for p_name in canonical_params
        for idx in range(1, n_components + 1)
    ]
    if n_dim_mcmc == len(canonical):
        return canonical
    canonical_no_pa = [tok for tok in canonical if not tok.startswith('PA_')]
    if n_dim_mcmc == len(canonical_no_pa):
        return canonical_no_pa

    # Backward compatibility with older backends that sampled beta.
    legacy_params = ('beta', 'h0', 'sigma_up', 'sigma_down', 'PA', 'Norm')
    legacy = [
        f'{p_name}_{idx}'
        for p_name in legacy_params
        for idx in range(1, n_components + 1)
    ]
    if n_dim_mcmc == len(legacy):
        return legacy
    legacy_no_pa = [tok for tok in legacy if not tok.startswith('PA_')]
    if n_dim_mcmc == len(legacy_no_pa):
        return legacy_no_pa

    names = list(params_mcmc_yaml.get('NAMES', []))
    token_pat = re.compile(r'^[A-Za-z][A-Za-z0-9_]*_[0-9]+$')
    if len(names) >= n_dim_mcmc and all(token_pat.match(x) for x in names[:n_dim_mcmc]):
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


def _reported_param_value(token, value):
    """Convert sampled theta values to human-reported values."""
    if token.startswith('Norm_'):
        return float(math.exp(float(value)))
    return float(value)


def _outer_edge_tukey_window(exclude_mask, width_pix, alpha=0.2):
    """Tukey taper only at the outer radial boundary of the model region."""
    if width_pix is None or float(width_pix) <= 0:
        return None

    allowed = ~np.asarray(exclude_mask, dtype=bool)
    if not np.any(allowed):
        return allowed.astype(np.float32)

    y, x = np.indices(allowed.shape)
    center_x, center_y = wfm.ALIGNED_CENTER
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    outer_radius = float(np.nanmax(radius[allowed]))
    dist_from_outer_edge = outer_radius - radius

    taper_width = max(int(round(float(width_pix))), 1)
    tukey_len = max(int(np.ceil(2 * taper_width / float(alpha))) + 1, 3)
    taper_profile = tukey(tukey_len, alpha=alpha)[: taper_width + 1]

    dist_clip = np.clip(dist_from_outer_edge, 0, taper_width)
    window = np.interp(dist_clip, np.arange(taper_width + 1), taper_profile)
    window[~allowed] = 0.0
    window[dist_from_outer_edge >= taper_width] = 1.0
    return window.astype(np.float32, copy=False)


def _apodize_model_components(model_list, params_mcmc_yaml):
    width_pix = params_mcmc_yaml.get(
        'WDH_TUKEY_WIDTH',
        params_mcmc_yaml.get('MODEL_EDGE_TUKEY_WIDTH', 10.0),
    )
    window = _outer_edge_tukey_window(wfm.WHEREMASK2GENERATEHALO, width_pix)
    if window is None:
        return model_list
    return [
        (np.asarray(model, dtype=np.float32) * window).astype(np.float32, copy=False)
        for model in model_list
    ]


def _format_summary_value(value):
    if value is None:
        return 'None'
    try:
        return f'{float(value):.8g}'
    except (TypeError, ValueError):
        return str(value)


def _format_summary_param(value, err_minus=None, err_plus=None):
    text = _format_summary_value(value)
    if err_minus is None or err_plus is None:
        return text
    return (
        f'{text} '
        f'-{_format_summary_value(abs(err_minus))}'
        f'/+{_format_summary_value(abs(err_plus))}'
    )


def _write_end_of_run_summary(
    params_mcmc_yaml,
    name_h5,
    reader,
    chain_shape,
    chain_flat,
    log_prob_samples_flat,
    tokens,
    theta_ml,
    component_params,
    model_list,
):
    """Write a text summary matching terminal output plus per-component sums."""
    raw_sums = [float(np.nansum(model)) for model in model_list]
    total_sum = float(np.nansum(raw_sums)) if raw_sums else np.nan
    largest_sum = max(raw_sums) if raw_sums else np.nan
    if not np.isfinite(largest_sum) or largest_sum == 0.0:
        norm_sums = [np.nan for _ in raw_sums]
    else:
        norm_sums = [val / largest_sum for val in raw_sums]
    if not np.isfinite(total_sum) or total_sum == 0.0:
        norm_to_total_sums = [np.nan for _ in raw_sums]
    else:
        norm_to_total_sums = [val / total_sum for val in raw_sums]

    tau_line = 'Max Tau times 50: unavailable'
    try:
        tau_chain = reader.get_chain(discard=0, thin=params_mcmc_yaml['THIN'])
        tau = autocorr.integrated_time(tau_chain[:, :, :], tol=5)
        tau_line = f'Max Tau times 50: {50 * np.min(tau):.8g}'
    except Exception as exc:
        tau_line = f'Max Tau times 50: unavailable ({exc})'

    ml_log_prob = float(np.nanmax(log_prob_samples_flat))
    whereml = np.where(log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    whereml0 = np.array(whereml).flatten()[0]
    param_stats = {}
    for i, token in enumerate(tokens):
        samples = np.asarray(chain_flat[:, i], dtype=float)
        if token.startswith('Norm_'):
            samples = np.exp(samples)
        percent = np.percentile(samples, [15.9, 50.0, 84.1])
        ml_val = samples[whereml0]
        param_stats[token] = {
            'ml': float(ml_val),
            'err_minus': float(percent[1] - percent[0]),
            'err_plus': float(percent[2] - percent[1]),
        }

    summary_path = os.path.join(mcmcresultdir, name_h5 + '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('WDH MCMC end-of-run summary\n')
        handle.write('===========================\n\n')
        handle.write('Run information\n')
        handle.write('---------------\n')
        handle.write(f'Backend: {name_h5}.h5\n')
        handle.write(f'Band: {params_mcmc_yaml["BAND_NAME"]}\n')
        handle.write(
            f'# of iteration in the backend chain initially: {reader.iteration}\n'
        )
        handle.write(f'{tau_line}\n')
        handle.write(f'Maximum Likelihood: {ml_log_prob:.12g}\n')
        handle.write(f'burn-in: {params_mcmc_yaml["BURNIN"]}\n')
        handle.write(f'thin: {params_mcmc_yaml["THIN"]}\n')
        handle.write(f'chain shape: {chain_shape}\n')
        handle.write(f'n walkers: {params_mcmc_yaml["NWALKERS"]}\n')
        handle.write(f'n parameters: {len(tokens)}\n\n')
        handle.write('Component sum normalization reference\n')
        handle.write('-----------------------------------\n')
        handle.write(f'total agglomerated best-fit model sum: {_format_summary_value(total_sum)}\n')
        handle.write(f'largest component model sum: {_format_summary_value(largest_sum)}\n\n')

        for idx, comp in enumerate(component_params, start=1):
            raw_sum = raw_sums[idx - 1] if idx <= len(raw_sums) else np.nan
            norm_sum = norm_sums[idx - 1] if idx <= len(norm_sums) else np.nan
            norm_total_sum = (
                norm_to_total_sums[idx - 1]
                if idx <= len(norm_to_total_sums)
                else np.nan
            )
            handle.write(f'WDH component {idx}\n')
            handle.write('-' * (14 + len(str(idx))) + '\n')
            handle.write(f'raw model sum: {_format_summary_value(raw_sum)}\n')
            handle.write(
                'normalized model sum (to largest component sum): '
                f'{_format_summary_value(norm_sum)}\n'
            )
            handle.write(
                'normalized model sum (to agglomerated best-fit model sum): '
                f'{_format_summary_value(norm_total_sum)}\n'
            )
            handle.write('best-fit parameters:\n')
            for key in ('beta', 'h0', 'sigma_up', 'sigma_down', 'PA', 'Norm'):
                token = f'{key}_{idx}'
                stats = param_stats.get(token)
                if stats is None:
                    handle.write(f'  {key}: {_format_summary_value(comp.get(key))}\n')
                else:
                    handle.write(
                        f'  {key}: '
                        f'{_format_summary_param(stats["ml"], stats["err_minus"], stats["err_plus"])}\n'
                    )
            handle.write('\n')


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
    chain_flat_for_ml = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(
        discard=burnin, flat=True, thin=thin
    )
    wheremin = np.where(log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    theta_ml = chain_flat_for_ml[wheremin0, :]

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
        print(f'{tok}: {_reported_param_value(tok, theta_ml[i])}')

    if n_dim_mcmc != len(axis_labels):
        raise ValueError(
            f'LABELS count ({len(axis_labels)}) != chain dim ({n_dim_mcmc}). '
            'Add LABELS keys for every entry in backend parameter tokens (see YAML).'
        )

    model_param_indices = {}
    for i, tok in enumerate(tokens):
        try:
            _, idx_str = tok.rsplit('_', 1)
            model_idx = int(idx_str)
        except ValueError:
            model_idx = 0
        model_param_indices.setdefault(model_idx, []).append(i)

    for model_idx in sorted(model_param_indices):
        param_indices = model_param_indices[model_idx]
        n_params_model = len(param_indices)
        _, axarr = plt.subplots(
            n_params_model,
            sharex=True,
            figsize=(
                max(2 * n_params_model, 3) * quality_plot,
                3 * quality_plot,
                ),
        )
        axarr = np.atleast_1d(axarr)

        for ax_idx, param_idx in enumerate(param_indices):
            axarr[ax_idx].set_ylabel(
                axis_labels[param_idx], fontsize=5 * quality_plot
            )
            axarr[ax_idx].tick_params(axis='y', labelsize=4 * quality_plot)
            for j in range(nwalkers):
                axarr[ax_idx].plot(chain[:, j, param_idx], linewidth=quality_plot)
            axarr[ax_idx].axvline(
                x=burnin, color='black', linewidth=1.5 * quality_plot
            )

        axarr[-1].tick_params(axis='x', labelsize=6 * quality_plot)
        axarr[-1].set_xlabel('Iterations', fontsize=10 * quality_plot)

        if model_idx == 0:
            suffix = 'misc'
            title = 'Unindexed parameters'
        else:
            suffix = f'model{model_idx}'
            title = f'WDH model {model_idx}'
        axarr[0].set_title(title, fontsize=8 * quality_plot)

        plt.savefig(os.path.join(mcmcresultdir, f'{name_h5}_chains_{suffix}.jpg'))
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

    # chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    n_dim_mcmc = chain_flat.shape[1]
    tokens = _backend_param_tokens(n_dim_mcmc, params_mcmc_yaml)
    base_labels = _axis_labels_for_tokens(tokens, params_mcmc_yaml)
    axis_labels = [
        _compact_axis_label(tok, lab) for tok, lab in zip(tokens, base_labels)
    ]
    #remove the flux norm params from the corner plot
    n_dim_mcmc_no_norm = n_dim_mcmc - wfm.N_WDH_COMPONENTS
    chain_flat_no_norm = chain_flat[:, :n_dim_mcmc_no_norm]
    axis_labels_no_norm = axis_labels[:n_dim_mcmc_no_norm]
    tokens_no_norm = tokens[:n_dim_mcmc_no_norm]
    base_labels_no_norm = base_labels[:n_dim_mcmc_no_norm]
    compact_axis_labels_no_norm = [
        _compact_axis_label(tok, lab) for tok, lab in zip(tokens_no_norm, base_labels_no_norm)
    ]
    for j in range(n_dim_mcmc_no_norm):
        chain4thatparam = chain_flat_no_norm[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros((len(chain4thatparam[wherenotnan]), n_dim_mcmc_no_norm))
        for i in range(n_dim_mcmc_no_norm):
            chainflatnonan[:, i] = chain_flat[wherenotnan, i]
        chain_flat_no_norm = chainflatnonan

    if n_dim_mcmc_no_norm != len(axis_labels_no_norm):
        raise ValueError(
            f'LABELS count ({len(axis_labels_no_norm)}) != chain dim ({n_dim_mcmc_no_norm}). '
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
        chain_flat_no_norm,
        labels=axis_labels_no_norm,
        quantiles=quants,
        show_titles=True,
        title_fmt=".3f",
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
        axes = np.array(fig.axes).reshape((n_dim_mcmc_no_norm, n_dim_mcmc_no_norm))
        for i in range(n_dim_mcmc):
            axes[i, i].axvline(truth[i], color='r')
        for yi in range(n_dim_mcmc_no_norm):
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

    samples_dict = {}
    for i, tok in enumerate(free_tokens):
        vals = np.asarray(chain_flat[:, i], dtype=float)
        if tok.startswith('Norm_'):
            vals = np.exp(vals)
        samples_dict[tok] = vals

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
    """2×2 figure: best intrinsic model and FM (left column), KLIP data and residuals (right)."""
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    band_name = params_mcmc_yaml['BAND_NAME']
    name_h5 = file_prefix + '_backend_file_mcmc'

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_shape = reader.get_chain(discard=0, thin=thin).shape
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

    free_params_backup = list(wfm.FREE_PARAMS)
    try:
        wfm.FREE_PARAMS = list(tokens)
        param_disk, _ = wfm.from_theta_to_params(theta_ml)
        component_params = param_disk['components']
        model_list = wfm.call_gen_disk(theta_ml)
    finally:
        wfm.FREE_PARAMS = free_params_backup
    model_list = _apodize_model_components(model_list, params_mcmc_yaml)
    intrinsic_sum = np.nansum(np.asarray(model_list), axis=0)
    _write_end_of_run_summary(
        params_mcmc_yaml,
        name_h5,
        reader,
        chain_shape,
        chain_flat,
        log_prob_samples_flat,
        tokens,
        theta_ml,
        component_params,
        model_list,
    )

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

    # Save each intrinsic best-fit WDH component for per-layer inspection.
    for i, model_comp in enumerate(model_list, start=1):
        fits.writeto(
            os.path.join(mcmcresultdir, f'{name_h5}_BestModel_comp{i}.fits'),
            np.asarray(model_comp, dtype=np.float32),
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
    fm_crop = crop_center_odd(model_fm, dim_crop_image)
    reduced_data_crop = crop_center_odd(reduced_data, dim_crop_image)
    residuals_crop = crop_center_odd(residuals, dim_crop_image)

    caracsize = 40 * quality_plot / 2.0
    fig = plt.figure(figsize=(6.4 * 2 * quality_plot, 4.8 * 2 * quality_plot))

    star = plt.Circle(
        (dim_crop_image / 2, dim_crop_image / 2), 2, color='r', alpha=0.8
    )

    # Top left: intrinsic best model (summed WDH)
    ax_bm = fig.add_subplot(2, 2, 1)
    im_bm = ax_bm.imshow(
        intrinsic_crop + 0.1,
        origin='lower',
        # norm=LogNorm(),
        cmap='bone',
        vmin=0,
        vmax=np.percentile(intrinsic_crop, 98.0),
    )
    ax_bm.set_title('Best Model', fontsize=caracsize, pad=caracsize / 3.0)
    fig.colorbar(im_bm, ax=ax_bm, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=caracsize * 3 / 4.0
    )
    ax_bm.add_artist(star)
    ax_bm.axis('off')

    # Top right: KLIP reduced data
    ax_dat = fig.add_subplot(2, 2, 2)
    im_dat = ax_dat.imshow(
        reduced_data_crop + 0.1,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap='viridis',
    )
    ax_dat.set_title('KLIP reduced data', fontsize=caracsize, pad=caracsize / 3.0)
    fig.colorbar(im_dat, ax=ax_dat, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=caracsize * 3 / 4.0
    )
    ax_dat.axis('off')

    # Bottom left: FM prediction
    ax_fm = fig.add_subplot(2, 2, 3)
    im_fm = ax_fm.imshow(
        fm_crop,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap='magma',
    )
    ax_fm.set_title('Best Model (FM)', fontsize=caracsize, pad=caracsize / 3.0)
    fig.colorbar(im_fm, ax=ax_fm, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=caracsize * 3 / 4.0
    )
    ax_fm.axis('off')

    # Bottom right: residuals
    ax_res = fig.add_subplot(2, 2, 4)
    im_res = ax_res.imshow(
        residuals_crop,
        origin='lower',
        vmin=int(np.round(vmin)),
        vmax=int(np.round(vmax)),
        cmap='viridis',
    )
    ax_res.set_title('Residuals', fontsize=caracsize, pad=caracsize / 3.0)
    fig.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=caracsize * 3 / 4.0
    )
    ax_res.axis('off')

    fig.subplots_adjust(hspace=0.25, wspace=0.35)
    # fig.suptitle(
    #     band_name + ': best WDH model and residuals',
    #     fontsize=5 / 4.0 * caracsize,
    #     y=0.985,
    # )
    fig.tight_layout()
    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.jpg'))
    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.pdf'))
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
