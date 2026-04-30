import numpy as np
from matplotlib import cm
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

viridis_g = cm.viridis.copy()
viridis_g.set_bad('0.5')
magma_g = cm.magma.copy()
magma_g.set_bad('0.5')

def plot_training(out_filename, reduced_data, freeform_fm_full, full_model_image, updates, mask_indices, ref_disk_median, ref_disk_std, min_percent=1.0, max_percent=99.9):
    print('Saving', out_filename, '...', end=' ')
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    fig, axs = plt.subplots(ncols=5, figsize=(15, 2.5))
    fig.subplots_adjust(left=0.05, right=0.95)
    mask_tmp = np.zeros(reduced_data.size)
    mask_tmp[mask_indices] = 1.0
    mask_good = (mask_tmp == 1.0).reshape(reduced_data.shape)
    mask_bad = (mask_tmp == 0.0).reshape(reduced_data.shape)
    data_vmax = np.percentile(reduced_data[mask_good], max_percent)
    data_vmin = -data_vmax
    # model_vmin, model_vmax = 0, np.percentile(full_model_image[mask_good], max_percent)
    model_max_val = np.percentile(full_model_image[mask_good], 99.9)
    model_vmin, model_vmax = 0, max(ref_disk_median + 3 * ref_disk_std, model_max_val)
    data_space_norm = simple_norm(reduced_data, 'linear', vmin=data_vmin, vmax=data_vmax)
    reduced_data_masked = np.array(reduced_data)
    reduced_data_masked[mask_bad] = np.nan
    plt.colorbar(axs[0].imshow(reduced_data_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    axs[0].set(title='Reduced data')
    axs[0].axis('off')

    freeform_fm_full_masked = np.array(freeform_fm_full)
    freeform_fm_full_masked[mask_bad] = np.nan
    plt.colorbar(axs[1].imshow(freeform_fm_full_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    axs[1].axis('off')
    axs[1].set(title='Freeform FM')

    model_diff = reduced_data_masked - freeform_fm_full_masked
    diff_vmax = np.nanmax(np.abs(model_diff))
    plt.colorbar(axs[2].imshow(model_diff, origin='lower', vmin=-diff_vmax, vmax=diff_vmax, cmap='RdBu_r'))
    axs[2].axis('off')
    axs[2].set(title='(Reduced) - (Freeform)')

    full_model_image_masked = np.array(full_model_image)
    full_model_image_masked[mask_bad] = np.nan
    axs[3].axis('off')
    plt.colorbar(axs[3].imshow(full_model_image_masked, origin='lower', norm=simple_norm(full_model_image, 'linear', vmin=model_vmin, vmax=model_vmax), cmap=magma_g))
    axs[3].set(title=r'Model')

    updates_vmax = np.max(np.abs(updates))
    plt.colorbar(axs[4].imshow(updates, origin='lower', vmax=updates_vmax, vmin=-updates_vmax, cmap='RdYlBu_r'))
    axs[4].set(title=r'Updates')
    axs[4].axis('off')
    fig.savefig(out_filename, dpi=256)
    plt.close(fig)
    print('Done.')
