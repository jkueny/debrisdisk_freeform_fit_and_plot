# pylint: disable=C0103
"""
a set of functions made to measure the psf specifically for GPI IFS disk data
check the sat spots and measure PSFs
author: Johan Mazoyer
"""

import os
import warnings

import numpy as np
import scipy.ndimage.filters as scipy_filters

import dev.pyklip.klip as klip

import utils.astro_unit_conversion as convert
import astropy.io.fits as fits


########################################################
def make_disk_mask(dim,
                   estimPA,
                   estiminclin,
                   estimminr,
                   estimmaxr,
                   aligned_center=[140., 140.],
                   half_n_only=False,
                   half_s_only=False):
    """ make a zeros mask for a disk. usind a set of parameters


    Args:
        dim: pixel, dimension of the square mask
        estimPA: degree, estimation of the PA
        estiminclin: degree, estimation of the inclination
        estimminr: pixel, inner radius of the mask
        estimmaxr: pixel, outer radius of the mask
        aligned_center: [pixel,pixel], position of the star in the mask

    Returns:
        a [dim,dim] array where the mask is at 0 and the rest at 1
    """
    #TODO: Maybe one day think carefully on make it work for non-square mask,
    # but honneslty not urgent
    if estimminr < 0:
        estimminr = 0
    PA_rad = np.radians(90 + estimPA)
    x = np.arange(dim, dtype=float)[None, :] - aligned_center[0]
    y = np.arange(dim, dtype=float)[:, None] - aligned_center[1]

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(np.radians(estiminclin))
    rho2dellip = np.sqrt(x**2 + y**2)
    
    mask_object_astro_zeros = np.ones((dim, dim))
    mask_object_astro_zeros[np.where((rho2dellip > estimminr)
                                     & (rho2dellip < estimmaxr))] = 0.
    if half_n_only:
        mask_object_astro_zeros[np.where(x < 0)] = 1.
    if half_s_only:
        mask_object_astro_zeros[np.where(x >= 0)] = 1.
    return mask_object_astro_zeros


def check_satspots_disk_intersection(dataset, params_mcmc_yaml, quiet=True):
    """ check in which image the disk intereset the satspots for
    GPI IFS data
    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        quiet: in False, print each rejected image and for which satspots

    Returns:
        a string list of the files for which the disk interesects the satspots
    """

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    dimx = dataset.input.shape[1]
    dimy = dataset.input.shape[2]
    pixscale_ins = params_mcmc_yaml['PIXSCALE_INS']
    distance_star = params_mcmc_yaml['DISTANCE_STAR']

    nfiles = int(np.nanmax(dataset.filenums)) + 1  # Get the number of files

    estimPA = params_mcmc_yaml['pa_init']
    estiminclin = params_mcmc_yaml['inc_init']
    estimminr = convert.au_to_pix(params_mcmc_yaml['r1_init'], pixscale_ins,
                                  distance_star)
    estimmaxr = convert.au_to_pix(params_mcmc_yaml['r2_init'], pixscale_ins,
                                  distance_star)

    ### Where is the disk ?
    # create ones masks for the disk
    mask_object_astro_ones = 1 - make_disk_mask(dimx,
                                                estimPA,
                                                estiminclin,
                                                estimminr - 3/np.cos(np.radians(params_mcmc_yaml['inc_init'])),
                                                estimmaxr + 3/np.cos(np.radians(params_mcmc_yaml['inc_init'])),
                                                aligned_center=aligned_center)

    # fits.writeto("/Users/jmazoyer/Desktop/initial_model.fits",
    #                          mask_object_astro_ones,
    #                          overwrite=True)
    filename_disk_intercept_satspot = []

    for i in range(dataset.input.shape[0]):

        filename_here = dataset.filenames[i]

        if filename_here in filename_disk_intercept_satspot:
            continue

        PA_here = dataset.PAs[i]
        Starpos = dataset.centers[i]
        wls = dataset.wvs[i]
        hdrindex = dataset.filenums[i]
        slice_here = dataset.wv_indices[i]

        model_mask_rot = np.round(
            np.abs(
                klip.rotate(mask_object_astro_ones,
                            PA_here,
                            aligned_center,
                            new_center=Starpos,
                            flipx=True)))

        # now grab the position of the sat spots by parsing the header
        hdr = dataset.exthdrs[hdrindex]

        spot0 = hdr['SATS{wave}_0'.format(wave=slice_here)].split()
        spot1 = hdr['SATS{wave}_1'.format(wave=slice_here)].split()
        spot2 = hdr['SATS{wave}_2'.format(wave=slice_here)].split()
        spot3 = hdr['SATS{wave}_3'.format(wave=slice_here)].split()

        for j, spot in enumerate([spot0, spot1, spot2, spot3]):
            posx = float(spot[0])
            posy = float(spot[1])

            x_sat = np.arange(dimx, dtype=np.float)[None, :] - posx
            y_sat = np.arange(dimy, dtype=np.float)[:, None] - posy
            rho2d_sat = np.sqrt(x_sat**2 + y_sat**2)
            wh_sat_spot = np.where((rho2d_sat < 3 / 1.6 * wls))

            is_on_the_disk = np.sum(model_mask_rot[wh_sat_spot]) > 0
            if is_on_the_disk:
                # if is_on_the_disk and wls > 1.6:
                #     model_mask_rot[wh_sat_spot] = 1
                #     fits.writeto("/Users/jmazoyer/Desktop/toto.fits",
                #                  model_mask_rot * dataset.input[i],
                #                  overwrite=True)
                #     fits.writeto("/Users/jmazoyer/Desktop/tutu.fits",
                #                  dataset.input[i],
                #                  overwrite=True)
                #     asd
                # print(filename_here,np.sum(model_mask_rot[wh_sat_spot]))
                if not quiet:
                    _, head = os.path.split(filename_here)
                    print(head, 'removed because of the sat spot #' + str(j))
                filename_disk_intercept_satspot.append(filename_here)
                break

    if filename_disk_intercept_satspot:
        print(file_prefix + ': We remove ' +
              str(len(filename_disk_intercept_satspot)) + ' files (out of ' +
              str(nfiles) + ') for the psf measurement' +
              ' because sat spots intersected the disk')
    else:
        print(
            'The disk never intersects the satspots, all satspots kept for the psf measurement'
        )
    return filename_disk_intercept_satspot


def check_satspots_snr(dataset_multi_wl, params_mcmc_yaml, quiet=True):
    """ check the SNR of the PSF created for each slice in GPI IFS.
        If too small (<3), we return the list of the PSF to reject.
    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
         quiet: if false print the SNR of each PSF for each color

    Returns:
        the PSF
    """

    wls = np.unique(dataset_multi_wl.wvs)
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    dimx = dataset_multi_wl.input.shape[1]
    dimy = dataset_multi_wl.input.shape[2]

    boxrad_here = 20

    # create a triangle nan mask for the bright regions in 2015 in some data probably due
    # to the malfunctionning diode (I don't think this has described in an article before)
    if (file_prefix == 'K2band_hr4796') or (file_prefix == 'K1band_hr4796'):
        x_image = np.arange(dimx, dtype=np.float)[None, :] - aligned_center[0]
        y_image = np.arange(dimy, dtype=np.float)[:, None] - aligned_center[1]
        triangle1 = 0.67 * x_image + y_image - 114.5
        triangle2 = -3.2 * x_image + y_image - 330

        mask_triangle1 = np.ones((dimx, dimy))
        mask_triangle2 = np.ones((dimx, dimy))

        mask_triangle1[np.where((triangle1 > 0))] = np.nan
        mask_triangle2[np.where((triangle2 > 0))] = np.nan
        mask_triangle = mask_triangle1 * mask_triangle2
    else:
        mask_triangle = np.ones((dimx, dimy))

    dataset_multi_wl.input = dataset_multi_wl.input * mask_triangle
    dataset_multi_wl.generate_psfs(boxrad=boxrad_here)

    snr = wls * 0.
    for j, psf in enumerate(dataset_multi_wl.psfs):
        y_img, x_img = np.indices(psf.shape, dtype=float)
        r_img = np.sqrt((x_img - psf.shape[0] // 2)**2 +
                        (y_img - psf.shape[1] // 2)**2)
        noise_annulus = np.where((r_img > 9 / 1.6 * wls[j])
                                 & (r_img <= 12 / 1.6 * wls[j]))
        signal_aperture = np.where(r_img <= 3 / 1.6 * wls[j])

        # psf[noise_annulus] = 1
        # psf[signal_aperture] = 1

        snr[j] = np.nanmean(psf[signal_aperture]) / np.nanstd(
            psf[noise_annulus])
        if not quiet:
            print(file_prefix +
                  ': SNR of time-averaged satspots at wl {0:.2f} is {1:.2f}'.
                  format(wls[j], snr[j]))

    bad_sat_spots = np.where(snr < 3)
    bad_sat_spots_list = bad_sat_spots[0].tolist()
    if bad_sat_spots_list:
        print(file_prefix +
              ': PSFs # {0} have SNR < 3: these WLs are removed'.format(
                  bad_sat_spots_list))
    else:
        print(file_prefix + ': all PSFs have high enough SNRs')
    return bad_sat_spots[0].tolist()


def make_collapsed_psf(dataset, params_mcmc_yaml, boxrad=10, collapse_channels = 1, smoothed = True):
    """ create a PSF from the satspots, with a smoothed box
    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        boxrad: size of the PSF. Must be larger than 12 to have the box
        collapse_channels (int): number of output channels to evenly-ish collapse the 
                        dataset into. Default is 1 (broadband). Collapsed is done the same
                        way as in pyklip spectral_collapse function
        smoothed: if true, multiply by a mask to smoothed the edges to avoid artifacts when c
                    convolving


    Returns:
        the PSF
    """

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    dimx = dataset.input.shape[1]
    dimy = dataset.input.shape[2]

    # create a traingle nan mask for the bright regions in 2015 probably due
    # to the malfunctionning diode
    if (file_prefix == 'K2band_hr4796') or (file_prefix == 'K1band_hr4796'):
        x_image = np.arange(dimx, dtype=np.float)[None, :] - aligned_center[0]
        y_image = np.arange(dimy, dtype=np.float)[:, None] - aligned_center[1]
        triangle1 = 0.67 * x_image + y_image - 114.5
        triangle2 = -3.2 * x_image + y_image - 330

        mask_triangle1 = np.ones((dimx, dimy))
        mask_triangle2 = np.ones((dimx, dimy))

        mask_triangle1[np.where((triangle1 > 0))] = np.nan
        mask_triangle2[np.where((triangle2 > 0))] = np.nan
        mask_triangle = mask_triangle1 * mask_triangle2
    else:
        mask_triangle = np.ones((dimx, dimy))

    dataset.input = dataset.input * mask_triangle

    dataset.generate_psfs(boxrad=boxrad)
    psfs_here = dataset.psfs

    numwvs = dataset.numwvs
    slices_per_group = numwvs // collapse_channels  # how many wavelengths per each output channel
    leftover_slices = numwvs % collapse_channels

    return_psf = np.zeros(
        [collapse_channels, psfs_here.shape[1], psfs_here.shape[2]])

    # populate the output image
    next_start_channel = 0  # initialize which channel to start with for the input images
    for i in range(collapse_channels):
        # figure out which slices to pick
        slices_this_group = slices_per_group
        if leftover_slices > 0:
            # take one extra slice, yummy
            slices_this_group += 1
            leftover_slices -= 1

        i_start = next_start_channel
        i_end = next_start_channel + slices_this_group  # this is the index after the last one in this group

        psfs_this_group = psfs_here[i_start:i_end, :, :]

        # Remove annoying RuntimeWarnings when input_4d is all nans
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return_psf[i, :, :] = np.nanmean(psfs_this_group, axis=0)
        next_start_channel = i_end

    if smoothed:
        r_smooth = boxrad - 1
        # # create rho2D for the psf square
        x_square = np.arange(2 * boxrad + 1, dtype=np.float)[None, :] - boxrad
        y_square = np.arange(2 * boxrad + 1, dtype=np.float)[:, None] - boxrad
        rho2d_square = np.sqrt(x_square**2 + y_square**2)

        smooth_mask = np.ones((2 * boxrad + 1, 2 * boxrad + 1))
        smooth_mask[np.where(rho2d_square > r_smooth - 1)] = 0.
        smooth_mask = scipy_filters.gaussian_filter(smooth_mask, 2.)
        smooth_mask[np.where(rho2d_square < r_smooth)] = 1.
        smooth_mask[np.where(smooth_mask < 0.01)] = 0.
        for i in range(collapse_channels):
            return_psf[i, :, :] *= smooth_mask

    # return_psf = return_psf / np.nanmax(return_psf)
    return_psf[np.where(return_psf < 0.)] = 0.
    return return_psf
