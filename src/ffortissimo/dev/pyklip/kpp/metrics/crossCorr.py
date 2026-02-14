__author__ = 'JB'

from scipy.signal import correlate2d
import numpy as np
from copy import copy

def calculate_cc(image, PSF,spectrum = None, nans2zero=False):
    """
    Perform a cross correlation on the current loaded file.

    Args:
        image: image to get the cross correlation from.
        PSF: Template for the cross correlation
        spectrum: If not None, combine the 3D cube into a 2D image using a weighted mean with spectrum.
        nans2zero: If True, temporarily replace all nans values with zeros for the cross correlation

    Return: cross correlated image.
    """
    if np.size(image.shape) == 2:
        ny,nx = image.shape
    if np.size(image.shape) == 3:
        nl,ny,nx = image.shape

    if np.size(PSF.shape) == 2:
        ny_PSF,nx_PSF = PSF.shape
    if np.size(PSF.shape) == 3:
        nl_PSF,ny_PSF,nx_PSF = PSF.shape

    image_cpy = copy(image)

    if spectrum is not None:
        image_collapsed = np.zeros((ny,nx))
        for k in range(nl):
            image_collapsed = image_collapsed + spectrum[k]*image_cpy[k,:,:]
        image_cpy = image_collapsed/np.sum(spectrum)

    if nans2zero:
        where_nans = np.where(np.isnan(image_cpy))
        image_cpy = np.nan_to_num(image_cpy)

    # We have to make sure the PSF dimensions are odd because correlate2d shifts the image otherwise...
    if (nx_PSF % 2 ==0):
        PSF_tmp = np.zeros((ny_PSF,nx_PSF+1))
        PSF_tmp[0:ny_PSF,0:nx_PSF] = PSF
        PSF = PSF_tmp
        nx_PSF = nx_PSF +1
    if (ny_PSF % 2 ==0):
        PSF_tmp = np.zeros((ny_PSF+1,nx_PSF))
        PSF_tmp[0:ny_PSF,0:nx_PSF] = PSF
        PSF = PSF_tmp
        ny_PSF = ny_PSF +1

    # Check if the input file is 2D or 3D
    if np.size(image_cpy.shape) == 3: # If the file is a 3D cube
        image_convo = np.zeros(image_cpy.shape)
        for l_id in np.arange(nl):
            image_convo[l_id,:,:] = correlate2d(image_cpy[l_id,:,:],PSF,mode="same")
    else: # image is 2D
        image_convo = correlate2d(image_cpy,PSF,mode="same")

    if nans2zero:
        image_convo[where_nans] = np.nan

    return image_convo