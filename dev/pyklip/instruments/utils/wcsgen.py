import numpy as np
from astropy import wcs

def generate_wcs(parangs, center, flipx=False, platescale=0.010, radec=None):
    """
    Generates a "good-enough" astropy.wcs object based on the parallactic angle and the center

    Args:
        parangs (float): angle to rotate the image by to align North up (degrees). Typically, this is dataset.PAs
        center (tuple): [x,y] pixel location for the star center
        flipx (bool): if True, the image is in a right-handed (North-up, East-right) coordinate system. Default is False.
        platescale (float): platescale of image in arcsec/pixel. This is just cosmetic, as pyKLIP only uses wcs for rotation.
        radec (tuple): RA and Dec coordinates of the object (degrees)

    Returns:
        astropy.wcs.WCS object with the orientation of each image
    """
    vert_ang = np.radians(parangs)
    if flipx:
        negsign = -1
    else:
        negsign = 1
    pc = np.array([[-negsign * np.cos(vert_ang), negsign * np.sin(vert_ang)],[np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * platescale /3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    
    new_hdr['CRPIX1'] = center[0]
    new_hdr['CRPIX2'] = center[1]

    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'

    new_hdr['CDELT1'] = platescale / 3600
    new_hdr['CDELT2'] = platescale / 3600

    if radec is not None:
        new_hdr['CRVAL1'] = radec[0]
        new_hdr['CRVAL2'] = radec[1]
    else:
        new_hdr['CRVAL1'] = 0 
        new_hdr['CRVAL2'] = 0

    w = wcs.WCS(new_hdr)

    return w
    