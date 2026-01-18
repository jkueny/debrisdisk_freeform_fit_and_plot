import numba
import numpy as np
import math as mt

@numba.njit(fastmath=True,parallel=False)
def fastmodintegrand_dxdy_2g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha1, ci, si, k, rc, m, n):
    # author : Max Millar Blanchaer
    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg1 = k * alpha1 * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2 -
                                            (2 * g2 * cos_phi))**1.5

    hg = hg1 + hg2

    #Radial power low r propto -beta
    int1 = hg * ((d1/rc)**(-2*m) + (d1/rc)**(-2*n))**(-0.5)

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * (d1**beta))
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

@numba.njit(fastmath=True,parallel=False)
def fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                     g1, g2, alpha1, a_r, Rc, m, n,
                     y_arr,
                     z_arr,
                     npts,
                     mask
                     ):
    """ author : Max Millar Blanchaer
        modified by Johan Mazoyer
        create a 2g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio. These
        normalization avoid weird correlation in the parameters


    Args:
        dim: dimension of the image in pixel assuming square image
        param_disk: a dict with keywords: 
                R1: inner radius of the disk
                R2: outer radius of the disk
                beta: radial power law of the disk between R1 and R2
                aspect_ratio=0.1 vertical width of the disk
                g1: %, 1st HG param
                g2: %, 2nd HG param
                Aplha: %, relative HG weight
                inc: degree, inclination
                pa: degree, principal angle
                dx: au, + -> NW offset disk plane Minor Axis
                dy: au, + -> SW offset disk plane Major Axis
                offset: vertical residue image
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """


    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    # maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function at 90
    hg1_90 = k * alpha1 * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2)**1.5

    hg_90 = hg1_90 + hg2_90


    hmask = mask
    # hmask = mask[:,140:] #Use only half the mask

    for i, yp in enumerate(y_arr):
        for j, zp in enumerate(z_arr):
            integral_value = 0.

            # if hmask[j,npts/2+i]: #This assumes
            # that the input mask has is the same size as
            # the desired image (i.e. ~ size / sampling)
            if hmask[j, i]:

                image[j, i] = 0.  #np.nan

            else:

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                deltax = (R2 - (-R2)) / npts
                for kk in range(npts):
                    x = -R2 + kk * deltax
                    integral_value += fastmodintegrand_dxdy_2g(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2, g2_2, alpha1, ci, si, k,
                                             Rc, m, n)
                
                image[j, i] = integral_value

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    # image = image

    return image