import numpy as np
import math as mt

import numba

def mod_integrand_dxdy_1g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, ci, si, maxe, 
                      dx, dy, k, rc, m, n, Norm,
                      ):
    # author : Max Millar Blanchaer
    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #added 1/24/22 to diagnose divide by zero JKK
    # if mt.sqrt(d2) == 0:
    #     print(f'd1 = {d1}')
    #     print(f'd2 = {d2}')
    #     print(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,ci,si)

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi = np.arccos(cos_phi)

    #Henyey Greenstein function
    hg = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5



    #Radial power low r propto -beta
    int1 = hg * ((d1/rc)**(-2*m) + (d1/rc)**(-2*n))**(-0.5)

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * (d1**beta))
    expo = zz * zz / (hh * hh)

    if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
        return 0.0

    # A_phi = 1/(1+(dphi/w)**2)


    int2 = np.exp(expo)
    int3 = int2 * d2

    return int1 / int3# * A_phi

def integrand_dxdy_1g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, ci, si, maxe, dx, dy,
                      k):
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
    hg = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5


    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**(-beta)

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1**1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

def gen_disk_dxdy_1g(dim,
                     param_disk,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ author : Max Millar Blanchaer
        modified by Johan Mazoyer
        create a 1g SPF disk model. The disk is normalized at Norm at 90degree
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

    R1 = param_disk['r1']
    R2 = param_disk['r2']
    beta = param_disk['beta']

    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']

    g1 = param_disk['g1']

    aspect_ratio = param_disk['a_r']
    offset = param_disk['offset']

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  # HG g squared
    
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90
    hg_90 = k * (1. - g1_2) / (1. + g1_2)**1.5


    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

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

                image[j, i] = quad(integrand_dxdy_1g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-3,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, ci, si, maxe, dx, dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

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

                    image[j, i] = quad(integrand_dxdy_1g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-3,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, ci, si, maxe, dx,
                                             dy, k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    image = image + offset

    return image

def mod_gen_disk_dxdy_1g(dim,
                     param_disk,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ author : Max Millar Blanchaer
        modified by Johan Mazoyer
        create a 1g SPF disk model. The disk is normalized at Norm at 90degree
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

    R1 = param_disk['r1']
    R2 = param_disk['r2']
    rc = param_disk['rc']
    beta = param_disk['beta']
    m = param_disk['alpha_in']
    n = param_disk['alpha_out']
    # phi0 = param_disk['phi0'] - 180
    # w = param_disk['lwidth']

    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']

    g1 = param_disk['g1']

    aspect_ratio = param_disk['a_r']
    # aspect_ratio = 0.02
    # aspect_ratio = 0.01 #fixing this value because the plot from backend code doesn't like it in the param dict, JKK 01/30/22
    # offset = param_disk['offset']
    # offset = 0

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    # print(f'max_fov: {max_fov}; xsize: {xsize}; npts: {npts}')


    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-(xsize), (xsize), num=npts)
    z = np.linspace(-(xsize), (xsize), num=npts)


    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl_r = np.radians(90 - inc)
    ci = mt.cos(incl_r)  #Cosine of inclination
    si = mt.sin(incl_r)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    # if mt.isnan(ci):
    #     print(f'ci is nan. incl_r = {incl_r} and inc = {inc}')
    # if mt.isnan(si):
    #     print(f'si is nan. incl_r = {incl_r} and inc = {inc}')

    # if mt.isnan(cos_pa):
    #     print(f'cos_pa is nan. cos_pa: {cos_pa}')
    # if mt.isnan(sin_pa):
    #     print(f'sin_pa is nan. sin_pa: {sin_pa}')


    #HG g value squared
    g1_2 = g1 * g1  # HG g squared
    
    #Constant for HG function
    k = 1. / (4 * np.pi) * 100

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90
    hg_90 = k * (1. - g1_2) / (1. + g1_2)**1.5

    #Troubleshooting
    # phis = []


    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                #Switched the signs here 01/29/22 JKK
                yy = yp * cos_pa + zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa - zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                # if mt.isnan(zpsi):
                #     print(f'zpsi is nan. zpsi: {zpsi}')
                # if mt.isnan(zpci):
                #     print(f'zpci is nan. zpci: {zpci}')
                # if mt.isnan(zpsi_dx):
                #     print(f'zpsi_dx is nan. zpsi_dx: {zpsi_dx}')

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                image[j, i] = quad(mod_integrand_dxdy_1g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-3,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, ci, si, dx,
                                             dy, k, rc, m, n))[0]
                                       #args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             #R1, R2, beta, a_r, g1, g1_2, ci, si, maxe, dx,
                                             #dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes
                # that the input mask has is the same size as
                # the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    #Switched the signs here 01/29/22 JKK
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

                    #The azimuthal component? in degrees...
                    # phi = np.arctan((zpsi_dx)/(yy_dy))*180/np.pi
                    # phi = (np.arctan2(zpsi_dx,yy_dy)*180/np.pi)
                    # # phi = np.angle([zpsi_dx,yy_dy],deg=True)
                    # dphi = (phi - phi0)
                    # dphi = (dphi + 180) % (360) - 180
                    # if dphi<-180:
                    #     dphi = dphi + 360
                    # elif dphi > 180:
                    #     dphi = dphi - 360
                    # phis.append(dphi)
                    # phi = 90 - phi

                    image[j, i] = quad(mod_integrand_dxdy_1g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-3,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, ci, si, maxe, 
                                             dx,dy, k, rc, m, n, Norm#, dphi, w
                                             ))[0]
                                       #args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             #R1, R2, beta, a_r, g1, g1_2, ci, si, maxe, dx,
                                             #dy, k))[0]

    # print("Running time: ", datetime.now()-starttime)

    #Troubleshooting....
    # print(f'Range of phis: {min(phis)},{max(phis)}')
    # exit()

    # # normalize the HG function by the width
    # image = image / a_r

    # # normalize the HG function at the PA
    image = Norm * (image / a_r) / hg_90

    # add offset
    # image = image + offset
    image[hmask == 1] = 0

    return image


@numba.njit(fastmath=True,parallel=False)
def fastintegrand_dxdy_1g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, sig):
    # author : Max Millar Blanchaer
    # compute the scattering integrand
    # see analytic-disk.nb

    # xx = (xp * ci + zpsi_dx)
    xx = (xp + zpsi_dx)

    # d1 = mt.sqrt((yp_dy2 + xx * xx))
    d1 = mt.sqrt((yp_dy2 + zpsi_dx * zpsi_dx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    # d2 = xp * xp + yp2 + zp2
    # d2 = np.sqrt(xp * xp + yp2 + zp2)

    # #The line of sight scattering angle
    # cos_phi = xp / mt.sqrt(d2)

    # #Henyey Greenstein function
    # hg = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5


    #Radial power low r propto -beta
    # int1 = hg * (R1 / d1)**beta
    int1 = (R1 / d1)**beta
    # int1 = 1

    #The scale height function
    # zz = (zpci - xp * si)
    zz = (zpci)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)
    expo2 = d1 / sig

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo) * np.exp(0.5 * expo2)
    # Here is the 1/r^2 factor; in d2; r = sqrt(x^2 + y^2)
    int3 = int2# * d2

    return int1 / int3

@numba.njit(fastmath=True,parallel=False)
def fastgen_wind(ctrlrad, beta, pa, dx, dy, Norm,
                     a_r, sig,
                     y_arr,
                     z_arr,
                     npts,
                     mask= None,
                     left=False
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
    R1 = 0.1 # some small number that is likely behind the coronagraph.

    #Some things we can precompute ahead of time
    # maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = 0 #np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)



    hmask = mask
    # hmask = mask[:,140:] #Use only half the mask

    for i, yp in enumerate(y_arr):
        for j, zp in enumerate(z_arr):
            if hmask[j, i]:

                image[j, i] = 0.  #np.nan

            else:
                integral_value = 0.

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                if yy < 0 and left:
                    image[j,i] = 0.
                elif yy >= 0 and not left:
                    image[j,i] = 0.
                else:

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

                    deltax = (ctrlrad - (-ctrlrad)) / npts
                    for kk in range(npts):
                        x = -ctrlrad + kk * deltax
                        integral_value += fastintegrand_dxdy_1g(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                                R1, ctrlrad, beta, a_r, sig)
                    
                    image[j, i] = integral_value

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / np.max(image)
    # image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image

    # add offset
    # image += offset

    return image

@numba.njit(fastmath=True,parallel=False)
def fastintegrand_dxdy_2g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha1, ci, si, k):
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
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

@numba.njit(fastmath=True,parallel=False)
def fastgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                     g1, g2, alpha1, a_r,
                     y_arr,
                     z_arr,
                     npts,
                     mask=None,
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
                    integral_value += fastintegrand_dxdy_2g(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                            R1, R2, beta, a_r, g1, g1_2, g2, g2_2, alpha1, ci, si, k)
                
                image[j, i] = integral_value

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    # image = image

    return image

@numba.njit(fastmath=True,parallel=False)
def fastintegrand_dxdy_3g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha1, g3, g3_2, alpha2, ci, si, k):
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
    hg1 = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg3 = k * (1. - g3_2) / (1. + g3_2 - (2 * g3 * cos_phi))**1.5

    hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3

    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

@numba.njit(fastmath=True,parallel=False)
def fastgen_disk_dxdy_3g(R1, R2, beta, inc, pa, dx, dy, Norm,
                     g1, g2, alpha1, g3, alpha2, a_r,
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
    g3_2 = g2 * g2 * g3  #Third HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function at 90
    hg1_90 = k * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1. - g2_2) / (1. + g2_2)**1.5
    hg3_90 = k * (1. - g3_2) / (1. + g3_2)**1.5

    hg_90 = alpha1 * hg1_90 + alpha2 * hg2_90 + (1 - alpha1 - alpha2) * hg3_90


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
                    integral_value += fastintegrand_dxdy_3g(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r,
                                             g1, g1_2, g2, g2_2, alpha1, g3, g3_2, alpha2,
                                             ci, si, k)
                
                image[j, i] = integral_value

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    # image = image

    return image

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


@numba.njit(fastmath=True,parallel=False)
def fastmodintegrand_dxdy_3g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha1, g3, g3_2, alpha2,
                      ci, si, k, rc, m, n):
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

    #Henyey Greenstein function at 90
    hg1_90 = k * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1. - g2_2) / (1. + g2_2)**1.5
    hg3_90 = k * (1. - g3_2) / (1. + g3_2)**1.5

    hg = alpha1 * hg1_90 + alpha2 * hg2_90 + (1 - alpha1 - alpha2) * hg3_90

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
def fastmodgen_disk_dxdy_3g(R1, R2, beta, inc, pa, dx, dy, Norm,
                     g1, g2, alpha1, g3, alpha2, a_r, Rc, m, n,
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
    g3_2 = g2 * g2 * g3  #Third HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function at 90
    hg1_90 = k * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1. - g2_2) / (1. + g2_2)**1.5
    hg3_90 = k * (1. - g3_2) / (1. + g3_2)**1.5

    hg_90 = alpha1 * hg1_90 + alpha2 * hg2_90 + (1 - alpha1 - alpha2) * hg3_90


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
                    integral_value += fastmodintegrand_dxdy_3g(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2, g2_2, alpha1, g3, g3_2, alpha2,
                                             ci, si, k, Rc, m, n)
                
                image[j, i] = integral_value

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    # image = image

    return image

@numba.njit(fastmath=True)
def fastintegrand_dxdy_custom(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, sc_angs_rad, recon_spf, ci, si):
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
    phi = mt.acos(cos_phi)

    if (phi < sc_angs_rad[0] or phi > sc_angs_rad[-1]):
        return 0.0
    spf = np.interp(x=phi,xp=sc_angs_rad,fp=recon_spf,
                    # left=np.nan, right=np.nan,
                    )

    #Radial power low r propto -beta
    int1 = spf * (R1 / d1)**beta
    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

@numba.njit(fastmath=True)
def fastgen_disk_dxdy_custom(R1, R2, beta, inc, pa, dx, dy, Norm, a_r,
                     scatt_angs,
                     spf_model,
                     y_arr,
                     z_arr,
                     npts,
                     mask):
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
    image = np.zeros((npts, npts), dtype=np.float32)


    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)


    spf_90 = spf_model[np.argmin(np.abs(scatt_angs - (np.pi/2)))]
    # spf_model /= spf_90
    spf_model -= (spf_90 - 1)

    # start = datetime.now()
    hmask = mask
    # hmask = mask[:,65:] #Use only half the mask

    for i, yp in enumerate(y_arr):
        for j, zp in enumerate(z_arr):
            integral_value = 0

            # if hmask[j,npts/2+i]: #This assumes
            # that the input mask has is the same size as
            # the desired image (i.e. ~ size / sampling)
            if hmask[j, i]:

                image[j, i] = 0.  #np.nan

            else:

                #This rotates the coordinates in the image frame
                #switched signs here JKK 01/31/22
                #switching back just know positive is N-W instead of N-E, 02/21/22 JKK
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
                for k in range(npts):
                    x = -R2 + k * deltax
                    integral_value += fastintegrand_dxdy_custom(x, yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                            R2, beta, a_r, scatt_angs, spf_model, ci, si)
                image[j, i] = integral_value

    image = Norm * (image / a_r)

    return image

@numba.njit(fastmath=True)
def fastmodintegrand_custom(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, Rc, m, n,
                      a_r, sc_angs_rad, recon_spf, ci, si):
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
    phi = mt.acos(cos_phi)

    if (phi < sc_angs_rad[0] or phi > sc_angs_rad[-1]):
        return 0.0
    spf = np.interp(x=phi,xp=sc_angs_rad,fp=recon_spf,
                    # left=np.nan, right=np.nan,
                    )

    #Radial power low r propto -beta
    int1 = spf * ((d1/Rc)**(-2*m) + (d1/Rc)**(-2*n))**(-0.5)
    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3

@numba.njit(fastmath=True)
def fastmodgen_disk_custom(R1, R2, Rc, m, n, inc, pa, dx, dy, Norm, a_r,
                     scatt_angs,
                     spf_model,
                     y_arr,
                     z_arr,
                     npts,
                     mask):
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
    image = np.zeros((npts, npts))


    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)


    spf_90 = spf_model[np.argmin(np.abs(scatt_angs - (np.pi/2)))]
    # spf_model /= spf_90
    spf_model -= (spf_90 - 1)

    # start = datetime.now()
    hmask = mask
    # hmask = mask[:,65:] #Use only half the mask

    for i, yp in enumerate(y_arr):
        for j, zp in enumerate(z_arr):
            integral_value = 0

            # if hmask[j,npts/2+i]: #This assumes
            # that the input mask has is the same size as
            # the desired image (i.e. ~ size / sampling)
            if hmask[j, i]:

                image[j, i] = 0.  #np.nan

            else:

                #This rotates the coordinates in the image frame
                #switched signs here JKK 01/31/22
                #switching back just know positive is N-W instead of N-E, 02/21/22 JKK
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
                for k in range(npts):
                    x = -R2 + k * deltax
                    integral_value += fastmodintegrand_custom(xp=x, yp_dy2=yy_dy2, yp2=y2, zp=zp, zp2=z2,
                                                            zpsi_dx=zpsi_dx, zpci=zpci, R1=R1,R2=R2,
                                                            Rc=Rc, m=m, n=n, a_r=a_r, sc_angs_rad=scatt_angs,
                                                            recon_spf=spf_model, ci=ci, si=si)
                image[j, i] = integral_value

    image = Norm * (image / a_r)

    return image