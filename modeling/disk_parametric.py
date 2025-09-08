"""
Parametric disk model class for fitting a quick and simple scattered-light
disk model to a KLIP reduced image. Using scipy.optimize.minimize.L-BFGS-B.

This simple fitted model is used to generate a reference power spectrum for
regularizing the freeform model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
from utils.model_tools import convolve_model
from modeling.disk_freeform import FreeFormDisk
from utils.io.yaml_handling import read_config
from scipy.special import huber
from scipy.optimize import minimize

class ParametricDisk(FreeFormDisk):
    def __init__(self, config):
        self.params_file = read_config(config)
        self.params_init = self._get_initial_params()
        self._load_dirs()
        self._load_metadata()
        self._load_klparams()


    def _get_initial_params(self):
        return self.params_init

    def _load_dirs(self):
        super()._load_dirs()

    def _load_metadata(self):
        super()._load_metadata()
    def _load_klparams(self):
        super()._load_klparams()

    def _get_initial_params(self):
        super()._get_initial_params()

    def _load_disk_model_params(self):
        disk_params = {}
        disk_params["r1"] = self.params_file["r1_init"]
        disk_params["r2"] = self.params_file["r2_init"]
        disk_params["rc"] = self.params_file["rc_init"]
        disk_params["alpha_in"] = self.params_file["alpha_in_init"]
        disk_params["alpha_out"] = self.params_file["alpha_out_init"]
        disk_params["beta"] = self.params_file["beta_init"]
        disk_params["a_r"] = self.params_file["a_r_init"]
        disk_params["inc"] = self.params_file["inc_init"]
        disk_params["pa"] = self.params_file["pa_init"]
        disk_params["dx"] = self.params_file["dx_init"]
        disk_params["dy"] = self.params_file["dy_init"]
        disk_params["Norm"] = self.params_file["N_init"]
        disk_params["g1"] = self.params_file["g1_init"]
        disk_params["g2"] = self.params_file["g2_init"]
        disk_params["alpha1"] = self.params_file["alpha1_init"]

        self.disk_params = disk_params    

    def _load_dirs(self):
        super()._load_dirs()


    def render_initial_disk_model(self):

        self._load_disk_model_params()

        beta = self.disk_params["beta"]
        a_r = self.disk_params["a_r"]
        inc = self.disk_params["inc"]
        pa = self.disk_params["pa"]
        dx = self.disk_params["dx"]
        dy = self.disk_params["dy"]

        R1 = self.disk_params['r1']
        R2 = self.disk_params['r2']

        Norm = self.disk_params['Norm']
        g1 = self.disk_params['g1']
        g2 = self.disk_params['g2']
        alpha1 = self.disk_params['alpha1']

        max_fov = self.image_size / 2. * self.pixscale  #maximum radial distance in AU from the center to the edge
        n_pts = int(np.floor(self.image_size / 1))
        xsize = max_fov * self.distance  #maximum radial distance in AU from the center to the edge

        #The coordinate system here [x,y,z] is defined :
        # +ve x is the line of sight
        # +ve y is going right from the center
        # +ve z is going up from the center

        # y = np.linspace(0,xsize,num=npts/2)
        y = np.linspace(-xsize, xsize, num=n_pts)
        z = np.linspace(-xsize, xsize, num=n_pts)
        
        beta = 1.
        rc = self.disk_params['rc']
        m = self.disk_params['alpha_in']
        n = self.disk_params['alpha_out']
        model = fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                                    g1, g2, alpha1, a_r, rc, m, n,
                                    y_arr=y,
                                    z_arr=z,
                                    npts=n_pts,
                                    mask=(1 - self.mask2generatedisk))
        return model
    
    def _render_disk_model(self, params, mask2generatedisk):
        # Unpack the parameters
        beta = params["beta"]
        a_r = params["a_r"]
        inc = params["inc"]
        pa = params["pa"]
        dx = params["dx"]
        dy = params["dy"]

        R1 = params['r1']
        R2 = params['r2']

        Norm = params['Norm']
        g1 = params['g1']
        g2 = params['g2']
        alpha1 = params['alpha1']

        max_fov = self.image_size / 2. * self.pixscale  #maximum radial distance in AU from the center to the edge
        n_pts = int(np.floor(self.image_size / 1))
        xsize = max_fov * self.distance  #maximum radial distance in AU from the center to the edge

        #The coordinate system here [x,y,z] is defined :
        # +ve x is the line of sight
        # +ve y is going right from the center
        # +ve z is going up from the center

        # y = np.linspace(0,xsize,num=npts/2)
        y = np.linspace(-xsize, xsize, num=n_pts)
        z = np.linspace(-xsize, xsize, num=n_pts)
        
        beta = 1.
        rc = params['rc']
        m = params['alpha_in']
        n = params['alpha_out']
        model = fastmodgen_disk_dxdy_2g(R1, R2, beta, inc, pa, dx, dy, Norm,
                                    g1, g2, alpha1, a_r, rc, m, n,
                                    y_arr=y,
                                    z_arr=z,
                                    npts=n_pts,
                                    mask=(1 - mask2generatedisk)) 
        return model
    def _objective_function(self, params):
        '''
        Objective function for the simple disk model fit.
        '''
        model = self._render_disk_model(params, self.mask2generatedisk)
        # Convolve the disk model with the PSF
        psf = self.psf
        model_image = convolve_model(model, psf)

        weights = 1. / self.noise_map
        raw_loss = (model_image - self.data)**2 * weights
        mean_huber = np.mean(huber(delta=1, r=raw_loss))

        return mean_huber
    
    def fit_simple_disk_model(self):
        '''
        Use scipy.optimize.minimize to fit a simple disk model to the data.
        '''
        # Use scipy.optimize.minimize to fit a simple disk model to the data.
        self.iteration = 0
        x0 = np.asarray([self.params_init["r1"],
                         self.params_init["r2"],
                         self.params_init["rc"],
                         self.params_init["alpha_in"],
                         self.params_init["alpha_out"],
                         self.params_init["beta"],
                         self.params_init["a_r"],
                         self.params_init["inc"],
                         self.params_init["pa"],
                         self.params_init["dx"],
                         self.params_init["dy"],
                         self.params_init["Norm"],
                         self.params_init["g1"],
                         self.params_init["g2"],
                         self.params_init["alpha1"]])
        result = minimize(self._objective_function, x0,
                          method="L-BFGS-B",
                          options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8},
                          callback=self._callback_function)
        self.params_opt = result.x
        best_model = self._render_disk_model(self.params_opt, self.mask2generatedisk)
        self.best_model = best_model
        return best_model
    
    def _callback_function(self, params):
        '''
        Callback function for the simple disk model fit.
        '''
        print(f"Iteration {self.iteration}: {params}")
        self.iteration += 1