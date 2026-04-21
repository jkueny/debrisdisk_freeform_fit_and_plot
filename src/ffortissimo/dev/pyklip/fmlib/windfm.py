# pylint: disable=C0103
from sys import version_info
from os import path
import multiprocessing as mp
from copy import deepcopy
import ctypes

import h5py

import numpy as np

from ffortissimo.dev.pyklip.fmlib.nofm import NoFM
import ffortissimo.dev.pyklip.fm as fm
from ffortissimo.dev.pyklip.klip import rotate, rotate_image

def fm_from_eigen_jit(
    klmodes,
    evals,
    evecs,
    input_img_num,
    ref_psfs_indicies,
    section_ind,
    target_img,
    ref_imgs,
    model_disks,
):
    """
    Forward-model a single section using JIT-compiled inner kernels.

    This is the fast path used by :meth:`WindFM.fm_parallelized_jit` for the
    typical MCMC iteration: ``load_from_basis=True``, no RDI, single
    ``numbasis``, no NaNs in the rotated WDH model. Numerics live in two
    ``@numba.njit`` helpers exported from ``fm.py``:

        - ``fm.perturb_jit``                 -> perturbed KL modes (delta_KL)
        - ``fm.calculate_fm_single_2d_jit``  -> forward-modeled PSF section

    The orchestrator itself is **not** ``@njit``: it does only array slicing
    plus two calls into the JIT'd kernels, which is where ~all of the wall
    time lives. Decorating the orchestrator would force us to either also
    JIT ``calculate_fm_singleNumbasis`` (with its complex spectral/multi-
    basis branches) or duplicate that logic; both options carry significant
    maintenance cost for negligible speedup.

    Args:
        klmodes: unperturbed KL modes, shape (max_basis, N_pix)
        evals: covariance eigenvalues, shape (max_basis,)
        evecs: covariance eigenvectors, shape (N_ref, max_basis)
        input_img_num: index of the science frame
        ref_psfs_indicies: 1D array of reference-PSF indices for this section
        section_ind: index array into the flattened image for this section
                     (used as ``section_ind[0]`` per pyklip convention)
        target_img: the science section pixels, shape (N_pix,)
        ref_imgs: the reference section pixels, shape (N_ref, N_pix)
        model_disks: ``self.model_wdhs``, shape (N_frames, N_image_pix);
                     this function slices the science and reference rows
                     internally.

    Returns:
        fm_psf: forward-modelled PSF section as a 1D numpy array of length
                N_pix. Caller passes this directly to
                ``fm._save_rotated_section`` as the ``sector`` argument.
    """

    # Section slices off self.model_wdhs. We materialize *contiguous*
    # float64 copies up front because:
    #   1. The inner @njit kernels emit a NumbaPerformanceWarning on
    #      strided non-contiguous BLAS inputs (the model_disks slices are
    #      strided views).
    #   2. Numba dispatches to BLAS more efficiently on C-contiguous arrays.
    # The cost of np.ascontiguousarray here is well under the cost of
    # `perturb_jit`'s O(N_ref^2 * N_pix) outer products that follow.
    section = section_ind[0]
    model_sci = np.ascontiguousarray(
        model_disks[input_img_num, section], dtype=np.float64
    )
    model_ref = np.ascontiguousarray(
        model_disks[ref_psfs_indicies, :][:, section], dtype=np.float64
    )

    sci = np.ascontiguousarray(target_img, dtype=np.float64)
    refs = np.ascontiguousarray(ref_imgs, dtype=np.float64)
    klmodes_c = np.ascontiguousarray(klmodes, dtype=np.float64)
    evals_c = np.ascontiguousarray(evals, dtype=np.float64)
    evecs_c = np.ascontiguousarray(evecs, dtype=np.float64)

    # 1) Perturb the KL modes from the proposed model (the BLAS-heavy step).
    delta_KL = fm.perturb_jit(evals_c, evecs_c, klmodes_c, refs, model_ref)

    # 2) Compose the forward-modelled PSF for this section.
    #    Returns a 1D (N_pix,) array directly — the previous tuple-return
    #    plumbing (`pk_psf[0]`-into-`_save_rotated_section`) relied on
    #    accidental shape broadcasting and was confusing.
    return fm.calculate_fm_single_2d_jit(delta_KL, klmodes_c, sci, model_sci)


class WindFM(NoFM):
    """Defining a model disk to which we apply the Forward Modelling. There are 3 ways:

            * "Save Basis mode" (save_basis=true), we are preparing to save the FM basis
            * "Load Basis mode" (load_from_basis = true), most of the parameters are
              derived from the previous fm.klip_dataset which measured FM basis.
            * "Simple FM mode" (save_basis = load_from_basis = False). Just
              for a unique disk FM.


        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, x, y)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            dataset: an instance of Instrument.Data. We need it to know the
                     parameters to "prepare" first inital model.
            model_disk: a model of the disk of size (wvs, x, y) or (x, y)
            basis_filename: filename to save and load the KL basis. Filenames can haves
                            2 recognizable extensions: .h5 or .pkl. We strongly
                            recommand .h5 as pickle have problem of compatibility
                            between python 2 and 3 and sometimes between computer
                            (e.g. KL modes not readable on another computer)
            load_from_basis: if True, load the KL basis at basis_filename. It only need
                             to be done once, after which you can measure FM with
                             only update_model()
            save_basis: if True, save the KL basis at basis_filename. If load_from_basis
                            is True, save_basis is automatically set to False, it is
                            useless to load and save the matrix at the same time.
            aligned_center: array of 2 elements [x,y] that all the model will be
                            centered on for image registration.
                            FIXME: This is the most problematic thing currently, the
                            aligned_center of the model and of the images can be set
                            independently, which will create false results.
                            - In "Load Basis mode", this parameter is not read, we just
                            use the aligned_center set for the images in the previous
                            fm.klip_dataset and save in basis_filename
                            - In "Save Basis mode", or "Simple FM mode" we define it
                            and then check that it is the same one used for the images
                            in fm.klip_dataset
            mode: deprecated parameter, ignored here and defined in fm.klip_dataset
            annuli: deprecated parameter, ignored here and defined in fm.klip_dataset
            subsections: deprecated parameter, ignored here and defined
                         in fm.klip_dataset
            numthreads: deprecated parameter. All centering are done in fm.klip_dataset 

        Returns:
            A DiskFM Object

    """
    def __init__(self,
                 inputs_shape,
                 numbasis,
                 dataset,
                 model_wdh_list,
                 model_pas_mask,
                 basis_filename="",
                 kl_basis_file=None,
                 load_from_basis=False,
                 save_basis=False,
                 aligned_center=None,
                 psf_library=None,
                 mode=None,
                 annuli=None,
                 subsections=None,
                 numthreads=None):
        """

            Initilaizes the DiskFM class

        """

        if load_from_basis:
            numbasis = 1
            inputs_shape = 1
            dataset = None

        # make sure the dimensions have the good shape
        # and that they are numpy arrays to access their shape
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

        if hasattr(inputs_shape, "__len__"):
            inputs_shape = np.array(inputs_shape)
        else:
            inputs_shape = np.array([inputs_shape])

        if mode is not None:
            print(mode)
            print("Warning: Argument 'mode' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized]. ")

        if numthreads is not None:
            print("Warning: Argument 'numthreads' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "All centering are done in fm.klip_dataset ")

        if annuli is not None:
            print("Warning: Argument 'annuli' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized].")

        if subsections is not None:
            print("Warning: Argument 'subsections' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized].")

        super(WindFM, self).__init__(inputs_shape, numbasis)

        self.supports_rdi = True # temporary flag until all FM classes supports RDI. 

        self.data_type = ctypes.c_double

        self.basis_filename = basis_filename
        self.save_basis = save_basis
        self.load_from_basis = load_from_basis

        if self.load_from_basis:
            # Its useless to save and load at the same time.
            self.save_basis = False
            save_basis = False

        if self.save_basis is True:
            # manager = mp.Manager()
            # self.klmodes_dict = manager.dict()
            # self.evecs_dict = manager.dict()
            # self.evals_dict = manager.dict()
            # self.aligned_images_dict = manager.dict()
            # self.ref_psfs_indicies_dict = manager.dict()
            # self.section_ind_dict = manager.dict()

            # self.radstart_dict = manager.dict()
            # self.radend_dict = manager.dict()
            # self.phistart_dict = manager.dict()
            # self.phiend_dict = manager.dict()
            # self.input_img_num_dict = manager.dict()

            # self.klparam_dict = manager.dict()
            # self.wdhPAs_dict = manager.dict()

            self.klmodes_dict = {}
            self.evecs_dict = {}
            self.evals_dict = {}
            self.aligned_images_dict = {}
            self.ref_psfs_indicies_dict = {}
            self.section_ind_dict = {}

            self.radstart_dict = {}
            self.radend_dict = {}
            self.phistart_dict = {}
            self.phiend_dict = {}
            self.input_img_num_dict = {}

            self.klparam_dict = {}
            self.wdhPAs_dict = {}
        # Coords where align_and_scale places model center

        if self.load_from_basis is True:  # We want to load the FM basis
            # We load the FM basis files, before preparing the model to
            # be sure that the aligned_center is identical to the one used
            # when measuring the KL
            self.load_basis_files(psf_library=psf_library,
                                  kl_basis_file=kl_basis_file)

        else:  # We want to save the basis or just a single disk FM

            # Attributes of input
            self.inputs_shape = inputs_shape

            self.numbasis = numbasis

            # Outputs attributes
            output_imgs_shape = inputs_shape + self.numbasis.shape
            self.output_imgs_shape = output_imgs_shape

            self.PAs = dataset.PAs
            self.wdhPAs = dataset._wdhPAs
            self.wvs = dataset.wvs

            self.nwvs = int(np.size(np.unique(
                self.wvs)))  # Get the number of wvls
            self.nfiles = int(self.inputs_shape[0] /
                              self.nwvs)  # Get the number of files

            # default aligned_center if none (same default as fm.parallelized):
            if aligned_center is None:
                centers = dataset.centers
                aligned_center = [
                    np.mean(centers[:, 0]),
                    np.mean(centers[:, 1])
                ]

            # define the center
            self.aligned_center = aligned_center

            # Prepare the first disk for FM
            self.validPAs = model_pas_mask
            self.update_wind(model_wdh_list)

    def update_wind(self, model_wdh_list):
        """
        Rotate and sum N WDH models based on wind direction PAs for each frame.

        Args:
            model_wdh_list: List of 2D WDH model images (unrotated).
                            Length must match number of wind PA sets available.
            valid_PA_mask: Boolean mask of same shape as model_wdh_list; (Nmodels, Kimages)

        Returns:
            None. Sets self.model_wdhs with rotated+summed WDH models per frame.
        """
        num_components = len(model_wdh_list)
        model_shape = np.shape(model_wdh_list[0])
        n_frames = int(self.inputs_shape[0])
        self.model_wdhs = np.zeros(self.inputs_shape)

        # Historical basis files can contain wdhPAs/validPAs measured with a
        # different number of components than the current model list (e.g. 2 in
        # basis, 3 in current run). Normalize both arrays to shape
        # (num_components, n_frames) so indexing is always safe.
        wind_pa_list = np.asarray(self.wdhPAs)
        valid_pa_mask = np.asarray(self.validPAs)

        if wind_pa_list.ndim == 1:
            wind_pa_list = np.tile(wind_pa_list[None, :], (num_components, 1))
        if valid_pa_mask.ndim == 1:
            valid_pa_mask = np.tile(valid_pa_mask[None, :], (num_components, 1))

        if wind_pa_list.shape[1] != n_frames:
            raise ValueError(
                "wdhPAs frame dimension does not match dataset frame count: "
                f"{wind_pa_list.shape[1]} vs {n_frames}"
            )
        if valid_pa_mask.shape[1] != n_frames:
            raise ValueError(
                "validPAs frame dimension does not match dataset frame count: "
                f"{valid_pa_mask.shape[1]} vs {n_frames}"
            )

        if wind_pa_list.shape[0] < num_components:
            pad = np.tile(
                wind_pa_list[0:1, :],
                (num_components - wind_pa_list.shape[0], 1),
            )
            wind_pa_list = np.vstack([wind_pa_list, pad])
        if valid_pa_mask.shape[0] < num_components:
            pad = np.ones(
                (num_components - valid_pa_mask.shape[0], n_frames),
                dtype=bool,
            )
            valid_pa_mask = np.vstack([valid_pa_mask.astype(bool), pad])

        # Hoist the float64 cast out of the inner loop. There are only
        # `num_components` unique input arrays, but the inner loop runs
        # n_frames * num_components times, so the previous deepcopy + cast
        # in-loop produced ~n_frames extra copies per call. rotate_image
        # (cv2.warpAffine) does not mutate its input, so reusing the same
        # cast array across frames is safe.
        models = [m.astype(np.float64, copy=True) for m in model_wdh_list]

        for i in range(n_frames):  # inputs_shape [Kimages xpix ypix]
            model_sum = np.zeros(model_shape)

            for j in range(num_components):
                if not valid_pa_mask[j][i]:
                    # Adding zeros is a no-op; skip the rotation and the
                    # zeros allocation we used to do here.
                    continue

                wind_pa_here = wind_pa_list[j][i]
                model_rot = rotate_image(
                    models[j],
                    wind_pa_here,
                    self.aligned_center,
                    flipx=True,
                )
                model_sum += model_rot

            model_sum[np.isnan(model_sum)] = 0.0
            self.model_wdhs[i] = model_sum

        self.model_wdhs = np.reshape(
            self.model_wdhs,
            (n_frames, self.inputs_shape[1] * self.inputs_shape[2]),
        )


    def alloc_fmout(self, output_img_shape):
        """Allocates shared memory for the output of the shared memory


        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            [mp.array to store FM data in, shape of FM data array]

        """

        fmout_size = int(np.prod(output_img_shape))
        fmout_shape = output_img_shape
        fmout = mp.Array(self.data_type, fmout_size)
        return fmout, fmout_shape

    def fm_from_eigen(self,
                      klmodes=None,
                      evals=None,
                      evecs=None,
                      input_img_shape=None,
                      output_img_shape=None,
                      input_img_num=None,
                      ref_psfs_indicies=None,
                      section_ind=None,
                      aligned_imgs=None,
                      radstart=None,
                      radend=None,
                      phistart=None,
                      phiend=None,
                      padding=None,
                      IOWA=None,
                      ref_center=None,
                      parang=None,
                      numbasis=None,
                      fmout=None,
                      flipx=True,
                      mode=None,
                      **kwargs):
        """
        Generate forward models using the KL modes, eigenvectors, and eigenvectors from
        KLIP. Calls fm.py functions to perform the forward modelling. If we wish to save
        the KL modes, it save in dictionnaries.

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in
                    ascending order(lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_image_shape: 2-D shape of inpt images ([ysize, xsize])
            input_img_num: index of sciece frame
            ref_psfs_indicies: array of indicies for each reference PSF
            section_ind: array indicies into the 2-D x-y image that correspond to
                            this section. Note: needs be called as section_ind[0]
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            IOWA: tuple (IWA,OWA) IWA = Inner working angle & OWA = Outer working angle,
                    both in pixels. It defines the separation interva in which klip will
                    be run.
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            numbasis: array of KL basis cutoffs
            fmout: numpy output array for FM output. Shape is (N, y, x, b)
            mode: mode of the reduction ('RDI', 'ADI', 'SDI'). If RDI only, we only 
                    measure the oversubctraction
            kwargs: any other variables that we don't use but are part of the input

        Returns:
            None

        """


        if self.load_from_basis == False:
            sci = aligned_imgs[input_img_num, section_ind[0]]
            refs = aligned_imgs[ref_psfs_indicies, :]
            refs = refs[:, section_ind[0]]
        else:
            wlstrkey = 'wl' + str(int(self.wvs[input_img_num] * 1000)).zfill(4)
            sci = self.aligned_images_dict[wlstrkey][input_img_num,
                                                     section_ind[0]]
            # in the case of load_from_basis, the images are already
            # saved in the DiskFM object, we can save a few tens of
            # Mbytes (per cpu) by not saving them
            # and just passing them to the nex function

        # use the disk model stored
        model_sci = self.model_wdhs[input_img_num, section_ind[0]]
        # model_sci[np.where(np.isnan(model_sci))] = 0
        # model_sci[model_sci != model_sci] = 0
        model_ref = self.model_wdhs[ref_psfs_indicies, :]
        model_ref = model_ref[:, section_ind[0]]
        # model_ref[np.where(np.isnan(model_ref))] = 0
        # model_ref[model_ref != model_ref] = 0
        if mode == 'RDI':
            #if only RDI we skip the deltaKL calculation since we do only over-subctraction
            delta_KL = klmodes * 0.
        else:
            # using original Kl modes and reference models, compute the perturbed KL modes
            # (spectra is already in models)
            if self.load_from_basis == False:
                delta_KL = fm.perturb_specIncluded(
                    evals,
                    evecs,
                    klmodes,
                    refs,
                    model_ref,
                    # return_perturb_covar=False,
                )
            else:
                # in the case of load_from_basis, the images are already saved in the
                # DiskFM object, we can save a few tens of Mbytes (per cpu) by not 
                # saving them and just passing them to the nex function
                delta_KL = fm.perturb_specIncluded(
                    evals,
                    evecs,
                    klmodes,
                    self.aligned_images_dict[wlstrkey][ref_psfs_indicies, :]
                    [:, section_ind[0]],
                    model_ref,
                    # return_perturb_covar=False,
                )

        # calculate postklip_psf using delta_KL
        postklip_psf, _, _ = fm.calculate_fm(delta_KL,
                                             klmodes,
                                             numbasis,
                                             sci,
                                             model_sci,
                                             inputflux=None)

        # write forward modelled disk to fmout (as output)
        # need to derotate the image in this step

        for thisnumbasisindex in range(np.size(numbasis)):
            fm._save_rotated_section(input_img_shape,
                                     postklip_psf[thisnumbasisindex],
                                     section_ind,
                                     fmout[input_img_num, :, :,
                                           thisnumbasisindex],
                                     None,
                                     parang,
                                     radstart,
                                     radend,
                                     phistart,
                                     phiend,
                                     padding,
                                     IOWA,
                                     ref_center,
                                     flipx=flipx)

        # We save the KL basis and params for this image and section in a dictionnaries
        if self.save_basis is True:
            # save the parameter used in KLIP-FM. We save a float64 to avoid pbs
            # in the saving and loading

            if mode == 'RDI':
                self.klparam_dict['isRDI'] = np.float64(1.)
            else:
                self.klparam_dict['isRDI'] = np.float64(0.)

            [IWA, OWA] = IOWA
            self.klparam_dict['IWA'] = np.float64(IWA)
            self.klparam_dict['OWA'] = np.float64(OWA)

            self.klparam_dict['input_img_shape'] = np.float64(input_img_shape)
            self.klparam_dict['numbasis'] = np.float64(numbasis)
            self.klparam_dict['output_imgs_shape'] = np.float64(
                output_img_shape)
            
            # Save the WDH PAs and PA mask
            self.wdhPAs_dict["PAs"] = np.float64(self.wdhPAs)
            self.wdhPAs_dict["PAmask"] = np.float64(self.validPAs)

            # To have a single identifier for each set of aligned images,
            # we save the wavelenght in nm
            wlstrkey = 'wl' + str(int(self.wvs[input_img_num] * 1000)).zfill(4)
            self.aligned_images_dict[wlstrkey] = aligned_imgs

            # save the center for aligning the image in KLIP-FM. In practice, this
            # center will be used for all the models after we load.
            self.klparam_dict['aligned_center_x'] = np.float64(ref_center[0])
            self.klparam_dict['aligned_center_y'] = np.float64(ref_center[1])

            # We save information about the dataset that will be used when we load the KL basis
            self.klparam_dict['PAs'] = np.float64(self.PAs)
            self.klparam_dict['wdhPAs'] = np.float64(self.wdhPAs)
            self.klparam_dict['wvs'] = np.float64(self.wvs)

            self.klparam_dict['nwvs'] = np.float64(self.nwvs)
            self.klparam_dict['nfiles'] = np.float64(self.nfiles)

            # To have a single identifier for each set of section/image for the
            # dictionnaries key, we use section first pixel and image number
            curr_im = str(input_img_num).zfill(3)
            namkey = 'idsec' + str(section_ind[0][0]) + 'i' + curr_im
            # saving the KL modes dictionnaries
            self.klmodes_dict[namkey] = klmodes
            self.evals_dict[namkey] = evals
            self.evecs_dict[namkey] = evecs
            self.ref_psfs_indicies_dict[namkey] = ref_psfs_indicies
            self.section_ind_dict[namkey] = section_ind

            # saving the section delimiters dictionnaries
            self.radstart_dict[namkey] = radstart
            self.radend_dict[namkey] = radend
            self.phistart_dict[namkey] = phistart
            self.phiend_dict[namkey] = phiend
            self.input_img_num_dict[namkey] = input_img_num

    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is
        the first one and not the last. We also use this function to save the KL basis
        because it is called by fm.py at the end fm.klip_parallelized

        Args:
            fmout: numpy array of ouput of FM

        Returns:
            Same but cleaned up if necessary
        """

        # save the KL basis.
        if self.save_basis:
            self.save_kl_basis()

        # FIXME We save the matrix here it here because it is called by fm.py at the end
        # fm.klip_parallelized but this is not ideal.

        dims = fmout.shape
        fmout = np.rollaxis(
            fmout.reshape((dims[0], dims[1], dims[2], dims[3])), 3)
        return fmout

    def save_fmout(self,
                   dataset,
                   fmout,
                   outputdir,
                   fileprefix,
                   numbasis,
                   klipparams=None,
                   calibrate_flux=False,
                   pixel_weights=1,
                   **kwargs):
        """
        Uses dataset parameters to save the forward model, the output of
        fm_paralellized or klip_dataset. No returm, data are saved
        in "fileprefix" .fits files

        Args:
            dataset: an instance of Instrument.Data . Will use its
                     dataset.savedata() function to save data
            fmout: output of forward modelling.
            outputdir: directory to save output files
            fileprefix: filename prefix for saved files
            numbasis: number of KL basis vectors to use
                      (can be a scalar or list like)
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data in the same way as
                            the klipped data
            pixel_weights: weights for each pixel for weighted mean. Leave this as a
                           single number for simple mean

        Returns:
            None

        """

        weighted = len(np.shape(pixel_weights)) > 1
        numwvs = dataset.numwvs
        fmout_spec = fmout.reshape([
            fmout.shape[0],
            fmout.shape[1] // numwvs,
            numwvs,
            fmout.shape[2],
            fmout.shape[3],
        ])  # (b, N_cube, wvs, y, x) 5-D cube

        # collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(pixel_weights * fmout_spec, axis=(1, 2))
        if weighted:
            # if the pixel weights aren't just 1 (i.e., weighted case),
            # we need to normalize for that
            KLmode_cube /= np.nanmean(pixel_weights, axis=(1, 2))

        # broadband flux calibration for KL mode cube
        if calibrate_flux:
            KLmode_cube = dataset.calibrate_output(KLmode_cube, spectral=False)

        dataset.savedata(
            path.join(outputdir, fileprefix + "-fmpsf-KLmodes-all.fits"),
            KLmode_cube,
            klipparams=klipparams.format(numbasis=str(numbasis)),
            filetype="KL Mode Cube",
            zaxis=numbasis,
        )

        # if there is more than one wavelength, save also spectral cubes
        if dataset.numwvs > 1:

            KLmode_spectral_cubes = np.nanmean(pixel_weights * fmout_spec,
                                               axis=1)
            if weighted:
                # if the pixel weights aren't just 1 (i.e., weighted case), we need to
                # normalize for that.
                KLmode_spectral_cubes /= np.nanmean(pixel_weights, axis=1)

            for KLcutoff, spectral_cube in zip(numbasis,
                                               KLmode_spectral_cubes):
                # calibrate spectral cube if needed
                if calibrate_flux:
                    spectral_cube = dataset.calibrate_output(spectral_cube,
                                                             spectral=True)
                dataset.savedata(
                    path.join(
                        outputdir, fileprefix +
                        "-fmpsf-KL{0}-speccube.fits".format(KLcutoff)),
                    spectral_cube,
                    klipparams=klipparams.format(numbasis=KLcutoff),
                    filetype="PSF Subtracted Spectral Cube",
                )

    def save_kl_basis(self):
        """
        Save the KL basis and other needed parameters

        Args:
            None

        Returns:
            None

        """

        # Convert everything to np arrays and types to be safe for the saving.
        for key in self.section_ind_dict.keys():
            self.section_ind_dict[key] = np.asarray(self.section_ind_dict[key])
            self.radstart_dict[key] = np.float64(self.radstart_dict[key])
            self.radend_dict[key] = np.float64(self.radend_dict[key])
            self.phistart_dict[key] = np.float64(self.phistart_dict[key])
            self.phiend_dict[key] = np.float64(self.phiend_dict[key])

        _, file_extension = path.splitext(self.basis_filename)


        if file_extension == ".h5":
            # transform mp dicts to normal dicts
            # make a single dictionnary and save in h5

            saving_in_h5_dict = {
                'aligned_images_dict': dict(self.aligned_images_dict),
                'klmodes_dict': dict(self.klmodes_dict),
                'evecs_dict': dict(self.evecs_dict),
                'evals_dict': dict(self.evals_dict),
                'ref_psfs_indicies_dict': dict(self.ref_psfs_indicies_dict),
                'section_ind_dict': dict(self.section_ind_dict),
                'radstart_dict': dict(self.radstart_dict),
                'radend_dict': dict(self.radend_dict),
                'phistart_dict': dict(self.phistart_dict),
                'phiend_dict': dict(self.phiend_dict),
                'input_img_num_dict': dict(self.input_img_num_dict),
                'klparam_dict': dict(self.klparam_dict),
                'wdhPAs_dict': dict(self.wdhPAs_dict),
            }

            _save_dict_to_hdf5(saving_in_h5_dict, self.basis_filename)

            del saving_in_h5_dict

        else:
            raise ValueError(file_extension +
                             """ is not a possible extension. Filenames can
                haves 2 recognizable extension2: .h5 and .pkl""")

    def load_basis_files(self, psf_library=None, kl_basis_file=None):
        """
        Loads in previously saved basis files and sets variables for fm_from_eigen

        Args:
            dataset: an instance of Instrument.Data, after fm.klip_dataset.
                     Allow me to pass in the structure some correction parameters
                     set by fm.klip_dataset, such as IWA, OWA, aligned_center.
                     KL basis and sections information are passed via global variables

        Returns:
            None
        """
        if kl_basis_file is not None:
            file_extension = ""
        else:
            _, file_extension = path.splitext(self.basis_filename)

        # Load in file

        if not file_extension == ".h5":
            print("Only .h5 file format supported for the basis file.")
        
        else:
            kl_basis_file = _load_dict_from_hdf5(self.basis_filename)


        self.aligned_images_dict = dict(
            kl_basis_file['aligned_images_dict'])

        self.klmodes_dict = dict(kl_basis_file['klmodes_dict'])
        self.evecs_dict = dict(kl_basis_file['evecs_dict'])
        self.evals_dict = dict(kl_basis_file['evals_dict'])
        self.ref_psfs_indicies_dict = dict(
            kl_basis_file['ref_psfs_indicies_dict'])
        self.section_ind_dict = dict(kl_basis_file['section_ind_dict'])

        self.radstart_dict = dict(kl_basis_file['radstart_dict'])
        self.radend_dict = dict(kl_basis_file['radend_dict'])
        self.phistart_dict = dict(kl_basis_file['phistart_dict'])
        self.phiend_dict = dict(kl_basis_file['phiend_dict'])
        self.input_img_num_dict = dict(
            kl_basis_file['input_img_num_dict'])
        self.wdhPAs_dict = dict(kl_basis_file["wdhPAs_dict"])
        self.validPAs = self.wdhPAs_dict["PAmask"]
        self.klparam_dict = dict(kl_basis_file['klparam_dict'])

        del kl_basis_file

        # read key name for each section and image
        self.dict_keys = sorted(self.klmodes_dict.keys())

        # load parameters of the correction that fm.klip_dataset produced
        # when we saved the FM basis.

        self.isRDI = (self.klparam_dict['isRDI'] == 1)
        self.IWA = self.klparam_dict['IWA']
        self.OWA = self.klparam_dict['OWA']

        numbasis = self.klparam_dict['numbasis'].astype(int)
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

        self.numbasis = numbasis

        self.aligned_center = [
            self.klparam_dict['aligned_center_x'],
            self.klparam_dict['aligned_center_y'],
        ]

        output_imgs_shape = tuple(
            self.klparam_dict['output_imgs_shape'].astype(int))

        self.output_imgs_shape = output_imgs_shape

        # Those are loaded to avoid depending at all on the dataset when we load the KL basis
        self.PAs = self.klparam_dict['PAs']
        self.wdhPAs = self.wdhPAs_dict["PAs"]
        self.wvs = self.klparam_dict['wvs']

        self.nwvs = int(self.klparam_dict['nwvs'])  # Get the number of wvls
        self.nfiles = int(
            self.klparam_dict['nfiles'])  # Get the number of wvls

        dim_frame = self.klparam_dict['input_img_shape']
        self.inputs_shape = np.array(
            (self.nfiles * self.nwvs, int(dim_frame[0]), int(dim_frame[1])))

        # After loading it, we stop saving the KL basis to avoid saving it every time
        # we run self.fm_parallelize.
        self.save_basis = False

    def fm_parallelized(self):
        """
        Functions like fm.klip_dataset, but it uses previously measured KL modes,
        section positions, and klip parameter to return the forward modelling.
        Do not save fits.

        Args:
            None

        Returns:
            fmout_np, a numpy array, output of forward modelling
                    * if N_wl = 1, size is [n_KL,x,y]
                    * if N_wl > 1, size is  [n_KL,N_wl,x,y]

        """

        # The per-section loop below is sequential: there is no inner
        # multiprocessing.Pool reading/writing fmout, so we do not need an
        # mp.Array (which incurs a synchronized shared-memory allocation
        # plus a resource_tracker round-trip per call). A plain numpy array
        # is materially faster in the MCMC hot path. alloc_fmout is left in
        # place for the fm.klip_dataset basis-construction path, which does
        # spawn workers that share fmout.
        fmout_np = np.zeros(self.output_imgs_shape, dtype=self.data_type)

        # this line is added to be able to use fm._save_rotated_section
        # which uses global var outputs_shape
        fm.outputs_shape = self.output_imgs_shape

        wvs = self.wvs

        if self.isRDI:
            mode = 'RDI'
        else:
            mode = None
            # We are only interested in the RDI mode
            # if not we don't care since it does not have an
            # impact at this point

        for key in self.dict_keys:  # loop pver the sections/images

            img_num = self.input_img_num_dict[key]

            # in load mode, we do not pass aligned_images_dict
            # because it is already in the class to
            # save memory
            self.fm_from_eigen(
                klmodes=self.klmodes_dict[key],
                evals=self.evals_dict[key],
                evecs=self.evecs_dict[key],
                input_img_shape=[self.inputs_shape[1], self.inputs_shape[2]],
                output_img_shape=self.output_imgs_shape,
                input_img_num=img_num,
                ref_psfs_indicies=self.ref_psfs_indicies_dict[key],
                section_ind=self.section_ind_dict[key],
                radstart=self.radstart_dict[key],
                radend=self.radend_dict[key],
                phistart=self.phistart_dict[key],
                phiend=self.phiend_dict[key],
                padding=0.0,
                IOWA=(self.IWA, self.OWA),
                ref_center=self.aligned_center,
                parang=self.PAs[img_num],
                numbasis=self.numbasis,
                fmout=fmout_np,
                flipx=True,
                mode=mode)

        # put any finishing touches on the FM Output
        fmout_np = self.cleanup_fmout(fmout_np)

        # If false then this is a collapsed-spec mode or pol mode: collapsed
        # across all files
        fmout_return = np.mean(fmout_np, axis=1)

        return fmout_return
    
    def fm_parallelized_jit(self):
        """
        Functions like fm.klip_dataset, but it uses previously measured KL modes,
        section positions, and klip parameter to return the forward modelling.
        Do not save fits.

        Args:
            None

        Returns:
            fmout_np, a numpy array, output of forward modelling
                    * if N_wl = 1, size is [n_KL,x,y]
                    * if N_wl > 1, size is  [n_KL,N_wl,x,y]

        """

        # See fm_parallelized: this loop is sequential, so we skip the
        # mp.Array shared-memory allocation in the MCMC hot path. alloc_fmout
        # remains for the fm.klip_dataset basis-construction path.
        fmout_np = np.zeros(self.output_imgs_shape, dtype=self.data_type)

        # this line is added to be able to use fm._save_rotated_section
        # which uses global var outputs_shape
        fm.outputs_shape = self.output_imgs_shape

        wvs = self.wvs

        if self.isRDI:
            mode = 'RDI'
        else:
            mode = None
            # We are only interested in the RDI mode
            # if not we don't care since it does not have an
            # impact at this point

        # Hoist branch-decision out of the per-section loop. All four
        # conditions are invariant across `self.dict_keys` for a single
        # MCMC step:
        #   - load_from_basis / isRDI / numbasis size are class-level config
        #   - model_wdhs is sanitized inside update_wind (NaNs zeroed) before
        #     fm_parallelized_jit is called from logl, so the np.isnan scan
        #     was always False here AND was an O(N) full-model traversal per
        #     section. We drop it entirely; if a future code path can re-
        #     introduce NaNs in model_wdhs, sanitize at the source instead.
        use_jit_path = (
            self.load_from_basis
            and not self.isRDI
            and np.size(self.numbasis) == 1
        )

        for key in self.dict_keys:  # loop pver the sections/images

            img_num = self.input_img_num_dict[key]

            # in load mode, we do not pass aligned_images_dict
            # because it is already in the class to
            # save memory
            if not use_jit_path:
                self.fm_from_eigen(
                    klmodes=self.klmodes_dict[key],
                    evals=self.evals_dict[key],
                    evecs=self.evecs_dict[key],
                    input_img_shape=[self.inputs_shape[1], self.inputs_shape[2]],
                    output_img_shape=self.output_imgs_shape,
                    input_img_num=img_num,
                    ref_psfs_indicies=self.ref_psfs_indicies_dict[key],
                    section_ind=self.section_ind_dict[key],
                    radstart=self.radstart_dict[key],
                    radend=self.radend_dict[key],
                    phistart=self.phistart_dict[key],
                    phiend=self.phiend_dict[key],
                    padding=0.0,
                    IOWA=(self.IWA, self.OWA),
                    ref_center=self.aligned_center,
                    parang=self.PAs[img_num],
                    numbasis=self.numbasis,
                    fmout=fmout_np,
                    mode=mode)
            # JIT fast path: load_from_basis, no RDI, single numbasis, no NaN
            # in the rotated WDH model. RDI is intentionally excluded — it skips
            # the perturb step entirely (delta_KL = 0) and would not benefit.
            else:
                wlstrkey = 'wl' + str(int(self.wvs[img_num] * 1000)).zfill(4)
                this_section = self.section_ind_dict[key][0]
                these_ref_indices = self.ref_psfs_indicies_dict[key]
                this_image = self.aligned_images_dict[wlstrkey][img_num,
                                                     this_section]
                these_refs = self.aligned_images_dict[wlstrkey][these_ref_indices, :]
                these_refs = these_refs[:, this_section]
                pk_psf = fm_from_eigen_jit(
                    self.klmodes_dict[key],
                    self.evals_dict[key],
                    self.evecs_dict[key],
                    img_num,
                    self.ref_psfs_indicies_dict[key],
                    self.section_ind_dict[key],
                    this_image,
                    these_refs,
                    self.model_wdhs,
                )
                # pk_psf is a 1D (N_pix,) array, written into the single-basis
                # slice of fmout_np via _save_rotated_section's standard plumbing.
                fm._save_rotated_section(
                    [self.inputs_shape[1], self.inputs_shape[2]],
                    pk_psf,
                    self.section_ind_dict[key],
                    fmout_np[img_num, :, :, 0],
                    None,
                    self.PAs[img_num],
                    self.radstart_dict[key],
                    self.radend_dict[key],
                    self.phistart_dict[key],
                    self.phiend_dict[key],
                    0.0,
                    (self.IWA, self.OWA),
                    self.aligned_center,
                    flipx=True,
                )
        # put any finishing touches on the FM Output
        fmout_np = self.cleanup_fmout(fmout_np)

        fmout_return = np.nanmean(fmout_np, axis=1)

        return fmout_return


##############################################################################
###### 4 routines to save and load h5 in dictionnaries
##############################################################################


def _save_dict_to_hdf5(dic, filename):
    """
    Saving a nested dictionnary into a h5 file

    Args:
        dic: the dictionnary to file
        filename: the filename of the h5 where it will be saved

    Returns:
        None

    """
    with h5py.File(filename, "w") as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)


def _load_dict_from_hdf5(filename):
    """
    Load a dictionnary from a h5 file

    Args:
        filename: the filename of the h5

    Returns:
        the dictionnary exctracted

    """

    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively explore the dictionnary for saving it

    Args:
        h5file: the file in which we save, opened with h5py.File
        path: the separator to aggregate the keys. Should not be set to a value that
            is likely to be in the dictionnary keys already
        dic: the dictionnary to deconstruct

    Returns
        None

    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/',
                                                     item)
        else:
            raise ValueError("Cannot save {0} type in h5 (key = {1})".format(
                type(item), path + key))


def _recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively explore the dictionnary for loading it

    Args:
        h5file: the file from which we load, opened with h5py.File
        path: the separator to aggregate the keys. Should be the same one used in
        _recursively_save_dict_contents_to_group
    Returns
        the rebuilt dictionnary
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans
