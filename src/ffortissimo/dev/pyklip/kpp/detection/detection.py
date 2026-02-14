__author__ = 'JB'

from scipy.signal import convolve2d
from pyklip.kpp.stat.stat_utils import *
from pyklip.kpp.utils.oi import *


def point_source_detection(image, center,threshold,pix2as=None,mask_radius = 4,maskout_edge=None,maskout_inner_edge=None,IWA=None, OWA=None):
        """
        Find the brightest blobs in the image/cube.

        Args:
            image: Image from which to get the SNR map
            center: center of the image (y_cen, x_cen)
            threshold: Threshold under which blob should be ignore.
            pix2as: Platescale (arcsec per pixel).
            mask_radius: Radius of the mask used for masking point sources or the surroundings of the current pixel out
                        of the data. Default value is 4 pixels.
            maskout_edge: mask a maskout_edge (int) pixels around the outer edge of the FOV containing NaNs.
            maskout_inner_edge: mask a maskout_inner_edge (int) pixels border around the center nan disk.
            IWA: inner working angle in pixels.
            OWA: outer working angle in pixels.

        :return: Detection table..

            Table containing the list of the local maxima with their info
            Description by column: ["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]
            1/ index of the candidate
            2/ Value of the maximum
            3/ Position angle in degree from North in [0,360]
            4/ Separation in pixel
            5/ Separation in arcsec
            6/ x position in pixel
            7/ y position in pixel
            8/ row index
            9/ column index
        """

        if image is not None:
            image = image
            if np.size(image.shape) == 2:
                ny,nx = image.shape
        if center is not None:
            center = [center]

        # Make a copy of the criterion map because it will be modified in the following.
        # Local maxima are indeed masked out when checked
        image_cpy = copy(image)
        stamp_size = mask_radius * 2 + 2
        image_cpy = np.pad(image_cpy,((stamp_size//2,stamp_size//2),(stamp_size//2,stamp_size//2)),mode="constant",constant_values=np.nan)

        # Build as grids of x,y coordinates.
        # The center is in the middle of the array and the unit is the pixel.
        # If the size of the array is even 2n x 2n the center coordinate in the array is [n,n].
        x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0][0],np.arange(0,ny,1)-center[0][1])


        # Definition of the different masks used in the following.
        # Mask to remove the spots already checked in criterion_map.
        stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_size,1)-stamp_size//2,np.arange(0,stamp_size,1)-stamp_size//2)
        stamp_mask = np.ones((stamp_size,stamp_size))
        r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
        stamp_mask[np.where(r_stamp < mask_radius)] = np.nan

        # Mask out a band of maskout_edge pixels around the edges of the finite pixels of the image.
        if maskout_edge is not None:
            IWA,OWA,inner_mask,outer_mask = get_occ(image_cpy, centroid = (center[0][0]+stamp_size//2,center[0][1]+stamp_size//2))
            conv_kernel = np.ones((maskout_edge,maskout_edge))
            flat_cube_wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
            image_cpy[np.where(np.isnan(flat_cube_wider_mask))] = np.nan
        if maskout_inner_edge is not None:
            IWA,OWA,inner_mask,outer_mask = get_occ(image_cpy, centroid = (center[0][0]+stamp_size//2,center[0][1]+stamp_size//2))
            conv_kernel = np.ones((maskout_inner_edge,maskout_inner_edge))
            flat_cube_wider_mask = convolve2d(inner_mask,conv_kernel,mode="same")
            image_cpy[np.where(np.isnan(flat_cube_wider_mask))] = np.nan


        # Number of rows and columns to add around a given pixel in order to extract a stamp.
        row_m = int(np.floor(stamp_size/2.0))    # row_minus
        row_p = int(np.ceil(stamp_size/2.0))     # row_plus
        col_m = int(np.floor(stamp_size/2.0))    # col_minus
        col_p = int(np.ceil(stamp_size/2.0))     # col_plus

        # Table containing the list of the local maxima with their info
        # Description by column:
        # 1/ index of the candidate
        # 2/ Value of the maximum
        # 3/ Position angle in degree from North in [0,360]
        # 4/ Separation in pixel
        # 5/ Separation in arcsec
        # 6/ x position in pixel
        # 7/ y position in pixel
        # 8/ row index
        # 9/ column index
        candidates_table = []
        table_labels = ["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]
        ## START WHILE LOOP.
        # Each iteration looks at one local maximum in the criterion map.
        k = 0
        max_val_criter = np.nanmax(image_cpy)
        while max_val_criter > threshold:# and k <= max_attempts:
            k += 1
            # Find the maximum value in the current criterion map. At each iteration the previous maximum is masked out.
            max_val_criter = np.nanmax(image_cpy)
            # Locate the maximum by retrieving its coordinates
            max_ind = np.where( image_cpy == max_val_criter )
            row_id,col_id = max_ind[0][0],max_ind[1][0]
            x_max_pos = x_grid[row_id-stamp_size//2,col_id-stamp_size//2]
            y_max_pos = y_grid[row_id-stamp_size//2,col_id-stamp_size//2]
            sep_pix = np.sqrt(x_max_pos**2+y_max_pos**2)
            if pix2as is not None:
                sep_arcsec = pix2as *sep_pix
            else:
                sep_arcsec = 0
            pa = np.mod(np.rad2deg(np.arctan2(-x_max_pos,y_max_pos)),360)

            # Mask the spot around the maximum we just found.
            image_cpy[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

            if IWA is not None:
                if sep_pix < IWA:
                    continue
            if OWA is not None:
                if sep_pix > OWA:
                    continue

            # Store the current local maximum information in the table
            candidates_table.append([k,max_val_criter,pa,sep_pix,sep_arcsec,x_max_pos,y_max_pos,row_id-stamp_size//2,col_id-stamp_size//2])
        ## END WHILE LOOP.

        return candidates_table