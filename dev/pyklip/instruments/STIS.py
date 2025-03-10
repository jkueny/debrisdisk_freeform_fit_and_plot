
import os
import glob
import astropy.io.fits as fits
import astropy
import scipy
import numpy as np
from astropy import wcs
import math
from astropy.nddata import Cutout2D
#import matplotlib.pyplot as plt # This can be done here only if you're not using parallelized, which pyklip does use
import timeit
import pyklip
import pyklip.klip as klip
import pyklip.parallelized as parallelized
import pyklip.instruments.utils.radonCenter as radonCenter
import pyklip.instruments.utils.wcsgen as wcsgen
from pyklip.instruments.utils.prntfFunction import prntf

import sys
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar
else:
    import configparser as ConfigParser 
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar

from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import product
import time
import warnings

import psutil
cpuCount = psutil.cpu_count(logical = False)

CCDPHYSICALWIDTH  = 1024
CCDPHYSICALHEIGHT = 1024
PLATESCALE        = 0.05
A10Y              = 215.238
AWEDGESLOPE       = 40.0 # nominal; has to be tested
AWEDGESLOPE       = 1.0 # nominal; has to be tested
BWEDGESLOPE       = 0.025 # (1/40)
BWEDGESLOPE       = 0.025568181818182
BWEDGESLOPE       = 0.032352941176471
EXPTIME_RATIO     = 1.3 # PROD. "o6bx32010" FITS file's EXPTIME of 756 divided by EXPTIME of 630 = 1.2, and, 1.2 would be within the threshold of 1.3...
#EXPTIME_RATIO = 1.1 # TEST/DIAG. This must be high enough to let through *some frames, or else your outputs will be null
#EXPTIME_RATIO = 1.199 # TEST/DIAG. Keep this DIAGNOSTIC value which is less than 755/630 BUT NOT LESS THAN 756/630 (which = 1.2). (Use PRODUCTION value of 1.3)






def get_file_mask ( maskFilename, VL ) :
    """
        Gets the shape of the mask in the incoming file, according to the FITS file, no processing applied.
        Args             :
            maskFilename : String. The FITS file name that contains the mask
            udh          : Int.    User Determined Height
            udw          : Int.    User Determined Weight
        Returns          : Int, Int. Height and width of the mask
        get_file_mask() is called from the polling section of primary_function()
        This is diagnostic for now since there will be a variety of shapes and sizes of masks.
        Eventually this is folded into the get_mask_function()... 
    """
    filename2       = os.path.basename ( maskFilename )
    directoryPath2  = os.path.dirname  ( maskFilename )
    split2          = os.path.splitext ( os.path.basename ( maskFilename ) )
    maskRootname    = split2[0]
    maskInput_hdu_0 = fits.open        ( maskFilename )
    file_Mask       = maskInput_hdu_0[0].data

    return file_Mask.shape[0], file_Mask.shape[1] # height is 0, width is 1






def get_mask_function ( hch, wcw, maskFilename, VL, dynMask4ple, runWedgeDecided ) :
    """
        Counts the effective aperture rows of the incoming mask, which may have dead rows above and below.
        Makes an array of only the effective aperture rows, which form the actual usable mask.
        Disregards dead rows above and below.
        Args             :
            hch          : Int.    Highest Common Height
            wcw          : Int.    Widest Common Width
            maskFilename : String. The FITS file name that contains the mask
            dynMask4ple  : ( dynamicMaskOverrideFileMask , maskWidthAtPSFPoint , userAddWidthToWedge , userAddWidthToSpiderLegs )
        Return           : numpy array, of the mask
        get_mask_function() is called from the polling section of primary_function()
    """
    if VL >= 3 : prntf("stis",20,"GMF dynMask4ple                       : ", dynMask4ple)
    if dynMask4ple [ 0 ] :

        largestWedgeThickInArcseconds = dynMask4ple [ 1 ]
        largestWedgeThickInPixels     = largestWedgeThickInArcseconds / PLATESCALE
        halfLargestWedgeThickInPixels = int ( largestWedgeThickInPixels / 2 )
        halfLargestWedgeThickAfterMod = halfLargestWedgeThickInPixels + dynMask4ple [ 2 ]
        verticalCenter                    = int ( hch / 2 )
        horizontalCenter                  = int ( wcw / 2 )
        spiderLegWidth                    = 20
        spiderLegStartX                   = horizontalCenter -  int ( spiderLegWidth / 2 )
        spiderLegStartY                   = verticalCenter

        dynamicMask = np.ones ( ( hch, wcw ) )
        dynamicMask [ spiderLegStartY ] [ spiderLegStartX ] = 0

        # Spider Legs Both Wedges
        spiderLegX = spiderLegStartX
        spiderLegY = spiderLegStartY
        if runWedgeDecided == 'A' :
            verticalLimitOfSpiderLegs = math.ceil ( hch / 2 )
        if runWedgeDecided == 'B' :
            verticalLimitOfSpiderLegs = math.ceil ( min ( hch, wcw ) / 2 )
            prntf("stis",20,"GMF verticalLimitOfSpiderLegs         : ", verticalLimitOfSpiderLegs)
        for row in range ( verticalLimitOfSpiderLegs ) :
            for col in range ( spiderLegWidth + dynMask4ple [ 3 ] ) :
                dynamicMask [   spiderLegY     ] [   spiderLegX + col ] = 0
                dynamicMask [ - spiderLegY - 1 ] [ - spiderLegX - col ] = 0
                dynamicMask [   spiderLegY     ] [ - spiderLegX - col ] = 0
                dynamicMask [ - spiderLegY - 1 ] [   spiderLegX + col ] = 0        
            spiderLegY = spiderLegY + 1
            spiderLegX = spiderLegX + 1

        AWEDGESLOPE       = .05 # nominal; has to be tested

        # Wedge A
        if runWedgeDecided == 'A' :
            spiderLegX = horizontalCenter
            spiderLegY = spiderLegStartY
            for row in range ( math.ceil ( hch / 2 ) ) :
                wedgeHalfWidthTop = halfLargestWedgeThickInPixels + ( ( spiderLegY + row ) * BWEDGESLOPE ) + dynMask4ple [ 2 ]

                for col in range ( int ( wedgeHalfWidthTop ) ) :

                    dynamicMask [ spiderLegY + row ] [ spiderLegX + col ] = 0
                    dynamicMask [ spiderLegY + row ] [ spiderLegX - col ] = 0
                wedgeHalfWidthBottom = halfLargestWedgeThickInPixels + ( ( spiderLegY - row ) * BWEDGESLOPE ) + dynMask4ple [ 2 ]

                for col in range ( int ( wedgeHalfWidthBottom ) ) :
                  
                    dynamicMask [ spiderLegY - row ] [ spiderLegX + col ] = 0
                    dynamicMask [ spiderLegY - row ] [ spiderLegX - col ] = 0
  

        # Wedge B
        elif runWedgeDecided == 'B' :
            spiderLegX = horizontalCenter
            spiderLegY = spiderLegStartY
            for col in range ( math.ceil ( wcw / 2 ) ) :
                wedgeHalfHeightLeft = halfLargestWedgeThickInPixels + ( ( spiderLegX - col ) * BWEDGESLOPE ) + dynMask4ple [ 2 ]
                for row in range ( int ( wedgeHalfHeightLeft ) ) : 
                    dynamicMask [ spiderLegY + row ] [ spiderLegX - col ] = 0
                    dynamicMask [ spiderLegY - row ] [ spiderLegX - col ] = 0
                wedgeHalfHeightRight = halfLargestWedgeThickInPixels + ( ( spiderLegX + col ) * BWEDGESLOPE ) + dynMask4ple [ 2 ]
                for row in range ( int ( wedgeHalfHeightRight ) ) : 
                    dynamicMask [ spiderLegY + row ] [ spiderLegX + col ] = 0
                    dynamicMask [ spiderLegY - row ] [ spiderLegX + col ] = 0
    
    if VL >= 3 : prntf("stis",20,"GMF hch                             : ", hch)
    if VL >= 3 : prntf("stis",20,"GMF wcw                             : ", wcw)
    filename2       = os.path.basename ( maskFilename )
    directoryPath2  = os.path.dirname  ( maskFilename )
    split2          = os.path.splitext ( os.path.basename ( maskFilename ) )
    maskRootname    = split2[0]
    maskInput_hdu_0 = fits.open        ( maskFilename )
    file_Mask    = maskInput_hdu_0[0].data

    if VL >= 3 : prntf("stis",20,"GMF file_Mask.shape                  : ", file_Mask.shape)
    file_MaskEffectiveApertureHeight = 0
    
    # Search every row in the mask
    for jjjRowIndex in range ( len ( maskInput_hdu_0[0].data ) ) :
        
        # Seach every column pixel in that row
        for iiiColIndex in range ( len ( maskInput_hdu_0[0].data [ jjjRowIndex ] ) ) :
            
            # If ANY column pixel in that row is a 1
            if maskInput_hdu_0[0].data[jjjRowIndex][iiiColIndex] == 1 :
                
                # THEN we know that this is a VALID aperture row, and must be counted into a final row height 
                file_MaskEffectiveApertureHeight += 1
                break
    
    if VL >= 3 : prntf("stis",20,"GMF file_MaskEffectiveApertureHeight : ", file_MaskEffectiveApertureHeight)
    
    if runWedgeDecided == "A" :
        if maskInput_hdu_0[0].data.shape[0] > hch:
            if VL >= 3 : prntf("stis",20,"GMF if maskInput_hdu_0[0].data.shape[0] > hch:")
            finalMaskHch           = hch
            if VL >= 3 : prntf("stis",20,"GMF finalMaskHch                    : ", finalMaskHch)
            # Do I NOT need to add x in here?
        else:
            if VL >= 3 : prntf("stis",20,"GMF if maskInput_hdu_0[0].data.shape[0] NOT > hch:")
            finalMaskHch           = maskInput_hdu_0[0].data.shape[0]
            hch                    = maskInput_hdu_0[0].data.shape[0] # Keep this uncommented as the starter default value
            if VL >= 3 : prntf("stis",20,"GMF finalMaskHch                    : ", finalMaskHch)
            if VL >= 3 : prntf("stis",20,"GMF hch                             : ", hch)

        # This is relevant to mask 401, but not to narrow, median or wide.
        if hch > file_MaskEffectiveApertureHeight : 
            if VL >= 3 : prntf("stis",20,"GMF if hch > file_MaskEffectiveApertureHeight") 
            hch                              = file_MaskEffectiveApertureHeight # reduce the Highest Common Height to the effective Aperture of 401 mask 
            if VL >= 3 : prntf("stis",20,"GMF hch                             : ", hch)
            finalMaskHch                     = hch

    elif runWedgeDecided == "B" :
        finalMaskHch                     = hch
    
    finalMaskWcw                         = wcw # 200 # Can use same treatment
    finalMaskSize                        = ( finalMaskHch , finalMaskWcw ) # "ny, nx" in Cutout2D documentation
    if VL >= 3 : prntf("stis",20,"GMF finalMaskSize                   : ", finalMaskSize)

    sourceMaskYcenter                    = ( maskInput_hdu_0[0].data.shape[0] / 2 )
    sourceMaskXcenter                    = ( maskInput_hdu_0[0].data.shape[1] / 2 )

    if VL >= 3 : prntf("stis",20,"GMF sourceMaskXcenter               : ", sourceMaskXcenter)
    if VL >= 3 : prntf("stis",20,"GMF sourceMaskYcenter               : ", sourceMaskYcenter)

    sourceMaskCenterPosition             = ( sourceMaskXcenter , sourceMaskYcenter ) # "(x, y) tuple of pixel coordinates" doc
    if VL >= 3 : prntf("stis",20,"GMF sourceMaskCenterPosition        : ", sourceMaskCenterPosition)

    maskCutout                           = Cutout2D ( file_Mask , sourceMaskCenterPosition , size = finalMaskSize )

    file_Mask                             = maskCutout.data    
    maskInput_hdu_0.close()

    
    if VL >= 3 : prntf("stis",20,"GMF done with GMF")
    if VL >= 3 : print()


    if dynMask4ple [ 0 ] : 
        return dynamicMask
    else :
        return file_Mask






def parallel_polling ( 
                                  frame               ,
                                  iiiIndexPoll        ,
                                  rcImport            ,                                  
                                  approvedFlag        ,
                                  tvr                 , 
                                  approvedFrameList   , 
                                  VL                  , 
                                  radonFlag           , 
                                  filename            ,  
                                  SIZAXIS1            , # Width
                                  SIZAXIS2            , # Height
                                  xStellarPoint       , # Horizontal 
                                  yStellarPoint       , # Vertical
                                  udw                 ,
                                  udh                 ,
                                  wcw                 , 
                                  hch 
                                  ):
    
    if VL >= 3 : print()
#    processID = os.getpid()
    VL = 0 # Default

    startCoreCall = time.time()

    
    if VL >= 3 : 
        prntf("poll",20,iiiIndexPoll," filename                         : ", filename)
        prntf("poll",20,iiiIndexPoll," iiiIndexPoll                     : ", iiiIndexPoll)
        prntf("poll",20,iiiIndexPoll," frame[0][0]                      : ", frame[0][0])
        prntf("poll",20,iiiIndexPoll," frame.shape[0]                   : ", frame.shape[0])
        prntf("poll",20,iiiIndexPoll," frame.shape[1]                   : ", frame.shape[1])
        prntf("poll",20,iiiIndexPoll," rcImport                         : ", rcImport)
        prntf("poll",20,iiiIndexPoll," approvedFlag                     : ", approvedFlag)
        prntf("poll",20,iiiIndexPoll," tvr                              : ", tvr)
        prntf("poll",20,iiiIndexPoll," approvedFrameList                : ", approvedFrameList)
        prntf("poll",20,iiiIndexPoll," radonFlag                        : ", radonFlag)

    if approvedFlag == True and tvr == 0 : # Do NOT add a radonFlag condition here, because approved frames have to operate regardless of whether User wants radonCenter...
        tupleOfFilenameAndIndex = ( filename , iiiIndexPoll )
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," tupleOfFilenameAndIndex          : ", tupleOfFilenameAndIndex)

        if tupleOfFilenameAndIndex not in approvedFrameList :  # The list of frames I want
            if VL >= 3 : prntf("poll",20,iiiIndexPoll," REJECTED tupleOfFilenameAndIndex : ", tupleOfFilenameAndIndex)            
            if VL >= 3 : prntf("poll",20,iiiIndexPoll," return None, 0, 1")
            return None, 0, 1
            if VL >= 3 : prntf("poll",20,iiiIndexPoll," returned None, 0, 1")            
    if approvedFlag == False and radonFlag == True : # In the first run, where approvedFlag == False, this grabs ALL frames whether they are tvr == 0 or tvr == 1 
        tupleOfFilenameAndIndex = ( filename , iiiIndexPoll )
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," tupleOfFilenameAndIndex          : ", tupleOfFilenameAndIndex)

    if approvedFlag == True and radonFlag == True and tvr == 1 : # In the second run, where approvedFlag == True, this grabs all references that were not treated by the target frame filter
        tupleOfFilenameAndIndex = ( filename , iiiIndexPoll )
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," tupleOfFilenameAndIndex          : ", tupleOfFilenameAndIndex)
        
    #if HDUList[iiiIndexPoll].header['EXTNAME'] == 'SCI' :
#    if frame.header['EXTNAME'] == 'SCI' : # I already stripped header info by dropping all SCI frames into an np array.
    # So there is no header to speak of at this level, in this function.
    if VL >= 3 : 
        prntf("poll",20,iiiIndexPoll," udw                              : ", udw)
        prntf("poll",20,iiiIndexPoll," udh                              : ", udh)
        prntf("poll",20,iiiIndexPoll," wcw                              : ", wcw)
        prntf("poll",20,iiiIndexPoll," hch                              : ", hch)
        prntf("poll",20,iiiIndexPoll," SIZAXIS1 (Width)                 : ", SIZAXIS1)
        prntf("poll",20,iiiIndexPoll," SIZAXIS2 (Height)                : ", SIZAXIS2)
        prntf("poll",20,iiiIndexPoll," xStellarPoint                    : ", xStellarPoint)
        prntf("poll",20,iiiIndexPoll," yStellarPoint                    : ", yStellarPoint)
        prntf("poll",20,iiiIndexPoll," xStellarPoint-(wcw/2)            : ", xStellarPoint-(wcw/2))
        prntf("poll",20,iiiIndexPoll," xStellarPoint+(wcw/2)            : ", xStellarPoint+(wcw/2))
        prntf("poll",20,iiiIndexPoll," yStellarPoint-hch/2              : ", yStellarPoint-hch/2)
        prntf("poll",20,iiiIndexPoll," yStellarPoint+hch/2              : ", yStellarPoint+hch/2)
    
    radon_wdw = SIZAXIS2 / 2 # radon_window ; Recommended approach per specs.
    if VL >= 3 : prntf("poll",20,iiiIndexPoll," radon_wdw                        : ", radon_wdw)
    
    smooth       = 1
    stisPosition = ( xStellarPoint , yStellarPoint )    
    rcPosition   = ( xStellarPoint , yStellarPoint ) # Leave as default in background.
    if VL >= 3 : prntf("poll",20,iiiIndexPoll," rcPosition                       : ", rcPosition)
    if radonFlag :
        if VL >= 3 : 
            for i in range(5) : prntf("poll",20,iiiIndexPoll," tvr ", tvr, "RC DETERMINATION")
        
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," APPROVED tupleOfFilenameAndIndex : ", tupleOfFilenameAndIndex)

        somePower = 1.1
        plframe = np.zeros(frame.shape)
        rowIndex = 0
        for row in frame:
            if VL >= 4 : prntf("poll",20,iiiIndexPoll," rowIndex : ", rowIndex)
            colIndex = 0
            for col in row:
#                print("rowIndex : ", rowIndex, "  colIndex : ", colIndex)
                radius = math.sqrt (  pow ( rowIndex - yStellarPoint , 2) + pow ( colIndex - xStellarPoint , 2 )  )
                if frame[rowIndex][colIndex] > 0 : 
                    plframe[rowIndex][colIndex] = frame[rowIndex][colIndex] * pow  ( radius , somePower )
                else : 
                    plframe[rowIndex][colIndex] = frame[rowIndex][colIndex]
                if VL >= 5 : prntf("poll",20,iiiIndexPoll, " ",radius, " ", frame[rowIndex][colIndex], ">>", plframe[rowIndex][colIndex])
                colIndex += 1
            rowIndex += 1
        if VL >= 5 : prntf("poll",20,iiiIndexPoll," frame : ", frame)
        if VL >= 5 : prntf("poll",20,iiiIndexPoll," plframe : ", plframe)
#        if VL >= 0 : prntf("poll",20,iiiIndexPoll," USING FRAME")
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," USING PLFRAME and somePower = ", somePower)
        rcPosition = radonCenter.searchCenter ( 
                                           frame.data                            , # watch out for this.
#                                           plframe.data                               ,
                                           xStellarPoint                         , # x
                                           yStellarPoint                         , # y
                                           size_window            = radon_wdw    , #,
                                           size_cost              = 7            ,
                                           m                      = 0.2          ,
                                           M                      = 0.8          , 
                                           smooth                 = smooth
                                           )
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," rcPosition       after radCent   : ", rcPosition)
        
    difference = rcPosition[1] - SIZAXIS2 / 2
    doubleDifference = difference * 2
    doubleDiffInt = int(doubleDifference)
    if VL >= 3 : 
        prntf("poll",20,iiiIndexPoll," difference       after radCent   : ", difference)
        prntf("poll",20,iiiIndexPoll," doubleDifference after radCent   : ", doubleDifference)
        prntf("poll",20,iiiIndexPoll," doubleDiffInt    after radCent   : ", doubleDiffInt)
        prntf("poll",20,iiiIndexPoll," wcw                              : ", wcw)
        prntf("poll",20,iiiIndexPoll," hch                              : ", hch)
        prntf("poll",20,iiiIndexPoll," int ( rcPosition [0] ) * 2 : ", int ( rcPosition [0] ) * 2)

    if rcPosition [0] < SIZAXIS1 / 2 :
        if wcw > int ( rcPosition [0] ) * 2 : # At present, this assumes left side of CCD
            wcw = int ( rcPosition [0] ) * 2
    else :
        wcw = int ( ( CCDPHYSICALWIDTH - rcPosition [0] ) * 2 - 1 )
    
    if VL >= 3 : prntf("poll",20,iiiIndexPoll," int ( rcPosition [1] ) * 2 : ", int ( rcPosition [1] ) * 2)    
    if SIZAXIS2 >= 512 :
        if hch > int ( rcPosition [1] ) * 2 : 
            hch = int ( rcPosition [1] ) * 2
    elif SIZAXIS2 == 80 :
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," hch : ", hch)
        hchPossible = int(SIZAXIS2-abs(SIZAXIS2/2-rcPosition [1])*2)
        if hch > hchPossible :
            hch = hchPossible                    
        if VL >= 3 : prntf("poll",20,iiiIndexPoll," hch : ", hch)
    else :
        if hch > SIZAXIS2 - doubleDiffInt - 1 : 
            hch = SIZAXIS2 - doubleDiffInt - 1

    if VL >= 3 : prntf("poll",20,iiiIndexPoll," wcw              after radCent   : ", wcw)
    if VL >= 3 : prntf("poll",20,iiiIndexPoll," hch              after radCent   : ", hch)
    
    filenameIndexTuple = ( filename , iiiIndexPoll )  
    frameTuple         = (filenameIndexTuple , stisPosition , rcPosition)
    if VL >= 3 : prntf("poll",20,iiiIndexPoll," frameTuple                       : \n                   ", frameTuple)
    if VL >= 3 : 
        for elementIndex in range ( len ( frameTuple ) ) :
            if VL >= 3 : prntf("poll",20, elementIndex, "                                      : ", frameTuple [ elementIndex ] )
        if VL >= 3 : prntf("poll",20,iiiIndexPoll,"                                  : ", frameTuple [ 1 ], frameTuple [ 2 ] )

    if VL >= 3 : print()

    endCoreCall = time.time()

    return frameTuple , 1 , 0 , wcw, hch # If it got this far, it was accepted. 1, 0 means 1 accepted 0 rejected. 






def parallel_implementation (
                                     timerIndex        , # distinct from the iiiIndexPoll, which is keyed off STIS ImageHDU Index 
                                     implRecord        , 
                                     VL                ,
                                     DQmax             ,
                                     yesApplyMaskFlag  ,
                                     divertMaskFlag    ,
                                     maskToBeUsed      ,
                                     Cutout2DSize
                                    ):
    if VL >= 3 : print()
#    processID = os.getpid()
    VL = 0 # Default
    
    startCoreCall    = time.time()
    implRecordShape  = implRecord[0].shape
    verticalOffset   = implRecordShape[0] / 2 - implRecord[3][1]
    horizontalOffset = implRecordShape[1] / 2 - implRecord[3][0]

    if VL >= 3 : 
        prntf("imp2",20,timerIndex," implRecord[0]       : \n{{{", implRecord[0] , "}}}\n") # SCI
        prntf("imp2",20,timerIndex," implRecord[3]       : ", implRecord[3]) # centers
        prntf("imp2",20,timerIndex," Cutout2DSize        : ", Cutout2DSize )
        prntf("imp2",20,timerIndex," implRecord[0].shape : ", implRecord[0].shape )
        prntf("imp2",20,timerIndex," implRecordShape     : ", implRecordShape )
        prntf("imp2",20,timerIndex," verticalOffset      : ", verticalOffset )
        prntf("imp2",20,timerIndex," horizontalOffset    : ", horizontalOffset )
    
    SCIcutout           = Cutout2D     ( 
                                        implRecord[0]            , # SCI
                                        position = implRecord[3] , # center 
                                        size     = Cutout2DSize  , 
                                        wcs      = implRecord[4]   # wcsIndexed
                                        )
    ERRcutout           = Cutout2D     ( 
                                        implRecord[1]            , # ERR 
                                        position = implRecord[3] , # center 
                                        size     = Cutout2DSize  , 
                                        wcs      = implRecord[4]   # wcsIndexed
                                        )    
    DQcutout            = Cutout2D     ( 
                                        implRecord[2]            , # DQ 
                                        position = implRecord[3] , # center 
                                        size     = Cutout2DSize  , 
                                        wcs      = implRecord[4]   # wcsIndexed
                                        )    
    for jjjRowIndex in range ( len ( SCIcutout.data ) ) :
        for iiiColIndex in range ( len ( SCIcutout.data [ jjjRowIndex ] ) ) :
            
            if DQcutout.data [jjjRowIndex][iiiColIndex] > DQmax:
                SCIcutout.data [jjjRowIndex][iiiColIndex] = np.nan
                ERRcutout.data [jjjRowIndex][iiiColIndex] = np.nan

            if   implRecord[5] == 4 and SCIcutout.data [jjjRowIndex][iiiColIndex] >  130000 / 4 :
                SCIcutout.data [jjjRowIndex][iiiColIndex] = np.nan
                ERRcutout.data [jjjRowIndex][iiiColIndex] = np.nan
                    
            elif implRecord[5] == 1 and SCIcutout.data [jjjRowIndex][iiiColIndex] >   33000:
                SCIcutout.data [jjjRowIndex][iiiColIndex] = np.nan
                ERRcutout.data [jjjRowIndex][iiiColIndex] = np.nan
            
            if yesApplyMaskFlag == True and  divertMaskFlag == False :
                
                if  maskToBeUsed[jjjRowIndex][iiiColIndex] == 0 :
                    SCIcutout.data [jjjRowIndex][iiiColIndex] = np.nan
                    ERRcutout.data [jjjRowIndex][iiiColIndex] = np.nan
    
    app_SCI_img = SCIcutout.data
    app_ERR_img = ERRcutout.data
    center = ( SCIcutout.input_position_cutout[0] , SCIcutout.input_position_cutout[1] )
    
    return app_SCI_img, app_ERR_img, center





def primary_function ( 
                                config                                  ,
                                udw                                     , # x dimension listed first
                                udh                                     , # y 
                                userTrgListRefList                      , 
                                outputFolder                            ,
                                yesApplyMaskFlag                        , 
                                divertMaskFlag                          ,  
                                maskFilename                            ,
                                fileMaskWedgeCode                       ,
                                DatasetPlotFlag         = False         ,
                                useRefandTrgFlag        = False         , 
                                DQmax                   = 512           , 
                                sizeMax                 = None          , 
                                VL                      = 0             , # Verbosity Level
                                approvedFlag            = False         , # False is conservative
                                radonFlag               = False         ,  # False is conservative
                                userTestingNumberFrames = 0             ,
                                dynMask4ple             = (False,0,0,0) , 
                                runWedgeDecided         = 'A'
                                
                              ):
    """
        Gets arrays specified by pyklip : data array, centers array, pas array, filename array
        In addition to that, it returns items that are of other use.
        Args:
            udh                           : Int.   U ser Determined Height / User Desired Height
            udw                           : Int.    User Determined Width  / User Desired Width
            userTrgListRefList            : tuple.  User's intended [ TARGET list, REFERENCE list ]
            yesApplyMaskFlag              : Bool.   True means: Yes, import the mask and create mask array.
            divertMaskFlag                : Bool.   True means: Yes, divert mask array to the function return
                                                             And No, do Not use mask array on the SCI frames.
            maskFilename                  : String. File containing mask.
            useRefandTrgFlag   = False    : Bool.   
            DQmax              = 512      : Int.    Maximum value for a pixel in the Data Quality STIS frame.  
            sizeMax            = None     : Int.    Diagnostic for development.
            VL                 = 0        : Int.    Verbosity Level
            approvedFlag       = False    : Bool.   True means: Yes, import the approvedFrames file
            radonFlag          = False    : Bool.   True means: Yes, perform radonCenter on all frames 
        
        Return:
            input_SCI                     : floats numpy array. Data from SCI frames. 
            input_centers                 : floats numpy array. Data from SCI headers' keywords.
            pas                           : floats numpy array. Data from SCI headers' WCS values.
            input_filenames               : strings list. Strings from SCI headers.
            rollAngleSet                  : floats list. Unique roll angles.
            rollAngleAverage              : float. Average of all roll angles, for fake planet injection.
            HDUList[0].header['RA_TARG' ] : float. RA.
            HDUList[0].header['DEC_TARG'] : float. DEC.
                                            TODO: Could send as tuple.
            rootnameList                  : strings list. Convenience for User to verify processing.
            sizeMax                       : tuple. Highest Common Height, Widest Common Width.
            tvrList                       : Int list. The critical discriminator between TARGET and REFERENCE.
            input_ERR                     : floats numpy array. Data from ERR frames. For nmf_imaging.
            maskToBeUsed                  : Int numpy array. Mask pyklip would use, but saved for nmf_imaging.
            approvedFrameList             : list of tuples
                                            Each tuple holding string filename, int ImageHDU index, float stddev
    """
    tvr = -1
    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","VL                               : ", VL)
        prntf("stis",20," ",tvr," ","dynMask4ple                      : ", dynMask4ple)
        prntf("stis",20," ",tvr," ","fileMaskWedgeCode                : ", fileMaskWedgeCode)
        prntf("stis",20," ",tvr," ","runWedgeDecided                  : ", runWedgeDecided)
        prntf("stis",20," ",tvr," ","outputFolder                     : ", outputFolder)
        prntf("stis",20," ",tvr," ","yesApplyMaskFlag                 : ", yesApplyMaskFlag)
        prntf("stis",20," ",tvr," ","divertMaskFlag                   : ", divertMaskFlag)
    #   G E T   L I S T   O F   P R E - A P P R O V E D   F R A M E S
    #   G E T   L I S T   O F   P R E - A P P R O V E D   F R A M E S
    incomingApprovedFrameList  = []   
    approvedFrameList          = []
    globFilesname              = "*"
    globFilesList              = glob.glob ( outputFolder + globFilesname )
    if VL >= 3 : prntf("stis",20," ",tvr," ","len(globFilesList)           : ", len(globFilesList))
    if VL >= 3 : prntf("stis",20," ",tvr," ","globFilesList                : ", globFilesList)
    for globbedFile in globFilesList:
        if "__atf__" in globbedFile:
            approvedTargetFrameFile = globbedFile 
    
    if approvedFlag == True : # change this to " if (passed in parameter filename) not None : "         
        if len ( approvedTargetFrameFile ) == 0 :            
            if VL >= 3 : prntf("stis",20," ",tvr," ","Make sure there is a file with approved frames")
            raise ValueError("len ( approvedTargetFrameFile ) == 0")
        if approvedTargetFrameFile is None :            
            if VL >= 3 : prntf("stis",20," ",tvr," ","Make sure there is a file with approved frames")            
            raise ValueError("if globFilesList is None")

        if VL >= 3 : prntf("stis",20," ",tvr," ","globFilesList[0] : ", globFilesList[0])
        if VL >= 3 : prntf("stis",20," ",tvr," ","approvedTargetFrameFile : ", approvedTargetFrameFile)
        justFilenamesList          = []
        with open ( approvedTargetFrameFile ) as openApprovedFrameFile :            
            for record in openApprovedFrameFile  :
                if record == "\n" : continue # you can separate groups of tuples with spaces
                record             = record.strip() #remove newline at the end
                record             = record.replace("'","")
                record             = record.replace("[","")
                record             = record.replace("]","")
                record             = record.replace(" ","")
                tokenizeAtComma    = record.split(',')
                tupleOfFilenameAndIndex             = ( tokenizeAtComma[0], int ( tokenizeAtComma[1] ) )
                incomingApprovedFrameList.append ( tupleOfFilenameAndIndex )        

                justFilenamesList.append ( tokenizeAtComma[0] )
        uniqueFilenamesList             = list ( set ( justFilenamesList ) ) # reduce to  unique filenames
        justFilenamesList.clear()
        uniqueFilenamesList.sort()
        if VL >= 1 : prntf("stis",20," ",tvr," ","uniqueFilenamesList            : ", uniqueFilenamesList)
        
        approvedFrameList               = list ( set ( incomingApprovedFrameList ) )
        incomingApprovedFrameList.clear()
        approvedFrameList.sort()
        if VL >= 3 : prntf("stis",20," ",tvr," ","approvedFrameList              : ", approvedFrameList)
        for uniqueApprovedRecord in approvedFrameList :
            if VL >= 3 : prntf("stis",20," ",tvr," ","uniqueApprovedRecord           : ", uniqueApprovedRecord)
        openApprovedFrameFile.close()
    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","END OF if approvedFlag == True :")
    
    hch = udh 
    wcw = udw
    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","udh                              : ", udh)
        prntf("stis",20," ",tvr," ","udw                              : ", udw)
        prntf("stis",20," ",tvr," ","hch                              : ", hch)
        prntf("stis",20," ",tvr," ","wcw                              : ", wcw)
        prntf("stis",20," ",tvr," ","sizeMax                          : ", sizeMax)    
        prntf("stis",20," ",tvr," ","DQmax                            : ", DQmax)
    
    radCentTotalTimeCost     = 0    
    input_numExposures       = 0 
    rootnameList             = []


    RADCENTX = 0
    RADCENTY = 0
    
    if yesApplyMaskFlag == True :
        
        # This is obviated by the newer code below. It harms nothing hanging out for now. 
        udh, udw = get_file_mask ( maskFilename, VL )
        if VL >= 3 : prntf("stis",20," ",tvr," ","udh                              : ", udh)
        if VL >= 3 : prntf("stis",20," ",tvr," ","udw                              : ", udw)
        # If radonCenter is involved, remove the choice from the User to determine limiting size
        # radonCenter needs a large minimum amount of data to remain consistent across bad, saturated frames.
        if radonFlag and divertMaskFlag :
            hch = udh
            wcw = udw
        
        
        # NEW CODE BELOW
        maskHeight, maskWidth = get_file_mask ( maskFilename, VL )
        if VL >= 3 : prntf("stis",20," ",tvr," ","maskHeight                       : ", maskHeight)
        if VL >= 3 : prntf("stis",20," ",tvr," ","maskWidth                        : ", maskWidth )
        # If radonCenter is involved, remove the choice from the User to determine limiting size
        # radonCenter needs a large minimum amount of data to remain consistent across bad, saturated frames.
        if radonFlag : #and divertMaskFlag :
            if hch < maskHeight :
                udh = hch = maskHeight
            if wcw < maskWidth :
                udw = wcw = maskWidth       

    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","maskHeight                       : ", maskHeight)
        prntf("stis",20," ",tvr," ","maskWidth                        : ", maskWidth )
        prntf("stis",20," ",tvr," ","udh                              : ", udh)
        prntf("stis",20," ",tvr," ","udw                              : ", udw       )
        prntf("stis",20," ",tvr," ","hch                              : ", hch)
        prntf("stis",20," ",tvr," ","wcw                              : ", wcw       )
    
    if VL >= 1 : 
        for i in range(2) : 
            prntf("stis",20," ",tvr," ","P O L L   F O R   H C H   A N D   W C W")
        
    input_centers_trgref2        = [[],[]] # empty list    
    pas_trgref2                  = [[],[]] # empty list    
    input_filenames_trgref2      = [[],[]] # empty list
    frameTupleList_trgref        = [[],[]] # A PAIR of LISTS, each holding different numbers of FRAME RECORDS, each itself a LIST.
    RADEC_trgref                 = [[],[]]
    filenameList                 = []
    frameTupleList_trgref[0].clear() # The LIST that holds TARGET    frame records 
    frameTupleList_trgref[1].clear() # The LIST that holds REFERENCE frame records

    tvr                                = 0
    runTrgHalfWedgePixels              = 0
    runTrgModHeaderHalfWedgeThickPixel = 0
    runAHalfWedgeDiffY1Y0Pixels        = 0
    runBHalfWedgeDiffX1X0Pixels        = -1000
    differentTrgHeaderWedgesExist      = False

    # Get wedge code from first file, according to that file's headers
    firstPROPAPER = fits.open ( userTrgListRefList[0][0] )[0].header[ 'PROPAPER' ]
    firstPROPAPER_2 = firstPROPAPER.replace("WEDGE","")
    if   "A" in firstPROPAPER_2 : 
        firstHeaderWedgeLetter = "A"
    elif "B" in firstPROPAPER_2 : 
        firstHeaderWedgeLetter = "B"
    firstHeaderWedgeThickArcsec     = float ( firstPROPAPER_2.replace ( firstHeaderWedgeLetter, "" ) )
    firstHeaderWedgeThickPixel      = firstHeaderWedgeThickArcsec / PLATESCALE
    firstHeaderHalfWedgeThickArcsec = firstHeaderWedgeThickArcsec / 2
    firstHeaderHalfWedgeThickPixel  = firstHeaderWedgeThickPixel / 2 
    firstHeaderWedgeCode            = firstHeaderWedgeLetter + str ( int ( firstHeaderWedgeThickArcsec * 10 ) )
    if tvr == 0 :
        lastTrgHeaderWedgeCode      = firstHeaderWedgeCode
    if VL >= 3 : prntf("stis",20," ",tvr," ","lastTrgHeaderWedgeCode           : ", lastTrgHeaderWedgeCode)

    for filelist in userTrgListRefList :
        if useRefandTrgFlag == False and filelist == userTrgListRefList[1] : # For when pyklip requires ONLY target frames
            continue
        if VL >= 1 : 
            for i in range(2) : prntf("stis",20," ",tvr," ","tvr polling loop tvr polling loop tvr polling loop tvr polling loop : ", tvr)

        totalApproved = 0 # This is on the basis of target frames approved OR reference frames approved ; it does not keep count of both (degenerate)
        totalRejected = 0 # This is on the basis of target frames rejected OR reference frames rejected ; it does not keep count of both (degenerate)
        totalApproved2 = 0
        totalRejected2 = 0
        

        HDUSCIDatalist = []
        
        # dnfn = 'Directory Name and File Name', where Directory is at local level to this script, STIS.py.
        for dnfn in filelist :
            HDUList                = fits.open ( dnfn )
            SUBARRAY        = bool ( HDUList [ 0 ].header[ 'SUBARRAY' ] )             # / SCIENCE INSTRUMENT CONFIGURATION   
            POSTARG1               = HDUList [ 0 ].header[ 'POSTARG1' ]               # / READOUT DEFINITION PARAMETERS
            POSTARG2               = HDUList [ 0 ].header[ 'POSTARG2' ]               # / READOUT DEFINITION PARAMETERS
            PROPAPER               = HDUList [ 0 ].header[ 'PROPAPER' ]               # / SCIENCE INSTRUMENT CONFIGURATION        
            CENTERA1               = HDUList [ 0 ].header[ 'CENTERA1' ]               # / READOUT DEFINITION PARAMETERS
            CENTERA2               = HDUList [ 0 ].header[ 'CENTERA2' ]               # / READOUT DEFINITION PARAMETERS
            SIZAXIS1               = HDUList [ 0 ].header[ 'SIZAXIS1' ]               # / READOUT DEFINITION PARAMETERS     
            SIZAXIS2               = HDUList [ 0 ].header[ 'SIZAXIS2' ]               # / READOUT DEFINITION PARAMETERS
            CRPIX1                 = HDUList [ 1 ].header[ 'CRPIX1'   ]               # / World Coordinate System
            CRPIX2                 = HDUList [ 1 ].header[ 'CRPIX2'   ]               # / World Coordinate System
            LTV1                   = HDUList [ 1 ].header[ 'LTV1'     ]               # / World Coordinate System
            LTV2                   = HDUList [ 1 ].header[ 'LTV2'     ]               # / World Coordinate System
            stisFrameHeight        = HDUList [ 1 ].data.shape[0]                      # stisFrameHeight
            stisFrameWidth         = HDUList [ 1 ].data.shape[1]                      # stisFrameWidth
            
            trgModHeaderHalfWedgeThickPixel = 0
            refModHeaderHalfWedgeThickPixel = 0
            refHalfWedgeArcsecs    = 0
            refHalfWedgePixels     = 0
            
            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","tvr                              : ", tvr)
                prntf("stis",20," ",tvr," ","dnfn                             : ", dnfn)
                prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist    : ", differentTrgHeaderWedgesExist)
                prntf("stis",20," ",tvr," ","fileMaskWedgeCode                : ", fileMaskWedgeCode)
                prntf("stis",20," ",tvr," ","PROPAPER                         : ", PROPAPER)
                prntf("stis",20," ",tvr," ","SUBARRAY                         : ", SUBARRAY)
                prntf("stis",20," ",tvr," ","POSTARG1                         : ", POSTARG1)
                prntf("stis",20," ",tvr," ","POSTARG2                         : ", POSTARG2)
                prntf("stis",20," ",tvr," ","CENTERA1                         : ", CENTERA1)
                prntf("stis",20," ",tvr," ","CENTERA2                         : ", CENTERA2)
                prntf("stis",20," ",tvr," ","SIZAXIS1                         : ", SIZAXIS1)
                prntf("stis",20," ",tvr," ","SIZAXIS2                         : ", SIZAXIS2)
                prntf("stis",20," ",tvr," ","CRPIX1                           : ", CRPIX1)
                prntf("stis",20," ",tvr," ","CRPIX2                           : ", CRPIX2)
                prntf("stis",20," ",tvr," ","LTV1                             : ", LTV1)
                prntf("stis",20," ",tvr," ","AWEDGESLOPE                      : ", AWEDGESLOPE)
                prntf("stis",20," ",tvr," ","BWEDGESLOPE                      : ", BWEDGESLOPE)

            PROPAPER_2 = PROPAPER.replace("WEDGE","")
            if   "A" in PROPAPER_2 : 
                headerWedgeLetter = "A"
            elif "B" in PROPAPER_2 : 
                headerWedgeLetter = "B"
            headerWedgeThickArcsec     = float ( PROPAPER_2.replace ( headerWedgeLetter, "" ) )
            headerWedgeThickPixel      = headerWedgeThickArcsec / PLATESCALE
            headerHalfWedgeThickArcsec = headerWedgeThickArcsec / 2
            headerHalfWedgeThickPixel  = headerWedgeThickPixel / 2 
            currentHeaderWedgeCode     = headerWedgeLetter + str ( int ( headerWedgeThickArcsec * 10 ) ) 
            
            if VL >= 3 :
                prntf("stis",20," ",tvr," ","PROPAPER_2                       : ", PROPAPER_2)
                prntf("stis",20," ",tvr," ","headerWedgeLetter                : ", headerWedgeLetter)
                prntf("stis",20," ",tvr," ","headerWedgeThickArcsec           : ", headerWedgeThickArcsec)
                prntf("stis",20," ",tvr," ","headerWedgeThickPixel            : ", headerWedgeThickPixel) 
                prntf("stis",20," ",tvr," ","headerHalfWedgeThickArcsec       : ", headerHalfWedgeThickArcsec)
                prntf("stis",20," ",tvr," ","headerHalfWedgeThickPixel        : ", headerHalfWedgeThickPixel)
                prntf("stis",20," ",tvr," ","currentHeaderWedgeCode           : ", currentHeaderWedgeCode)

            POSTARG1_pixels      = POSTARG1    / PLATESCALE
            POSTARG2_pixels      = POSTARG2    / PLATESCALE                        
            CRPIX1mLTV1          = CRPIX1      - LTV1
            CRPIX2mLTV2          = CRPIX2      - LTV2
            xActual              = CRPIX1mLTV1 + POSTARG1_pixels
            yActual              = CRPIX2mLTV2 + POSTARG2_pixels
            CENTERA1mxActual     = CENTERA1    - xActual
            CENTERA2myActual     = CENTERA2    - yActual
            pixelsFromLeft       = math.ceil ( xActual ) # still needs handling for which half left or right
            pixelsFromBottom     = math.ceil ( yActual ) # still needs handling for which half top or bottom 
            frameMaxWidth        = pixelsFromLeft * 2
            xStellarPoint        = CRPIX1  # Default ; will get overridden in specific cases
            yStellarPoint        = CRPIX2  # Default for 110, 427, 512, 1024 ( 80 and 316 must override )
            xDifference          = xActual     - CRPIX1 # Accounts for LTV and POS-TARG together
            yDifference          = yActual     - CRPIX2
            BHalfWedgeDiffX1X0Pixels = xDifference * BWEDGESLOPE
            AHalfWedgeDiffY1Y0Pixels = yDifference * AWEDGESLOPE
            
            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","POSTARG1_pixels                  : ", POSTARG1_pixels)
                prntf("stis",20," ",tvr," ","POSTARG2_pixels                  : ", POSTARG2_pixels)   
                prntf("stis",20," ",tvr," ","CRPIX1mLTV1                      : ", CRPIX1mLTV1)
                prntf("stis",20," ",tvr," ","CRPIX2mLTV2                      : ", CRPIX2mLTV2)
                prntf("stis",20," ",tvr," ","xActual                          : ", xActual, " = CRPIX1mLTV1 + POSTARG1_pixels")
                prntf("stis",20," ",tvr," ","yActual                          : ", yActual, " = CRPIX2mLTV2 + POSTARG2_pixels")
                prntf("stis",20," ",tvr," ","CENTERA1mxActual                 : ", CENTERA1mxActual)
                prntf("stis",20," ",tvr," ","CENTERA2myActual                 : ", CENTERA2myActual)
                prntf("stis",20," ",tvr," ","xStellarPoint                    : ", xStellarPoint, " = CRPIX1")
                prntf("stis",20," ",tvr," ","yStellarPoint                    : ", yStellarPoint, " = CRPIX2")
                prntf("stis",20," ",tvr," ","xDifference                      : ", xDifference, " = xActual     - CRPIX1")
                prntf("stis",20," ",tvr," ","dnfn                             : ", dnfn)
                prntf("stis",20," ",tvr," ","runBHalfWedgeDiffX1X0Pixels      : ", runBHalfWedgeDiffX1X0Pixels)
                prntf("stis",20," ",tvr," ","   BHalfWedgeDiffX1X0Pixels      : ", BHalfWedgeDiffX1X0Pixels, " = xDifference * BWEDGESLOPE")
            if tvr == 0 and runBHalfWedgeDiffX1X0Pixels < BHalfWedgeDiffX1X0Pixels :
                if VL >= 3 : prntf("stis",20," ",tvr," ","if runBHalfWedgeDiffX1X0Pixels > BHalfWedgeDiffX1X0Pixels :")
                if VL >= 3 : prntf("stis",20," ",tvr," ","runBHalfWedgeDiffX1X0Pixels      : ", runBHalfWedgeDiffX1X0Pixels)
                if VL >= 3 : prntf("stis",20," ",tvr," ","BHalfWedgeDiffX1X0Pixels         : ", BHalfWedgeDiffX1X0Pixels)
                runBHalfWedgeDiffX1X0Pixels = BHalfWedgeDiffX1X0Pixels
                if VL >= 3 : prntf("stis",20," ",tvr," ","runBHalfWedgeDiffX1X0Pixels      : ", runBHalfWedgeDiffX1X0Pixels)
                
            if VL >= 3 : prntf("stis",20," ",tvr," ","runBHalfWedgeDiffX1X0Pixels      : ", runBHalfWedgeDiffX1X0Pixels)
            if tvr == 0 and headerWedgeLetter == "B" :
                trgModHeaderHalfWedgeThickPixel = headerHalfWedgeThickPixel + BHalfWedgeDiffX1X0Pixels
            if tvr == 1 and headerWedgeLetter == "B" :
                refModHeaderHalfWedgeThickPixel = headerHalfWedgeThickPixel + BHalfWedgeDiffX1X0Pixels
            if runTrgModHeaderHalfWedgeThickPixel < trgModHeaderHalfWedgeThickPixel :
                runTrgModHeaderHalfWedgeThickPixel = trgModHeaderHalfWedgeThickPixel
            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","   trgModHeaderHalfWedgeThickPixel : ", trgModHeaderHalfWedgeThickPixel)
                prntf("stis",20," ",tvr," ","   refModHeaderHalfWedgeThickPixel : ", refModHeaderHalfWedgeThickPixel)
                prntf("stis",20," ",tvr," ","runTrgModHeaderHalfWedgeThickPixel : ", runTrgModHeaderHalfWedgeThickPixel)
            if refModHeaderHalfWedgeThickPixel > runTrgModHeaderHalfWedgeThickPixel : 
                if VL >= 3 : prntf("stis",20," ",tvr," ","if refModHeaderHalfWedgeThickPixel > runTrgModHeaderHalfWedgeThickPixel :")
                if VL >= 3 : prntf("stis",20," ",tvr," ","The file containing this reference frame should be removed because the frame has a higher final wedge thickness than the target. This frame will be rejected automatically.")
                warnings.warn("The file containing this reference frame should be removed because the frame has a higher final wedge thickness than the target. This frame will be rejected automatically.")
                continue
            
            if VL >= 3 : prntf("stis",20," ",tvr," ","currentHeaderWedgeCode           : ", currentHeaderWedgeCode)
            if VL >= 3 : prntf("stis",20," ",tvr," ","lastTrgHeaderWedgeCode           : ", lastTrgHeaderWedgeCode)
                
            if tvr == 0 and len ( lastTrgHeaderWedgeCode ) > 0 and currentHeaderWedgeCode != lastTrgHeaderWedgeCode :
                differentTrgHeaderWedgesExist          = True
                if VL >= 3 : prntf("stis",20," ",tvr," ","if len(lastTrgHeaderWedgeCode)>0 and dnfnSplit[0] != lastTrgHeaderWedgeCode :")
                if VL >= 3 : prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist    : ", differentTrgHeaderWedgesExist)
            if tvr == 0 and currentHeaderWedgeCode > fileMaskWedgeCode :
                differentTrgHeaderWedgesExist          = True
                if VL >= 3 : prntf("stis",20," ",tvr," ","if tvr == 0 and currentHeaderWedgeCode > fileMaskWedgeCode and runWedgeDecided == A :")
                if VL >= 3 : prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist    : ", differentTrgHeaderWedgesExist)
            
            if tvr == 0 : 
                lastTrgHeaderWedgeCode                 = currentHeaderWedgeCode
                
                if   'A' in currentHeaderWedgeCode  : 
                    trgWedgeArcsecs              = int ( currentHeaderWedgeCode.replace('A','') ) / 10
                elif 'B' in currentHeaderWedgeCode  : 
                    trgWedgeArcsecs              = int ( currentHeaderWedgeCode.replace('B','') ) / 10            
                trgHalfWedgeArcsecs              = trgWedgeArcsecs / 2
                trgHalfWedgePixels               = int ( trgHalfWedgeArcsecs / PLATESCALE )
                if runTrgHalfWedgePixels         < trgHalfWedgePixels :
                    runTrgHalfWedgePixels        = trgHalfWedgePixels

            if tvr == 1 :
                if   'A' in currentHeaderWedgeCode  : 
                    refWedgeArcsecs              = int ( currentHeaderWedgeCode.replace('A','') ) / 10
                elif 'B' in currentHeaderWedgeCode  : 
                    refWedgeArcsecs              = int ( currentHeaderWedgeCode.replace('B','') ) / 10            
                refHalfWedgeArcsecs              = refWedgeArcsecs / 2
                refHalfWedgePixels               = int ( refHalfWedgeArcsecs / PLATESCALE )

                if refHalfWedgePixels            > runTrgHalfWedgePixels : 
                    if VL >= 3 : prntf("stis",20," ",tvr," ","REJECTING REFERENCE frame as having too large of wedge          : ", dnfn, "\n")
                    continue

            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist    : ", differentTrgHeaderWedgesExist)
                prntf("stis",20," ",tvr," ","trgWedgeArcsecs                  : ", trgWedgeArcsecs)
                prntf("stis",20," ",tvr," ","trgHalfWedgeArcsecs              : ", trgHalfWedgeArcsecs)
                prntf("stis",20," ",tvr," ","trgHalfWedgePixels               : ", trgHalfWedgePixels)
                prntf("stis",20," ",tvr," ","runTrgHalfWedgePixels            : ", runTrgHalfWedgePixels)
                prntf("stis",20," ",tvr," ","refHalfWedgeArcsecs              : ", refHalfWedgeArcsecs)
                prntf("stis",20," ",tvr," ","refHalfWedgePixels               : ", refHalfWedgePixels, "\n")
            
            filename                 = os.path.basename ( dnfn )
            if VL >= 3 : prntf("stis",20," ",tvr," ","filename                         : ", filename)
            directoryPath            = os.path.dirname ( dnfn )
            split                    = os.path.splitext ( os.path.basename ( dnfn ) )
            token_                   = split[0].split('_')
            rootname                 = token_[0]

            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","sys.getsizeof ( HDUList )        : ", sys.getsizeof ( HDUList ) )
                prntf("stis",20," ",tvr," ","len ( HDUList )                  : ", len ( HDUList )     , "physical elements in the list, counting from 1 to total")
                prntf("stis",20," ",tvr," ","len ( HDUList ) - 1              : ", len ( HDUList ) - 1 , "indices available, starting from index of 0")
            
            HDUSCIDatalist.clear()
            
            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","len ( HDUSCIDatalist)            : ", len ( HDUSCIDatalist) )
                prntf("stis",20," ",tvr," ","HDUSCIDatalist                   : ", HDUSCIDatalist )
                prntf("stis",20," ",tvr," ","userTestingNumberFrames          : ", userTestingNumberFrames )
            
            # First, set the default
            iiiIndexMax = len ( HDUList ) - 1 # default
            if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax )
            
            # Second, attempt to comply with User request for number of testing frames
            if userTestingNumberFrames != 0 :  # User testing
                iiiIndexMax = userTestingNumberFrames * 3
                if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax )
            
            # Third, if User requested testing frames is more than the file can offer, revert back to default 
            if iiiIndexMax > len ( HDUList ) - 1 :
                iiiIndexMax = len ( HDUList ) - 1 # revert back to default
                if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax )

            bypassRCFlag = True # Assume that EVERY SINGLE frame in the file has its own RADCENTX and RADCENTY
            bypassRCFlag = False # TESTING. Make STIS.py do radonCenter FRESH for every frame.
            allRadonCenterImports = [] # All radonCenter imports from this present file
            
            # Get the mode of EXPTIME, or barring that get the nonzero MIN of EXPTIME
            exptimeList = [ 0 ] * int ( ( len ( HDUList ) - 1 ) / 3 )
            for hhhIndex in range (0, len ( exptimeList ) ):
                if VL >= 3 : prntf("stis",20," ",tvr," ","hhhIndex                         : ", hhhIndex, " filename : ", filename, " EXPTIME : ", HDUList [ ( hhhIndex * 3 ) + 1 ].header[ 'EXPTIME' ]  )                
                exptimeList [ hhhIndex ] = HDUList [ ( hhhIndex * 3 ) + 1 ].header[ 'EXPTIME' ] 
            import statistics
            # Try to find the mode of the file (assuming there is a mode)
            exptimeMode = 0
            exptimeMin = 0
            try : 
                exptimeMode = statistics.mode ( exptimeList )
            # And if there is not a mode, because there are several different unique values, get the briefest EXPTIME
            except :
                exptimeMin = min ( exptimeList )
                if VL >= 3 : prntf("stis",20," ",tvr," ","There is no EXPTIME mode so we will use the shortest EXPTIME (", exptimeMin, ") as a standard,",  )
                if VL >= 3 : prntf("stis",20," ",tvr," ","and we will use any EXPTIME that is within 'EXPTIME_RATIO' : ", EXPTIME_RATIO, " of that shortest EXPTIME (", exptimeMin, ")")
            
            if VL >= 3 : prntf("stis",20," ",tvr," ","exptimeMin                       : ", exptimeMin )
            
            STISx_minus_RCx_accumulator = 0
            STISy_minus_RCy_accumulator = 0
            
            startAppend = time.time()

            for iiiIndex in range (1, iiiIndexMax + 1 ): # in range syntax will stop at one less than last value                
                if VL >= 3 : 
                    prntf("stis",20," ",tvr," ","dnfn                                     : ", dnfn)
                    prntf("stis",20," ",tvr," ","iiiIndex                                 : ", iiiIndex)
                    prntf("stis",20," ",tvr," ","HDUList[iiiIndex].data[0][0]             : ", HDUList[iiiIndex].data[0][0] )
                    prntf("stis",20," ",tvr," ","sys.getsizeof ( HDUList[iiiIndex].data ) : ", sys.getsizeof ( HDUList[iiiIndex].data ) )
                if VL >= 4 : prntf("stis",20," ",tvr," ","HDUList[iiiIndex].data                   : \n", HDUList[iiiIndex].data )
                
                filenameList.append ( filename ) # We need to get a list that gives the filename for every frame in the filelist
                
                if HDUList[iiiIndex].header['EXTNAME'] == 'SCI' : 
                    if  ( 
                            (                     HDUList[iiiIndex].header['EXPTIME'] == exptimeMode                   )
                            or 
                            ( exptimeMin  > 0 and HDUList[iiiIndex].header['EXPTIME'] <= exptimeMin  * EXPTIME_RATIO )  
                            or
                            ( exptimeMode > 0 and HDUList[iiiIndex].header['EXPTIME'] <= exptimeMode * EXPTIME_RATIO )
                        ) :
                    
                        if VL >= 3 : prntf("stis",20," ",tvr," ","filename, iiiIndex                       : ",filename, iiiIndex)
                        if VL >= 3 : prntf("stis",20," ",tvr," ","HDUList[iiiIndex].header['EXPTIME']      : ", HDUList[iiiIndex].header['EXPTIME'])
                        
                        HDUSCIDatalist.append ( HDUList[iiiIndex].data ) # Keep in mind that I benchmarked exact allocation and it was slower...                    
                        if VL >= 3 : prntf("stis",20," ",tvr," ","sys.getsizeof ( HDUSCIDatalist )         : ", sys.getsizeof ( HDUSCIDatalist ) )
                        if VL >= 3 : prntf("stis",20," ",tvr," ","          len ( HDUSCIDatalist )         : ", len ( HDUSCIDatalist ) )
                        
                        #if 'RADCENTX' in  HDUList [ iiiIndex ].header and 'RADCENTY' in  HDUList [ iiiIndex ].header and yesImportRadonCenterValuesFromHeaderFlag :
                        if 'RADCENTX' in  HDUList [ iiiIndex ].header and 'RADCENTY' in  HDUList [ iiiIndex ].header :
                            if VL >= 3 : prntf("stis",20," ",tvr," ","if 'RADCENTX' in  HDUList [ iiiIndex ].header and 'RADCENTY' in  HDUList [ iiiIndex ].header :")
                            RADCENTX   = HDUList [ iiiIndex ].header[ 'RADCENTX' ]
                            RADCENTY   = HDUList [ iiiIndex ].header[ 'RADCENTY' ]
                            allRadonCenterImports.append ( ( float(RADCENTX) , float(RADCENTY) ) )
                            if VL >= 3 : prntf("stis",20," ",tvr," ","RADCENTX                                 : [", RADCENTX, "]")
                            if VL >= 3 : prntf("stis",20," ",tvr," ","RADCENTY                                 : [", RADCENTY, "]")
                                
                        else : 
                            if VL >= 3 : prntf("stis",20," ",tvr," ","RADCENTX and RADCENTY are not both in header.")
                            if VL >= 3 : prntf("stis",20," ",tvr," ","RadonCenter must be run and both values saved to header.")
                            bypassRCFlag = False
                            allRadonCenterImports.append ( ( 0.0 , 0.0 ) ) # Without these placeholder values, polling won't run.

                        RADEC_trgref[tvr].append ( ( HDUList[0].header[ 'RA_TARG'  ], HDUList[0].header[ 'DEC_TARG' ] ) )
                        if VL >= 3 : prntf("stis",20," ",tvr," ","len(RADEC_trgref[tvr])                   : ", len(RADEC_trgref[tvr]) )
                        if VL >= 5 : prntf("stis",20," ",tvr," ","RADEC_trgref                             : ", RADEC_trgref )
                        
                    # If there is no such thing as a mode, but there is a min, and a frame is longer than EXPTIME_RATIO * min,
                    #    then frame will be treated as "highly likely" to be saturated and unusable.  
                    elif exptimeMode == 0 and HDUList[iiiIndex].header['EXPTIME'] > ( exptimeMin * EXPTIME_RATIO ) :
                        
                        if VL >= 3 : prntf("stis",20," ",tvr," ",filename, " ", iiiIndex, "has exposure time ", HDUList[iiiIndex].header['EXPTIME'], " which is not within ", EXPTIME_RATIO, " of the minimum EXPTIME ", exptimeMin)
                        if VL >= 3 : prntf("stis",20," ",tvr," ",filename, " ", iiiIndex, "will be ignored.")
                        continue                   
                    
            
            proxyRadonFlag = radonFlag # For the time being while I hammer out stuff.
            if bypassRCFlag == True :
                if VL >= 3 : prntf("stis",20," ",tvr," ","if bypassRCFlag == True :")
                proxyRadonFlag = False # We will still go through polling para, but the version of radonFlag sent in will be false

            filenameIndexTuple = (filename,iiiIndex)

            stopAppend         = time.time()

            durationAppend     = stopAppend - startAppend
                        
            HDUSCIDataNPArray  = np.array(HDUSCIDatalist)
            
            if approvedFlag == True and tvr == 0 : # Presently, only have an approved filename-frame for TARGET
                if filename not in uniqueFilenamesList : continue
            
            cropToFileMaskDimensionsFlag = True
            cropToFileMaskDimensionsFlag = False
            if cropToFileMaskDimensionsFlag == True : 
                if wcw > udw : wcw = udw # reduce the  Widest Common Width
                if hch > udh : hch = udh # reduce the Highest Common Height
            
            if wcw > frameMaxWidth : wcw = frameMaxWidth

            hchPossible = 1044
            if   PROPAPER == 'WEDGEA0.6' and SIZAXIS2 == 256 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA0.6' and SIZAXIS2 == 256")
            elif PROPAPER == 'WEDGEA0.6' and SIZAXIS2 == 137 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA0.6' and SIZAXIS2 == 137")
            elif PROPAPER == 'WEDGEA0.6' and SUBARRAY == False and SIZAXIS2 == 1044 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA0.6' and SUBARRAY == False and SIZAXIS2 == 1044")
            
            elif PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 80 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 80")                
                yStellarPoint = SIZAXIS2 / 2 + CENTERA2myActual
                hchPossible2 = int ( SIZAXIS2 - abs ( SIZAXIS2 / 2 - yStellarPoint ) * 2 ) 
            elif PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 316 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 316")                
                yStellarPoint = yActual # Could be 1 row too low, going by visual match
                if udh  > int ( yStellarPoint ) * 2 : 
                    udh = int ( yStellarPoint ) * 2
                if hch  > int ( yStellarPoint ) * 2 : 
                    hch = int ( yStellarPoint ) * 2
            elif PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 110 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 110 ")
                hchPossible = int ( SIZAXIS2 - abs ( SIZAXIS2 / 2 - CRPIX2 ) * 2 ) 
            elif PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 427 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 427")
                hchPossible = int ( SIZAXIS2 - abs ( SIZAXIS2 / 2 - CRPIX2 ) * 2 )
            elif PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 512 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SIZAXIS2 == 512")
                if hch >= 512 - 1 : hch = 427 # int ( A10Y * 2 )
            elif PROPAPER == 'WEDGEA1.0' and SUBARRAY == False and SIZAXIS2 == 1044 : # 1024 height
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.0' and SUBARRAY == False and SIZAXIS2 == 1044")
                if hch >= 1024 - 1 : hch = 427 # int ( A10Y * 2 )        
            elif PROPAPER == 'WEDGEA1.8' and SIZAXIS2 == 512 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.8' and SIZAXIS2 == 512")
            elif PROPAPER == 'WEDGEA1.8' and SIZAXIS2 == 1044 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA1.8' and SIZAXIS2 == 1044")
            elif PROPAPER == 'WEDGEA2.0' and SIZAXIS2 == 512 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA2.0' and SIZAXIS2 == 512")
            elif PROPAPER == 'WEDGEA2.0' and SIZAXIS2 == 1044 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA2.0' and SIZAXIS2 == 1044")
            elif PROPAPER == 'WEDGEA2.5' :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA2.5'")
            elif PROPAPER == 'WEDGEA2.8' and SIZAXIS2 == 1044 :
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEA2.8' and SIZAXIS2 == 1044")
            # Need for B10, which I showed Ewan last week
            elif PROPAPER == 'WEDGEB1.8' and SIZAXIS2 == 1044:
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEB1.8' and SIZAXIS2 == 1044")
                xStellarPoint = xActual
            elif PROPAPER == 'WEDGEB2.0' and SIZAXIS2 == 1044:
                if VL >= 3 : prntf("stis",20," ",tvr," ","PROPAPER == 'WEDGEB1.8' and SIZAXIS2 == 1044")
                xStellarPoint = xActual
            
            if hch > hchPossible :
                hch = hchPossible                    

            position = ( xStellarPoint , yStellarPoint )

            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","udh                                 : ", udh)
                prntf("stis",20," ",tvr," ","hch                                 : ", hch)
                prntf("stis",20," ",tvr," ","xStellarPoint                       : ", xStellarPoint)
                prntf("stis",20," ",tvr," ","yStellarPoint                       : ", yStellarPoint)
                prntf("stis",20," ",tvr," ","position                            : ", position)
                prntf("stis",20," ",tvr," ","hch                                 : ", hch, "\n")



            # BUILD TUPLE BRIDGE (I assume that CRPIX1, CRPIX2 will not change over course of one file.)
            # BUILD TUPLE BRIDGE (And if they ever do, then pull that discovery under this while loop...)
            if VL >= 3 : prntf("stis",20," ",tvr," ","len ( HDUList )                     : ", len ( HDUList ))
            if VL >= 3 : prntf("stis",20," ",tvr," ","len ( HDUList ) - 1                 : ", len ( HDUList ) - 1)

            # First, set the default
            iiiIndexMax = len ( HDUList ) - 1 # default
            
            # Second, attempt to comply with User request for number of testing frames
            if userTestingNumberFrames != 0 :  # User testing
                iiiIndexMax = userTestingNumberFrames * 3
            
            # Third, if User requested testing frames is more than the file can offer, revert back to default 
            if iiiIndexMax > len ( HDUList ) - 1 :
                iiiIndexMax = len ( HDUList ) - 1 # revert back to default

            iterableIndicesPoll = range ( 1, iiiIndexMax, 3 ) # start, stop, stepsize ; rename iterableIndicesPoll to SCIindices eventually

            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","iiiIndexMax                        : ", iiiIndexMax)
                prntf("stis",20," ",tvr," ","New Multiprocess Pool")            
                prntf("stis",20," ",tvr," ","iterableIndicesPoll                 : ", iterableIndicesPoll)
                prntf("stis",20," ",tvr," ","len(HDUSCIDataNPArray)              : ", len(HDUSCIDataNPArray))    
                prntf("stis",20," ",tvr," ","allRadonCenterImports               : ", allRadonCenterImports)
                prntf("stis",20," ",tvr," ","SIZAXIS1 (Width)                    : ", SIZAXIS1)
                prntf("stis",20," ",tvr," ","SIZAXIS2 (Height)                   : ", SIZAXIS2)
                prntf("stis",20," ",tvr," ","hch                                 : ", hch)
                prntf("stis",20," ",tvr," ","wcw                                 : ", wcw)
            if VL >= 4 : prntf("stis",20," ",tvr," ","HDUSCIDataNPArray                   : ", HDUSCIDataNPArray)

            startPollingProcessTime = time.time()
            pollingProcessPool = Pool()
            if VL >= 3 : prntf("stis",20," ",tvr," ","pollingProcessPool                  : ", pollingProcessPool)
            VL_last = VL
            # It is possible that allowing the polling function to log will hang your system.
            # Advise to set VL = 0 unless diagnosing.
            # Set VL to 0, here, to keep the polling section from logging.
            VL = 0

            tasks = [
                     pollingProcessPool.apply_async (
                                    parallel_polling ,
                                    args = (
                                                SCI_img             ,
                                                iiiIndexPoll        ,                                           
                                                rcImport            ,
                                                approvedFlag        ,
                                                tvr                 ,  
                                                approvedFrameList   , 
                                                0                   , # VL = 0 
                                                proxyRadonFlag      , 
                                                filename            , 
                                                SIZAXIS1            , # Width
                                                SIZAXIS2            , # Height
                                                xStellarPoint       , # Horizontal 
                                                yStellarPoint       , # Vertical                                            
                                                udw                 , 
                                                udh                 , 
                                                wcw                 , 
                                                hch
                                                )
                                   )
                     for SCI_img, iiiIndexPoll, rcImport in zip ( HDUSCIDataNPArray , iterableIndicesPoll, allRadonCenterImports )
                     
                     ]

            pollingProcessPool.close()
            pollingProcessPool.join()
            endPollingProcessTime = time.time()
            
            allRadonCenterImports.clear()

            VL = VL_last
            

            PollParallelizedResult = [ task.get() for task in tasks ]
            if VL >= 3 : prntf("stis",20," ",tvr," ","tasks : ", tasks)
            if VL >= 3 : prntf("stis",20," ",tvr," ","PollParallelizedResult : ", PollParallelizedResult)
            if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch)
    

            if VL >= 3 : prntf("stis",20," ",tvr," ","individualResult :     filename iiiIndex yStellarPoint position totalApproved totalRejected")
            for individualResult in PollParallelizedResult :
                if VL >= 3 : print()
                if VL >= 3 : prntf("stis",20," ",tvr," ","individualResult : ", individualResult)

                for result in individualResult:
                    if VL >= 3 : prntf("stis",20," ",tvr," ","result : ", result)                
                
                if individualResult [ 0 ] : # If this particular return result is NOT None....                    
                    jjjIndex = individualResult [ 0 ] [ 0 ] [ 1 ]         

                    if VL >= 3 : prntf("stis",20," ",tvr," ","jjjIndex : ", jjjIndex, " ", individualResult[0])

                    STISx_minus_RCx =  individualResult [ 0 ] [ 1 ] [ 0 ] - individualResult [ 0 ] [ 2 ] [ 0 ]  
                    STISx_minus_RCx_accumulator = STISx_minus_RCx_accumulator + STISx_minus_RCx 
                    STISy_minus_RCy =  individualResult [ 0 ] [ 1 ] [ 1 ] - individualResult [ 0 ] [ 2 ] [ 1 ]
                    STISy_minus_RCy_accumulator = STISy_minus_RCy_accumulator + STISy_minus_RCy
                    
                    if VL >= 3 : prntf("stis",20," ",tvr," ","STISx_minus_RCx              : ", STISx_minus_RCx)
                    if VL >= 3 : prntf("stis",20," ",tvr," ","STISy_minus_RCy              : ", STISy_minus_RCy)
                    if VL >= 3 : prntf("stis",20," ",tvr," ","STISx_minus_RCx_accumulator  : ", STISx_minus_RCx_accumulator)
                    if VL >= 3 : prntf("stis",20," ",tvr," ","STISy_minus_RCy_accumulator  : ", STISy_minus_RCy_accumulator, "\n")
                    
                    if proxyRadonFlag == True :
                        if VL >= 3 : prntf("stis",20," ",tvr," ","if proxyRadonFlag == True :")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","individualResult [ 0 ] [ 2 ] [ 0 ] : ", individualResult [ 0 ] [ 2 ] [ 0 ])
                        if VL >= 3 : prntf("stis",20," ",tvr," ","individualResult [ 0 ] [ 2 ] [ 1 ] : ", individualResult [ 0 ] [ 2 ] [ 1 ])

                    elif 'RADCENTX' in  HDUList [ jjjIndex ].header and 'RADCENTY' in  HDUList [ jjjIndex ].header :
                        prntf("stis",20," ",tvr," ","if 'RADCENTX' in  HDUList [ jjjIndex ].header and 'RADCENTY' in  HDUList [ jjjIndex ].header :")
                        RADCENTX   = HDUList [ jjjIndex ].header[ 'RADCENTX' ]
                        RADCENTY   = HDUList [ jjjIndex ].header[ 'RADCENTY' ]
                        if VL >= 3 : prntf("stis",20," ",tvr," ","RADCENTX : [", RADCENTX, "] , RADCENTY : [", RADCENTY, "]")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","Compare fresh values (", individualResult [ 0 ], ") with canned values.")
                    
                    else : 
                        prntf("stis",20," ",tvr," ","There are no RC keywords in header. Therefore we need to add RC keywords to header.")
                        prntf("stis",20," ",tvr," ","This is the newest radonCenter determination for this frame: ", individualResult [ 0 ] [ 2 ] )
                        HDUList [ jjjIndex ].header ['RADCENTX'] = str ( individualResult [ 0 ] [ 2 ] [ 0 ] )
                        HDUList [ jjjIndex ].header ['RADCENTY'] = str ( individualResult [ 0 ] [ 2 ] [ 1 ] )
                        if VL >= 3 : prntf("stis",20," ",tvr," ","HDUList [ jjjIndex ].header: ", HDUList [ jjjIndex ].header )

#                    # RADON CENTER FIXER. DO NOT REMOVE.
#                    # Keep the next five lines commented out always, 
#                    #   unless the radonCenter values in the header need to be updated manually.
#                    # To use this, comment out the preceding section starting with "if 'RADCENTX' in  HDUList..."
#                    prntf("stis",20," ",tvr," ","There are no accurate RC keywords in header. Therefore we need to add RC keywords to header.")
#                    prntf("stis",20," ",tvr," ","This is the newest radonCenter determination for this frame: ", individualResult [ 0 ] [ 2 ] )
#                    HDUList [ jjjIndex ].header ['RADCENTX'] = str ( individualResult [ 0 ] [ 2 ] [ 0 ] )
#                    HDUList [ jjjIndex ].header ['RADCENTY'] = str ( individualResult [ 0 ] [ 2 ] [ 1 ] )
#                    if VL >= 3 : prntf("stis",20," ",tvr," ","HDUList [ jjjIndex ].header: ", HDUList [ jjjIndex ].header )
                    
                    
                    
                    if tvr == 0 and wcw > individualResult [ 3 ] : # individualResult [ 3 ] is width
                        if VL >= 3 : prntf("stis",20," ",tvr," ","if tvr == 0 and wcw > individualResult [ 3 ] :")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","wcw : ", wcw, "individualResult [ 3 ] : ", individualResult [ 3 ])
                        wcw = individualResult [ 3 ]
                    
                    # In the case of a target frame, ratchet down hch to the particular frame that had the very worst radonCentering
                    # The very worst radonCentering result may be a function of saturation, structure, etc....
                    if tvr == 0 and hch > individualResult [ 4 ] : # individualResult [ 4 ] is height
                        if VL >= 3 : prntf("stis",20," ",tvr," ","if hch > individualResult [ 4 ]")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch, "individualResult [ 4 ] : ", individualResult [ 4 ])
                        if VL >= 3 : prntf("stis",20," ",tvr," ","It is to be expected that if this frame's radonCenter-drive hch is radically different from others,")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","  it is going to be the result of a saturated or otherwise visibly unusable frame,")
                        if VL >= 3 : prntf("stis",20," ",tvr," ","     that the stddev process will be able to ascertain and therefore exclude from the Top 90.")
                        hch = individualResult [ 4 ]
                        
                    
                    # In the case of a reference frame, ignore it.
                    # Do not append this reference, as it will shrink the final result.
                    # This tries to prevent every good target frame from being truncated due to a single bad reference frame.
                    if tvr == 1 and hch > individualResult [ 4 ] :
                        if VL >= 0 : prntf("stis",20," ",tvr," ","if tvr == 1 and hch > individualResult [ 4 ] :")                        
                        if VL >= 0 : prntf("stis",20," ",tvr," ","hch : ", hch, " is greater than this frame's radonCenter-driven hch ", individualResult [ 4 ])

                        warnings.warn("reference frame has lower height than target. Final output will truncate to this lower height.")
                        hch = individualResult [ 4 ]
                    if VL >= 3 : prntf("stis",20," ",tvr," ","individualResult [ 0 ] : [ ", individualResult [ 0 ], "]")
                    frameTupleList_trgref [ tvr ] . append ( individualResult [ 0 ] ) # This tuple does not contain accepted, rejected, width or height
                else:
                    if VL >= 3 : prntf("stis",20," ",tvr," ","None")

                if VL >= 3 : prntf("stis",20," ",tvr," ","wcw : ", wcw)
                if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch)

                totalApproved = totalApproved + individualResult[1]
                totalRejected = totalRejected + individualResult[2]
                if VL >= 3 : prntf("stis",20," ",tvr," ","totalApproved : ", totalApproved)
                if VL >= 3 : prntf("stis",20," ",tvr," ","totalRejected : ", totalRejected)
                if VL >= 3 : prntf("stis",20," ",tvr," ","len(frameTupleList_trgref) : ", len(frameTupleList_trgref))

                if VL >= 1 : prntf("stis",20," ",tvr," ",)
            
            averageDeviationx = STISx_minus_RCx_accumulator / len(HDUSCIDataNPArray)
            averageDeviationy = STISy_minus_RCy_accumulator / len(HDUSCIDataNPArray)
            sumOfAverageDeviations = averageDeviationx + averageDeviationy 
            if VL >= 3 : prntf("stis",20," ",tvr,"averageDeviationx      : ", averageDeviationx)
            if VL >= 3 : prntf("stis",20," ",tvr,"averageDeviationy      : ", averageDeviationy)
            if VL >= 3 : prntf("stis",20," ",tvr,"sumOfAverageDeviations : ", sumOfAverageDeviations)
            
            
            
            if VL >= 1 : prntf("stis",20," ",tvr," ",)
            if VL >= 1 : prntf("stis",20," ",tvr," ",)

            if VL >= 1 : prntf("stis",20," ",tvr," ","dnfn : ", dnfn)
            
            # I can toggle this off and on until I am sure that keywords written are being read in and acted upon correctly...
            # I need to figure out when new radonCenter values have been calculated because o fhtem not already being in the header...
            if bypassRCFlag == False :
                if VL >= 1 : prntf("stis",20," ",tvr," ","if bypassRCFlag == False : ; So we are writing to header")
                HDUList.writeto ( dnfn , overwrite = True )

            HDUList.close()
            
            
            if VL >= 3 : prntf("stis",20," ",tvr," ","tvr : " , tvr, ", Approved : ", totalApproved, ", Rejected : ", totalRejected)
        tvr = tvr + 1 # within outer loop
    tvr = -2
    
    if cropToFileMaskDimensionsFlag == True : 
        if hch > udh :
            if VL >= 3 : prntf("stis",20," ",tvr," ","if hch > udh :")
            if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch, ", udh : ", udh)
            hch = udh # further reduce the Highest Common Height
            if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch, ", udh : ", udh)
        if wcw > udw :
            if VL >= 3 : prntf("stis",20," ",tvr," ","if wcw > udw :")        
            wcw = udw # further reduce the  Widest Common Width
    if VL >= 3 : prntf("stis",20," ",tvr," ","wcw                                 : ", wcw)
    if VL >= 3 : prntf("stis",20," ",tvr," ","hch                                 : ", hch)
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref       ) : ", len ( frameTupleList_trgref       ) )
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[0]    ) : ", len ( frameTupleList_trgref[0]    ) )
    if len ( frameTupleList_trgref[0]    ) == 0 :
        raise ValueError("\nIt appears that no TARGET files are making it into the class. \n Check the TARGET folder, prefix and wildcard. \n Look for spelling errors or typos.")
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[0][0] ) : ", len ( frameTupleList_trgref[0][0] ) )

    if useRefandTrgFlag == True :
        if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[1]    ) : ", len ( frameTupleList_trgref[1]    ) )
        if len ( frameTupleList_trgref[1]    ) == 0 :
            raise ValueError("\nIt appears that no REFERENCE files are making it into the class. \n Check the REFERENCE folder(s), prefix(es) and wildcard(s). \n Look for spelling errors or typos. \n Look for POS-TARG adjustments that would alter the wedge position and therefore alter its thickness.")
        if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[1][0] ) : ", len ( frameTupleList_trgref[1][0] ) )
    
    if VL >= 3 : prntf("stis",20," ",tvr," ","wcw                                 : ", wcw)
    if VL >= 3 : prntf("stis",20," ",tvr," ","hch                                 : ", hch)
    if wcw > frameMaxWidth : wcw = frameMaxWidth
    largestCommonSize = ( hch, wcw )
    if VL >= 3 : prntf("stis",20," ",tvr," ","largestCommonSize                   : ", largestCommonSize)

    if sizeMax is None:
        sizeMax = largestCommonSize
        if VL >= 3 : prntf("stis",20," ",tvr," ","sizeMax                             : ", sizeMax)

    if sizeMax is not None:
        hch = sizeMax[0]
        wcw = sizeMax[1]
        largestCommonSize = ( hch, wcw )
        if VL >= 3 : prntf("stis",20," ",tvr," ","wcw                                 : ", wcw)
        if VL >= 3 : prntf("stis",20," ",tvr," ","hch                                 : ", hch)
        if VL >= 3 : prntf("stis",20," ",tvr," ","largestCommonSize                   : ", largestCommonSize)
        if VL >= 3 : prntf("stis",20," ",tvr," ","sizeMax                             : ", sizeMax)


    if VL >= 3 : prntf("stis",20," ",tvr," ","runTrgHalfWedgePixels               : ", runTrgHalfWedgePixels)
    if VL >= 3 : prntf("stis",20," ",tvr," ","( dynMask4ple[1] / PLATESCALE ) / 2 : ", ( dynMask4ple[1] / PLATESCALE ) / 2)
    if VL >= 3 : prntf("stis",20," ",tvr," ","dynMask4ple[2]                      : ", dynMask4ple[2])

    IWACalculation = 11 # default for A10
    IWACalculation = math.ceil ( int ( ( dynMask4ple [ 1 ] / PLATESCALE ) / 2 ) + dynMask4ple [ 2 ] )
    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","IWACalculation, initial             : ", IWACalculation)
        prntf("stis",20," ",tvr," ","dynMask4ple[0]                      : ", dynMask4ple[0])
        prntf("stis",20," ",tvr," ","dynMask4ple[1]                      : ", dynMask4ple[1])
        prntf("stis",20," ",tvr," ","dynMask4ple[2]                      : ", dynMask4ple[2])
        prntf("stis",20," ",tvr," ","dynMask4ple[3]                      : ", dynMask4ple[3])
        prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist       : ", differentTrgHeaderWedgesExist)
        prntf("stis",20," ",tvr," ","dynMask4ple                         : ", dynMask4ple)
        prntf("stis",20," ",tvr," ","IWACalculation                      : ", IWACalculation)
        prntf("stis",20," ",tvr," ","runTrgHalfWedgePixels               : ", runTrgHalfWedgePixels)
        prntf("stis",20," ",tvr," ","runBHalfWedgeDiffX1X0Pixels         : ", runBHalfWedgeDiffX1X0Pixels)

    if differentTrgHeaderWedgesExist :
        if VL >= 0 : prntf("stis",20," ",tvr," ","differentTrgHeaderWedgesExist = True")
        if VL >= 0 : prntf("stis",20," ",tvr," ","dynMask4ple will be overridden to True.")
        
        if runTrgHalfWedgePixels > ( ( dynMask4ple[1] / PLATESCALE ) / 2 + dynMask4ple[2] ) :
            if VL >= 3 : prntf("stis",20," ",tvr," ","if runTrgHalfWedgePixels > ( ( dynMask4ple[1] / PLATESCALE ) / 2 + dynMask4ple[2] ) :")
            
            # Don't remove this explainer of the dynamic mask tuple 
            # dynMask4ple = ( dynMask4ple[0], dynMask4ple[1], dynMask4ple[2], dynMask4ple[3] ) # Retain this
            # dynMask4ple[0] the flag is set to True
            # dynMask4ple[1] the wedge width in arcseconds is set to 0, because the next term will do the work...
            # dynMask4ple[2] the 'add pixels to wedge width' is now set to the largest discovered wedge halfwidth
            # dynMask4ple[3] the 'add pixels to spider legs' stays the same / is carried forward
            if dynMask4ple[0] :
                if VL >= 3 : prntf("stis",20," ",tvr," ","if dynMask4ple[0] == True :")
                dynMask4ple = ( True, dynMask4ple[1] + 0, dynMask4ple[2] + runTrgHalfWedgePixels, dynMask4ple[3] )
            else :
                if VL >= 3 : prntf("stis",20," ",tvr," ","if dynMask4ple[0] == False :")
                dynMask4ple = ( True,                  0,                  runTrgHalfWedgePixels, dynMask4ple[3] )
            # Need the IWA calculation here, and it needs to get communicated to dataset.IWA
            IWACalculation = math.ceil ( int ( ( dynMask4ple [ 1 ] / PLATESCALE ) / 2 ) + dynMask4ple [ 2 ] )
            if VL >= 3 : prntf("stis",20," ",tvr," ","IWACalculation                      : ", IWACalculation)
                
        # Using elif to prevent doubling up on modifications
        elif runBHalfWedgeDiffX1X0Pixels != 0 : # BHalfWedgeDiffX1X0Pixels is measured in y / rows... 
            if VL >= 3 : prntf("stis",20," ",tvr," ","elif runBHalfWedgeDiffX1X0Pixels != 0 :")
            dynMask4ple = ( True, dynMask4ple[1] + 0, dynMask4ple[2] + runBHalfWedgeDiffX1X0Pixels, dynMask4ple[3] )
                
        differentTrgHeaderWedgesExist = False       
    if VL >= 3 : prntf("stis",20," ",tvr," ","IWACalculation                      : ", IWACalculation)            
    if VL >= 3 : prntf("stis",20," ",tvr," ","dynMask4ple                         : ", dynMask4ple)            

    IWACalculation = math.ceil ( int ( ( dynMask4ple [ 1 ] / PLATESCALE ) / 2 ) + dynMask4ple [ 2 ] )
    if VL >= 3 : prntf("stis",20," ",tvr," ","IWACalculation, subsequent          : ", IWACalculation)
    if VL >= 3 : 
        prntf("stis",20," ",tvr," ","dynMask4ple[0]                      : ", dynMask4ple[0])
        prntf("stis",20," ",tvr," ","dynMask4ple[1]                      : ", dynMask4ple[1])
        prntf("stis",20," ",tvr," ","dynMask4ple[2]                      : ", dynMask4ple[2])
        prntf("stis",20," ",tvr," ","dynMask4ple[3]                      : ", dynMask4ple[3])
        prntf("stis",20," ",tvr," ","C A L L   F O R   M A S K   I F   R E Q U E S T E D")
        prntf("stis",20," ",tvr," ","hch : ", hch)
        prntf("stis",20," ",tvr," ","wcw : ", wcw)
    if runWedgeDecided == 'B' : yesApplyMaskFlag = True 
    if yesApplyMaskFlag == True:
        
        # This function returns the highest common height, even for mask 401.
        maskToBeUsed = get_mask_function ( hch, wcw, maskFilename, VL, dynMask4ple, runWedgeDecided )
        if VL >= 3 : prntf("stis",20," ",tvr," ","maskToBeUsed.shape                      : ", maskToBeUsed.shape)
        
        
    else:
        maskToBeUsed = None
    if VL >= 3 : prntf("stis",20," ",tvr," ","M A S K   N O W    R E T U R N E D   I F   R E Q U E S T E D ")
    if VL >= 2 : print()
    
    if yesApplyMaskFlag == True:        
        if  hch > maskToBeUsed.shape[0] : 
            hch = maskToBeUsed.shape[0] # further reduce the Highest Common Height
            if VL >= 1 : prntf("stis",20," ",tvr," ","hch : ", hch)
        if  wcw > maskToBeUsed.shape[1] : 
            wcw = maskToBeUsed.shape[1] # further reduce the  Widest Common Width
            if VL >= 1 : prntf("stis",20," ",tvr," ","wcw : ", wcw)
    if VL >= 3 : prntf("stis",20," ",tvr," ","hch : ", hch)
    if VL >= 3 : prntf("stis",20," ",tvr," ","wcw : ", wcw)
    
    # vvv candidate for lcSize
    largestCommonSize = ( hch, wcw )
    if VL >= 3 : prntf("stis",20," ",tvr," ","largestCommonSize : ", largestCommonSize)

    input_SCI_trgref      = [ np.empty ( ( 0, hch, wcw ) ) , np.empty ( ( 0, hch, wcw ) ) ]
    input_ERR_trgref      = [ np.empty ( ( 0, hch, wcw ) ) , np.empty ( ( 0, hch, wcw ) ) ]
    if VL >= 3 : prntf("stis",20," ",tvr," ","input_SCI_trgref : ", input_SCI_trgref )
    if VL >= 3 : prntf("stis",20," ",tvr," ","input_ERR_trgref : ", input_ERR_trgref )


    if VL >= 1 : 
        for i in range(2) : 
            prntf("stis",20," ",tvr," ","I M P L E M E N T   ---  H C H   A N D   W C W   M U S T    B E   F I N A L")
            
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref       ) : ", len ( frameTupleList_trgref       ) )
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[0]    ) : ", len ( frameTupleList_trgref[0]    ) )
    if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[0][0] ) : ", len ( frameTupleList_trgref[0][0] ) )
    if useRefandTrgFlag == True :
        if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[1]    ) : ", len ( frameTupleList_trgref[1]    ) )
        if VL >= 3 : prntf("stis",20," ",tvr," ","len ( frameTupleList_trgref[1][0] ) : ", len ( frameTupleList_trgref[1][0] ) )
    if VL >= 3 : prntf("stis",20," ",tvr," ","tvr : ", tvr)
    
    tvrList = []
    tvr = 0

    
    # Put uniquePixelsIngested above and ahead of all filelists that will come through, which is BOTH target filelist and references filelist
    uniquePixelsIngested  = 0
    uniquePixelsProcessed = 0
    uniqueFilesProcessed  = 0
    uniqueFramesProcessed = 0
    Cutout2DSize = ( hch, wcw )
    
    # THIS IS PROBABLY BEING ALLOCATED IN THE WRONG PLACE AND WITH THE WRONG VALUES
    implFilelistSCIFrameList = [ [None] * len ( frameTupleList_trgref[0] ) , [None] * len ( frameTupleList_trgref[1] ) ]
    implFilelistERRFrameList = [ [None] * len ( frameTupleList_trgref[0] ) , [None] * len ( frameTupleList_trgref[1] ) ]
                                     
    for filelist in userTrgListRefList :
        hhhIndexMax = len ( frameTupleList_trgref [ tvr ] ) - 1
        if VL >= 3 : prntf("stis",20," ",tvr," ","hhhIndexMax : ", hhhIndexMax)
        
        if useRefandTrgFlag == False and filelist == userTrgListRefList[1] :
            continue # Don't do this run if doing the second and third call to pyklip or parallelized, which only want target
        if VL >= 2 : print()
        if VL >= 2 : print()
        if VL >= 1 : 
            for i in range(2) : prntf("stis",20," ",tvr," ","tvr implementation loop tvr implementation loop tvr implementation loop : ", tvr)
        if VL >= 3 : 
            for i in range(8) : prntf("stis",20," ",tvr," ","tvr implementation loop tvr implementation loop tvr implementation loop : ", tvr)
        
        totalApproved   = 0
        totalRejected   = 0
        hhhIndex        = 0
        aaaIndex = 0

        implRecordList = [None] * len ( frameTupleList_trgref[tvr] ) 
        if VL >= 3 : prntf("stis",20," ",tvr," ", "tvr : ",tvr, ", len ( implRecordList ) : ", len ( implRecordList ) )

        for dnfn in filelist :

            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","dnfn : ", dnfn)
                prntf("stis",20," ",tvr," ","This is the first line in the _for dnfn in filelist :_")
                prntf("stis",20," ",tvr," ","What we have so far in the can :")
                prntf("stis",20," ",tvr," ","input_SCI_trgref[0].shape        : ", input_SCI_trgref[0].shape)
                prntf("stis",20," ",tvr," ","input_SCI_trgref[1].shape        : ", input_SCI_trgref[1].shape)
                prntf("stis",20," ",tvr," ","input_ERR_trgref[0].shape        : ", input_ERR_trgref[0].shape)
                prntf("stis",20," ",tvr," ","input_ERR_trgref[1].shape        : ", input_ERR_trgref[1].shape)
            
            
            if VL >= 1 : prntf("stis",20," ",tvr," ","dnfn                             : ", dnfn)
            HDUList          = fits.open ( dnfn )
            
            # First, set the default
            iiiIndexMax = len ( HDUList ) - 1 # default
            if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax)
            
            # Second, attempt to comply with User request for number of testing frames
            if userTestingNumberFrames != 0 :  # User testing
                iiiIndexMax = userTestingNumberFrames * 3
                if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax)
            
            # Third, if User requested testing frames is more than the file can offer, revert back to default 
            if iiiIndexMax > len ( HDUList ) - 1 :
                iiiIndexMax = len ( HDUList ) - 1 # revert back to default
                if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                      : ", iiiIndexMax)

            filename                 = os.path.basename ( dnfn )
            if VL >= 3 : prntf("stis",20," ",tvr," ","filename : ", filename)
            if approvedFlag == True and tvr == 0 : # Presently, only have an approved filename-frame for TARGET
                if filename not in uniqueFilenamesList : continue
            
            CCDGAIN           = HDUList[0].header[ 'CCDGAIN'  ]
            if VL >= 3 : prntf("stis",20," ",tvr," ","CCDGAIN                          : ", CCDGAIN)

            if VL >= 3 : print()
            if VL >= 3 : print()
            if VL >= 3 : print()
            for iiiIndex in range (1, iiiIndexMax + 1, 3 ): # in range syntax will stop at one less than last value
                
                tupleFilenameIiiIndex4 = (filename,iiiIndex)
                if VL >= 3 : prntf("stis",20," ",tvr," ","candidate tupleFilenameIiiIndex4 : ", tupleFilenameIiiIndex4)
                if VL >= 4 : prntf("stis",20," ",tvr," ","iiiIndex : ", iiiIndex, ", ( iiiIndex - 1 ) % 3 : ", ( iiiIndex - 1 ) % 3)
                if VL >= 4 : prntf("stis",20," ",tvr," ","iiiIndex : ", iiiIndex, ", aaaIndex : ", aaaIndex, "len(frameTupleList_trgref [ tvr ]) : ", len(frameTupleList_trgref [ tvr ])  )
                
                if ( iiiIndex - 1 ) % 3 == 0 and aaaIndex < len(frameTupleList_trgref[tvr]) and tupleFilenameIiiIndex4 in frameTupleList_trgref [ tvr ][ aaaIndex ] :
                    
                    if VL >= 3 : prntf("stis",20," ",tvr," ","HDUList[iiiIndex].header['EXTNAME']     : ", HDUList[iiiIndex].header['EXTNAME'])
                    if VL >= 3 : prntf("stis",20," ",tvr," ","frameTupleList_trgref [ tvr ][aaaIndex] : [", frameTupleList_trgref [ tvr ][aaaIndex], "]")
                    STISfindingImpl = frameTupleList_trgref [ tvr ][ aaaIndex ][ 1 ] # This is the radonCenter
                    if VL >= 3 : prntf("stis",20," ",tvr," ","STISfindingImpl            : ", STISfindingImpl, "\n")
                    RCfindingImpl   = frameTupleList_trgref [ tvr ][ aaaIndex ][ 2 ] # This is the radonCenter
                    if VL >= 3 : prntf("stis",20," ",tvr," ","RCfindingImpl              : ", RCfindingImpl, "\n")
                    wcsIndexedImpl                                  = wcs.WCS(HDUList[iiiIndex].header) # Exists only in the ImageHDU, not the PrimaryHDU
                    if VL >= 4 : prntf("stis",20," ",tvr," ","wcsIndexedImpl             : \n", wcsIndexedImpl)
                    

                    implRecord = ( 
                                    np.array ( HDUList[iiiIndex + 0].data ) ,
                                    np.array ( HDUList[iiiIndex + 1].data ) ,
                                    np.array ( HDUList[iiiIndex + 2].data ) ,
                                  RCfindingImpl                           , 
#                                    STISfindingImpl                           , 
                                    wcsIndexedImpl                          ,
                                    CCDGAIN
                                  ) 
                    if VL >= 4 : prntf("stis",20," ",tvr," ","implRecord                 : \n {{{", implRecord, "}}}\n")
                    if VL >= 3 : prntf("stis",20," ",tvr," ","ACCEPTED tupleFilenameIiiIndex4 : ", tupleFilenameIiiIndex4)
                    implRecordList [ aaaIndex ] = implRecord

                    
                    SIZAXIS1          = HDUList [ 0 ].header[ 'SIZAXIS1' ]               # / READOUT DEFINITION PARAMETERS
                    SIZAXIS2          = HDUList [ 0 ].header[ 'SIZAXIS2' ]               # / READOUT DEFINITION PARAMETERS
                    if VL >= 3 : prntf("stis",20," ",tvr," ","SIZAXIS1                   : ", SIZAXIS1)
                    if VL >= 3 : prntf("stis",20," ",tvr," ","SIZAXIS2                   : ", SIZAXIS2)
                    
                    uniquePixelsIngested  = uniquePixelsIngested + SIZAXIS1 * SIZAXIS2 # native width * native height
                    if VL >= 3 : prntf("stis",20," ",tvr," ","uniquePixelsIngested       : ", uniquePixelsIngested)
                    uniquePixelsProcessed = uniquePixelsProcessed + wcw * hch         # widest common width * highest common height
                    if VL >= 3 : prntf("stis",20," ",tvr," ","uniquePixelsProcessed      : ", uniquePixelsProcessed)
                    uniqueFramesProcessed = uniqueFramesProcessed + 1
                    if VL >= 3 : prntf("stis",20," ",tvr," ","uniqueFramesProcessed      : ", uniqueFramesProcessed)
                    
                    aaaIndex = aaaIndex + 1 # this is the only way to increment this index. This index lasts as long as this filelist. This index should reach the end of frameTupleList_trgref before or by the time that the end of filelist is reached.
                else :
                    if VL >= 3 : prntf("stis",20," ",tvr," ","REJECTED  tupleFilenameIiiIndex4 : ", tupleFilenameIiiIndex4)

                
                if VL >= 3 : prntf("stis",20," ",tvr," ","len ( implRecordList )         : ", len ( implRecordList ) )
            
            uniqueFilesProcessed = uniqueFilesProcessed + 1
            if VL >= 3 : prntf("stis",20," ",tvr," ","uniqueFilesProcessed           : ", uniqueFilesProcessed)
            directoryPath            = os.path.dirname ( dnfn )
            split                    = os.path.splitext ( os.path.basename ( dnfn ) )
            token_                   = split[0].split('_')
            rootname                 = token_[0]
            rootnameList.append(rootname)
            HDUList                  = fits.open ( dnfn )
            sciencePixelCount        = 0
            highValuePixelCount      = 0
            
            # get gain here.
            SUBARRAY          = HDUList[0].header[ 'SUBARRAY'  ]
            SIZAXIS1          = HDUList[0].header[ 'SIZAXIS1' ]
            SIZAXIS2          = HDUList[0].header[ 'SIZAXIS2' ]        

            if VL >= 3 : prntf("stis",20," ",tvr," ","len ( HDUList )                : ", len ( HDUList ))
            if VL >= 3 : prntf("stis",20," ",tvr," ","len ( HDUList ) - 1            : ", len ( HDUList ) - 1)
            
            # First, set the default
            iiiIndexMax = len ( HDUList ) - 1 # default
            
            # Second, attempt to comply with User request for number of testing frames
            if userTestingNumberFrames != 0 :  # User testing
                iiiIndexMax = userTestingNumberFrames * 3
            
            # Third, if User requested testing frames is more than the file can offer, revert back to default 
            if iiiIndexMax > len ( HDUList ) - 1 :
                iiiIndexMax = len ( HDUList ) - 1 # revert back to default
            
            if VL >= 3 : prntf("stis",20," ",tvr," ","iiiIndexMax                    : ", iiiIndexMax)
            
            iiiIndex                 = 1                
            
            if VL >= 3 : 
                prntf("stis",20," ",tvr," ","iiiIndex : ", iiiIndex, "iiiIndexMax : ", iiiIndexMax) # All frames on offer index
                prntf("stis",20," ",tvr," ","hhhIndex : ", hhhIndex, "hhhIndexMax : ", hhhIndexMax) # Approved frames index
                prntf("stis",20," ",tvr," ","This is the last line before the _while iiiIndex <= iiiIndexMax and hhhIndex <= hhhIndexMax :_")
                prntf("stis",20," ",tvr," ","bef input_SCI_trgref [",tvr,"].shape        : ", input_SCI_trgref [tvr].shape)
                prntf("stis",20," ",tvr," ","aft input_SCI_trgref [",tvr,"].shape        : ", input_SCI_trgref [tvr].shape)
                prntf("stis",20," ",tvr," ","bef input_ERR_trgref [",tvr,"].shape        : ", input_ERR_trgref [tvr].shape)
                prntf("stis",20," ",tvr," ","aft input_ERR_trgref [",tvr,"].shape        : ", input_ERR_trgref [tvr].shape)
                prntf("stis",20," ",tvr," ","input_SCI_trgref[0].shape               : ", input_SCI_trgref[0].shape)
                prntf("stis",20," ",tvr," ","input_SCI_trgref[1].shape               : ", input_SCI_trgref[1].shape)
                prntf("stis",20," ",tvr," ","input_ERR_trgref[0].shape               : ", input_ERR_trgref[0].shape)
                prntf("stis",20," ",tvr," ","input_ERR_trgref[1].shape               : ", input_ERR_trgref[1].shape)
                prntf("stis",20," ",tvr," ","This is the last line in the _for dnfn in filelist :_")
                
        if VL >= 3 : 
            prntf("stis",20," ",tvr," ","frameTupleList_trgref    : ", frameTupleList_trgref )
            prntf("stis",20," ",tvr," ","len(implRecordList)      : ", len ( implRecordList ) )
            prntf("stis",20," ",tvr," ","implRecordList[0][3]cent : ", implRecordList[0][3] )        
            prntf("stis",20," ",tvr," ","maskToBeUsed             : \n{{{", maskToBeUsed, "}}}\n" )
            prntf("stis",20," ",tvr," ","Cutout2DSize             : ", Cutout2DSize )
            prntf("stis",20," ",tvr," ","VL                       : ", VL )
            prntf("stis",20," ",tvr," ","DQmax                    : ", DQmax )
            prntf("stis",20," ",tvr," ","yesApplyMaskFlag         : ", yesApplyMaskFlag )
            prntf("stis",20," ",tvr," ","divertMaskFlag           : ", divertMaskFlag )
        if VL >= 4 : prntf("stis",20," ",tvr," ","implRecordList      : ", implRecordList )

        implementationProcessPool = Pool ( cpuCount - 1 )

        iterableTimerIndicesImpl = range ( len ( implRecordList ) ) # first timerIndex = 0 for 0 seconds wait, then 1 sec wait, etc.
        
        tasks = [
                     implementationProcessPool.apply_async (
                                                            parallel_implementation , 
                                                            args = (
                                                                    timerIndex        ,
                                                                    implRecord        , 
                                                                    VL                ,
                                                                    DQmax             ,
                                                                    yesApplyMaskFlag  ,
                                                                    divertMaskFlag    ,
                                                                    maskToBeUsed      ,
                                                                    Cutout2DSize
                                                                    )
                                                            )
                     for timerIndex, implRecord in zip ( iterableTimerIndicesImpl, implRecordList )
                     ]    
        implementationProcessPool.close()
        implementationProcessPool.join()
        ImplParallelizedResult = [ task.get() for task in tasks ]
        if VL >= 3 : prntf("post",20,"len(ImplParallelizedResult) : ", len(ImplParallelizedResult))
        if VL >= 4 : prntf("post",20,"ImplParallelizedResult      : ", ImplParallelizedResult)
        if VL >= 3 : prntf("post",20,"individualResult            : app_SCI_img, app_ERR_img")        
        
        returnCounter = 0
        for individualResult2 in ImplParallelizedResult :
            if VL >= 4 : prntf("post",20,returnCounter,"     individualResult2          : \n{{{",      individualResult2   ,"}}}\n")
            if VL >= 4 : prntf("post",20,returnCounter,"     individualResult2[0]       : \n{{{",      individualResult2[0],"}}}\n") # SCI
            if VL >= 4 : prntf("post",20,returnCounter,"     individualResult2[1]       : \n{{{",      individualResult2[1],"}}}\n") # ERR
            returnCounter = returnCounter + 1

        implFilelistSCIFrameList [ tvr ] = [ col [0]    for col in ImplParallelizedResult ]         
        implFilelistERRFrameList [ tvr ] = [ col [1]    for col in ImplParallelizedResult ]
        input_filenames_trgref2  [ tvr ] = [ col [0][0] for col in frameTupleList_trgref [ tvr ] ]
        input_centers_trgref2    [ tvr ] = [ col [2]    for col in ImplParallelizedResult ]
        pas_trgref2[tvr]                 =       [ None for col in implRecordList ]
        wcsBlockList                     = [ col [4]    for col in implRecordList ]
        wwwIndex                         = 0
        for wcsBlock in wcsBlockList :
            rot_angle                        = np.rad2deg ( math.atan2 ( wcsBlock.wcs.cd[1][0], wcsBlock.wcs.cd[0][0] ) )
            wcsOrientat                      = 180 * np.sign ( rot_angle ) - rot_angle
            pas_trgref2 [ tvr ] [ wwwIndex ] = 180 * np.sign ( rot_angle ) - rot_angle
            wwwIndex                         = wwwIndex + 1

        for all in frameTupleList_trgref [ tvr ] :
            tvrList.append ( tvr )
        tvr = tvr + 1 # within outer loop

        
        implRecordList.clear()
    tvr = -3 
    frameTupleList_trgref.clear()
    
    if VL >= 3 : prntf("stis",20," ",tvr," ","COMPLETED : for filelist in userTrgListRefList " )
   
    
    
    # FQPN loop has concluded
    rollAngleUniqueList     = list ( set ( pas_trgref2 [ 0 ] ) ) # IS there a way to make this not print?
    
    rollAngleSum            = sum ( rollAngleUniqueList, 0 )
    
    rollAngleAverage        = rollAngleSum / len ( rollAngleUniqueList )
    
    if useRefandTrgFlag    == True:    
        input_SCI_0_nparray = np.vstack        ( implFilelistSCIFrameList[0] )
        input_SCI_1_nparray = np.vstack        ( implFilelistSCIFrameList[1] ) 
        input_SCI_0_test    = np.concatenate   ( implFilelistSCIFrameList[0] )
        input_SCI           = np.concatenate ( ( implFilelistSCIFrameList[0] , implFilelistSCIFrameList[1] ) )
        input_ERR           = np.concatenate ( ( implFilelistERRFrameList[0] , implFilelistERRFrameList[1] ) )
        input_centers       = np.concatenate ( ( input_centers_trgref2[0]    , input_centers_trgref2[1]    ) )
        pas                 = np.concatenate ( ( pas_trgref2[0]              , pas_trgref2[1]              ) )
        input_filenames     = np.concatenate ( ( input_filenames_trgref2[0]  , input_filenames_trgref2[1]  ) )
        RADECList           = np.concatenate ( ( RADEC_trgref[0]             , RADEC_trgref[1]             ) )
    
    else:            
        input_SCI           = implFilelistSCIFrameList[0]
        input_ERR           = implFilelistERRFrameList[0]
        input_centers       = input_centers_trgref2[0]
        pas                 = np.array ( pas_trgref2[0] )
        input_filenames     = input_filenames_trgref2[0]
        RADECList           = RADEC_trgref[0]

    rollAngleSetReturn = pas_trgref2[0]      

    input_centers_trgref2.clear()

    pas_trgref2.clear()

    input_filenames_trgref2.clear()

    return input_SCI, input_centers, pas, input_filenames, rollAngleSetReturn, rollAngleAverage, RADECList, HDUList[0].header[ 'RA_TARG'  ], HDUList[0].header[ 'DEC_TARG' ], rootnameList, sizeMax, tvrList, input_ERR, maskToBeUsed, approvedFrameList, uniquePixelsIngested, uniquePixelsProcessed, uniqueFramesProcessed, uniqueFilesProcessed, IWACalculation





class STISData ( Data ):
    
    # Constructor
    def __init__    (
                     self                                    ,
                     trgSCIListrefSCIList                    ,
                     outputFolder                            ,                     
                     wvs                     = None          ,
                     xExtent                 = 1024          , # STIS native width
                     yExtent                 = 1024          , # STIS native height
                     IWA                     = 0             ,
                     OWA                     = 724.1         , # STIS native radial hypotenuse
                     flipx                   = False         ,
                     yesApplyMaskFlag        = False         ,
                     divertMaskFlag          = False         ,
                     maskFilename            = None          ,
                     DatasetPlotFlag         = False         ,
                     DQmax                   = 512           ,
                     VL                      = 0             , # Verbosity Level
                     approvedFlag            = False         , # False is conservative
                     radonFlag               = False         , # False is conservative
                     userTestingNumberFrames = 0             , 
                     dynMask4ple             = (False,0,0,0) ,
                     runWedgeDecided         = 'A'           ,
                     useRefandTrgFlag        = False         ,
                     sizeMax                 = None          ,
                     fileMaskWedgeCode       = 'A10'
                     
                     ):
        """
            Args    :
                self                            : object.   Instrument class member ".dataset"
                trgSCIListrefSCIList            : tuple.    [ TARGET list, REFERENCE list ]
                wvs              = None         : (number)  wavelengths. Not used now for STIS. (could remove)
                xExtent          = 1024         : Int.      STIS image's native width
                yExtent          = 1024         : Int.      STIS image's native height
                IWA              = 0            : Float.    Inner Working Angle - usedby pyklip 
                OWA              = 724.1        : Float.    Outer Working Angel . Use the STIS image's native radial hypotenuse.
                flipx            = False        : Bool.     For flipping x. Not used. (could remove) 
                yesApplyMaskFlag = False        : Bool.     True means: Yes, import the mask and create mask array.
                divertMaskFlag   = False        : Bool.     True means: Yes, divert mask array to the function return
                maskFilename     = None         : String.   File containing mask.
                useRefandTrgFlag = False        : Bool.     True means: Yes, use both target and reference frames during this while loop.
                DQmax            = 512          : Int.      Maximum value for a pixel in the Data Quality STIS frame.  
                sizeMax          = None         : Int.      Diagnostic for development.
                VL               = 0            : Int.      Verbosity Level
                approvedFlag     = False        : Bool.     True means: Yes, import the approvedFrames file
                radonFlag        = False        : Bool.     True means: Yes, perform radonCenter on all frames
                
            Attributes:
                input                           : Array of shape (N,y,x) for N images of shape (y,x). STIS SCI frames. 
                error                           : Array of shape (N,y,x) for N images of shape (y,x). STIS ERR frames.
                centers                         : Array of shape (N,2) for N input centers in the format [x_cent, y_cent]
                PAs                             : Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
                filenames                       : Array of size N for the actual filepath of the file that corresponds to the data
                filenums                        : Array of size N for the numerical index to map data to file that was passed in
                wvs                             : Array of N wavelengths of the images (used for SDI) [in microns].
                                                  For polarization data, defaults to "None"
                xExtent                         : User Determined Width
                yExtent                         : User Determined Height
                IWA                             : a floating point scalar (not array). Specifies to inner working angle in pixels
                OWA                             : (optional) specifies outer working angle in pixels
                flipx                           : (optional) False by default. Determines whether a reflection about the x axis 
                                                  is necessary to rotate image North-up East left
                wcs                             : Array of N wcs astrometry headers for each input image.
                output                          : Array of shape (b, len(files), len(uniq_wvs), y, x) 
                                                  where b is the number of different KL basis cutoffs
                rootnameList                    : Diagnostic
                sizeMax                         : Diagnostic
                tvrList                         : list of 0, 1. Critical discriminator between TARGET and REFERENCE 
                maskToBeUsed                    : numpy array. mask.
                approvedFrameList               : List of tuples ( filename, ImageHDU index, frame stddev )
                
                output_centers                  : Array of shape (N,2) for N output centers. 
                                                  Also coresponds to FM centers (does not need to be implemented)
                output_wcs                      : Array of N wcs astrometry headers for each output image (does not need to be implemneted)
                klipparams                      : (optional) a string that saves the most recent KLIP parameters
                
            Methods:
                prntf("stis",20,)                           : Diagnostic prints and User reporting prints
                get_file_mask ()            : Get the native size values from the unprocessed STIS frame of the mask
                get_mask_function ()              : Get the Highest Common Height values of the effective aperture of the mask
                primary_function ()     : Get the data requested by pyklip. Also get some other data for NMF.
                get_output_framesXPR ()              : Take in the products of pyklip or NMF and perform statistics, sort it, and produce FITS.
                savedata()                      : Pyklip uses this to save data to the .dataset member.
                sanity()                        : Diagnostic to print plots
        """
        super ( STISData, self ).__init__()
        
        # read in STIS configuration file and set these static variables
        package_directory = os.path.dirname ( os.path.abspath ( __file__ ) )
        configfile = package_directory + "/" + "STIS.ini"
        config = ConfigParser.ConfigParser()
        try:
            config.read ( configfile )
        except ConfigParser.Error as e :
            print("Error reading STIS configuration file: {0}".format(e.messsage))
            raise e

        input_SCI2, centers2, pas2, input_filenames2, self.rollAngleSet, self.rollAngleAverage, RADECList2, RA_TARG, DEC_TARG, rootnameList, returnSizeMax, tvrList, input_ERR2, maskToBeUsed2, approvedFrameList2, uniquePixelsIngested2, uniquePixelsProcessed2, uniqueFramesProcessed2, uniqueFilesProcessed2, runTrgHalfWedgePixels2 = primary_function ( config, xExtent, yExtent, trgSCIListrefSCIList, outputFolder, yesApplyMaskFlag, divertMaskFlag, maskFilename, fileMaskWedgeCode, DatasetPlotFlag, useRefandTrgFlag, DQmax, sizeMax, VL, approvedFlag, radonFlag, userTestingNumberFrames, dynMask4ple, runWedgeDecided )


        self.config = config
        
        self._input              = np.array ( input_SCI2 )
        self._error              = np.array ( input_ERR2 )
        nfiles                   = self.input.shape[0]
        
        self.centers             = np.array ( centers2 )
        if self.centers.shape [0] != nfiles:
            raise ValueError ( "Input data has shape {0} but centers has shape {1}".format ( self.input.shape, self.centers.shape ) )

        self._PAs                = pas2
        self._filenames          = input_filenames2
        
        unique_filenames         = np.unique ( input_filenames2 )
        
        self._filenums           = np.array ( [ np.argwhere ( filename == unique_filenames ) .ravel()[0] for filename in input_filenames2 ] )
        
        if wvs is not None:
            self._wvs            = wvs
        else:
            self._wvs            = np.ones ( nfiles )
        
        self.xExtent             = xExtent
        
        self.yExtent             = yExtent
        
        self.IWA                 = IWA
        self.IWA                 = runTrgHalfWedgePixels2

        self.OWA                 = OWA  # this is a private/init member of the parent class, Data.
                                    # along with creator, klipparams, flipx, output_centers, output_wcs
                                    # try to keep in mind that it is the parent class' variable you are changing
        
        self.flipx               = flipx

        self._wcs = [ None ] * len ( self._PAs )        
        for iiiIndex in range ( len ( self._PAs ) ) :
            self._wcs [iiiIndex] = wcsgen.generate_wcs ( self._PAs[iiiIndex], self.centers[iiiIndex], flipx = flipx )

        
        for i in range ( len ( self.wcs ) ):
            self.wcs[i].wcs.crval[0] = RADECList2[i][0]
            self.wcs[i].wcs.crval[1] = RADECList2[i][1]
        
        self._output               = None
        self.rootnameList          = rootnameList
        self.sizeMax               = returnSizeMax
        self.tvrList               = tvrList
        self.maskToBeUsed              = maskToBeUsed2
        self.approvedFrameList     = approvedFrameList2
        self.uniquePixelsIngested  = uniquePixelsIngested2 
        self.uniquePixelsProcessed = uniquePixelsProcessed2 
        self.uniqueFramesProcessed = uniqueFramesProcessed2
        self.uniqueFilesProcessed  = uniqueFilesProcessed2

        self.runTrgHalfWedgePixels = runTrgHalfWedgePixels2 



    ################################
    ### Instance Required Fields ###
    ################################

    @property
    def input ( self ):
        return self._input
    @input.setter
    def input ( self, newval ):
        self._input = newval

    @property
    def centers ( self ):
        return self._centers
    @centers.setter
    def centers ( self, newval ):
        self._centers = newval

    @property
    def filenums ( self ):
        return self._filenums
    @filenums.setter
    def filenums ( self, newval ):
        self._filenums = newval

    @property
    def filenames ( self ):
        return self._filenames
    @filenames.setter
    def filenames ( self, newval ):
        self._filenames = newval

    @property
    def PAs ( self ):
        return self._PAs
    @PAs.setter
    def PAs ( self, newval ):
        self._PAs = newval

    @property
    def wvs ( self ):
        return self._wvs
    @wvs.setter
    def wvs ( self, newval ):
        self._wvs = newval

    @property
    def wcs ( self ):
        return self._wcs
    @wcs.setter
    def wcs ( self, newval ):
        self._wcs = newval

    @property
    def IWA ( self ):
        return self._IWA
    @IWA.setter
    def IWA ( self, newval ):
        self._IWA = newval

    @property
    def output ( self ):
        return self._output
    @output.setter
    def output ( self, newval ):
        self._output = newval
    
    def sanity(self, VL, sanityPlotFlag) :        
        selfDatasetIndex = 0
        for frame in self._input :
            filename = self._filenames[selfDatasetIndex]
            rootname = filename.split('_')
            selfDatasetIndex = selfDatasetIndex + 1
            # prints can be added in here if necessary

    def __del__(self):
        prntf("stis",20,"Destructor called.")
    
    



    
    def get_output_frames ( 
                        self                                                     ,
                        approvedFlag                     = False                 , # False is conservative 
                        radonFlag                        = False                 , # False is conservative 
                        percentThreshold                 = 90                    ,
                        stddevSortFlag                   = True                  ,
                        stisSortFlag                     = True                  ,
                        OutFrPlotFlag                    = False                 ,
                        blockApyUserWarn                 = True                  ,
                        zeroOutputFrames_wcsOrientatFlag = False                 ,
                        outputFolder                     = ""                    ,
                        runNumberString                  = ""                    ,
                        incomingArray                    = None                  ,
                        VL                               = 0                     , # Verbosity Level 
                        username                         = "zyx"                 #,  # default
                        ) :        
        """
            Performs a variety of operations on data stored in Instrument class "dataset" member
            Args   :
                self                     : Class member. The Instrument class "dataset" member
                incomingArray    = None  : numpy array.  If     None, Pyklip results accessed from self._output
                                                         If Not None, NMFresults passed in from notebook
                                           TODO: Best practice would be to also send in the Pyklip data
                                                 And not have an internal call to the Class object
                approvedFlag     = False : Bool.         True means: Yes, import the approvedFrames file
                radonFlag        = False : Bool.         True means: Yes, perform radonCenter on all frames
                percentThreshold = 90    : Int.          User determined top percentage of frame stddev.
                stddevSortFlag   = True  : Bool.         True means: Yes, send stddev sort to approved frames. 
                stisSortFlag     = True  : Bool.         True means: Yes, send STIS   sort to approved frames.
                OutFrPlotFlag    = False : Bool.         True means: Yes, plot the frame where possible. DIAG.
                blockApyUserWarn = True  : Bool.         True means: Yes, block the pink warnings.
                VL               = 0     : Int.          Verbosity Level 
                username         = "zyx" : String.       Username
            Return :
                self._output             : pointer.      Placeholder return value. Same thing as what is accessed for pyklip.
                                           TODO: See if there's any rationale to keep or dispose. Leave it for now. 
        """
        if VL >= 3 : 
            prntf("outp",20,"outputFolder                      : ", outputFolder)
            prntf("outp",20,"VL                                : ", VL)
            prntf("outp",20,"OutFrPlotFlag                     : ", OutFrPlotFlag)        
            prntf("outp",20,"approvedFlag                      : ", approvedFlag)
            prntf("outp",20,"radonFlag                         : ", radonFlag)
            prntf("outp",20,"incomingArray                     : ", incomingArray)
            prntf("outp",20,"runNumberString                   : ", runNumberString)
    
        if incomingArray is None :                      
            if VL >= 3 : prntf("outp",20,"Pyklip has already updated dataset._output")
            if VL >= 3 : prntf("outp",20,"self._output  : ", self._output)
            if VL >= 3 : prntf("outp",20,"self._output.shape  : ", self._output.shape)
            squeeze2           = np.squeeze(self._output, axis = 2) # wvs is identical ; these have been stored North Up
            if VL >= 3 : prntf("outp",20,"squeeze2.shape      : ", squeeze2.shape)
            if VL >= 3 : prntf("outp",20,"Select the first mode (or only mode) for the purposes of stddev sorting.")
            internalArray = squeeze2[0]
        else :                                          
            if VL >= 3 : prntf("outp",20,"Output from the NMF program is being passed into get_output_frames")
            if VL >= 3 : prntf("outp",20,"incomingArray.shape               : ", incomingArray.shape) # STIS Up.
            internalArray      = incomingArray
        
        lengthInternalArray       = len (internalArray)

        if VL >= 3 : 
            prntf("outp",20,"internalArray is residual frames that have been stripped of their filename and their ImageHDU index.")
            prntf("outp",20,"internalArray : ", internalArray)
            prntf("outp",20,"lengthInternalArray               : ", lengthInternalArray)
            prntf("outp",20,"STIS dataset member approvedFrameList is in tuples of filename, ImageHDU, and stddev")
            prntf("outp",20,"self.approvedFrameList            : ", self.approvedFrameList)
            prntf("outp",20,"We're going to use astropy statistics module under catch_warnings")
            prntf("outp",20,"in order to suppress the warning about nans (if the User sets the flag).")

        import warnings
        from astropy.utils.exceptions import AstropyUserWarning
        with warnings.catch_warnings() :
            if blockApyUserWarn : 
                warnings.simplefilter("ignore", AstropyUserWarning)
            
            sigma                  = 2
            selfDatasetIndex       = 0
            filename               = self._filenames [ selfDatasetIndex ]
            lastFilename           = filename
            ImageHDUIndex          = 1 
            tuple7List             = []
            tuple7List.clear()        
            
            if VL >= 3 : 
                prntf("outp",20,"Set selfDatasetIndex = 0")
                prntf("outp",20,"The STISData class object 'dataset' has a member '_filenames' holding all one filename for every frame.")
                prntf("outp",20,"The index starts at 0.")
                prntf("outp",20,"Get the very first filename, index = 0, in self._filenames.")
                prntf("outp",20,"filename                          : ", filename, "\n")
                prntf("outp",20,"Set ImageHDUIndex = 1 because STIS ImageHDU indices begin at _1_")
                prntf("outp",20,"unSigmaClippedFrame means frame has not yet been sigma clipped, and still has high value pixels...")
                prntf("outp",20,"...in the form of diffraction spikes, wedge diffraction, PSF, and disk structure.")
                        
            if VL >= 3 : prntf("outp",20,"STARTTTT for unSigmaClippedFrame in internalArray :")
            if VL >= 3 : prntf("outp",20,"len(internalArray)          : ", len(internalArray))
            for unSigmaClippedFrame in internalArray :
                if VL >= 3 : prntf("outp",20,"selfDatasetIndex         : ", selfDatasetIndex)                
                
                if approvedFlag == True :
                    if VL >= 4 : prntf("outp",20,"If we're importing approvedFrameList, get the next 3-tuple from approvedFrameList.")
                    approved3Tuple = self.approvedFrameList [ selfDatasetIndex ]
                    if VL >= 3 : prntf("outp",20,"approved3Tuple           : ", approved3Tuple)

                if VL >= 4 : prntf("outp",20,"Get the next filename from STISData class 'dataset' member 'self._filenames'. ")
                filename       = self._filenames            [ selfDatasetIndex ]
                if VL >= 2 : prntf("outp",20,"filename , ImageHDUIndex : ", filename, " ", ImageHDUIndex, " checking if approved...")
                
                if VL >= 4 : prntf("outp",20,"Get the next wcs keyword set from STISData class 'dataset' member 'self._wcs'. ")
                wcsKeywords    = self._wcs                  [ selfDatasetIndex ]
                if VL >= 4 : prntf("outp",20,"wcsKeywords           : ", wcsKeywords)
                cd11           = wcsKeywords.wcs.cd[0][0]
                if VL >= 4 : prntf("outp",20,"cd11 : ", cd11)
                cd21           = wcsKeywords.wcs.cd[1][0]
                if VL >= 4 : prntf("outp",20,"cd21 : ", cd21)
                
                center         = self.centers               [ selfDatasetIndex ]
                
                if VL >= 4 : prntf("outp",20,"Convert WCS matrix elements to orientation angle.")
                rho_a          = math.atan2    ( cd21, cd11 ) #  math.atan2(y, x)
                rot_angle      = np.rad2deg    ( rho_a )
                wcsOrientat    = 180 * np.sign ( rot_angle ) - rot_angle
                if VL >= 3 : prntf("outp",20,"wcsOrientat             : ", wcsOrientat)

                if VL >= 4 : prntf("outp",20,"Get the next PA from STISData class 'dataset' member 'self._PAs'. Sanity check only.")
                sanityPA  = self._PAs                  [ selfDatasetIndex ]
                if VL >= 3 : prntf("outp",20,"sanityPA                : ", sanityPA)

                if approvedFlag == True :
                    if filename == approved3Tuple [ 0 ] and ImageHDUIndex != approved3Tuple [ 1 ] :
                        if VL >= 2 : prntf("outp",20,"if filename == approved3Tuple [ 0 ] and ImageHDUIndex != approved3Tuple [ 1 ] :")
                        if VL >= 4 : prntf("outp",20,"This means we ended a contiguous block of approved frames while advancing through this filename.") 
                        if VL >= 4 : prntf("outp",20,"Ratchet ImageHDUIndex ahead to wherever the gap resumes with approved frames.")
                        ImageHDUIndex              = approved3Tuple [ 1 ]
                        if VL >= 4 : prntf("outp",20,"ImageHDUIndex            : ", ImageHDUIndex)

                if filename                     != lastFilename :
                    if VL >= 2 : prntf("outp",20,"if filename           != lastFilename")
                    if VL >= 4 : prntf("outp",20,"This means we ended a block of approved frames belonging to one filename.")
                    if VL >= 4 : prntf("outp",20,"Reset ImageHDUIndex bacvk to 1, because we're about to start a new filename.")
                    ImageHDUIndex              = 1
                    if VL >= 4 : prntf("outp",20,"ImageHDUIndex        : ", ImageHDUIndex)
                if VL >= 2 : prntf("outp",20,"filename , ImageHDUIndex : ", filename, " ", ImageHDUIndex, " confirmed approved.")                        

                if VL >= 4 : prntf("outp",20,"Sigma clip the frame to get ONLY the background pixels.")
                if VL >= 4 : prntf("outp",20,"Getting ONLY the background pixels allows getting statistics on background noise level.")
                from astropy import stats

                stisUpSigmaClippedFrame  = stats.sigma_clip        ( unSigmaClippedFrame, sigma = sigma, maxiters = 5 )

                if VL >= 4 : prntf("outp",20,"Get mean, median and stddev on ONLY the background pixels.")

                
                frameMeanMedStddev = stats.sigma_clipped_stats ( stisUpSigmaClippedFrame, sigma = sigma, maxiters = 5)
                

                stisUpSigmaClippedFrameNPArray        = np.array ( stisUpSigmaClippedFrame )
                rotationCenter                  = ( stisUpSigmaClippedFrameNPArray . shape[1] / 2 , stisUpSigmaClippedFrameNPArray . shape[0] / 2 )
                rotationAngle = wcsOrientat
                if zeroOutputFrames_wcsOrientatFlag == True :
                    rotationAngle = 0
                from pyklip.klip import rotate as pyklipRotate
                northUpSigmaClippedFrameNPArray = pyklipRotate ( stisUpSigmaClippedFrameNPArray, rotationAngle, rotationCenter )                

                if VL >= 4 : prntf("outp",20,"tuple7List is a list of tuples and is the main variable that all following processes will access.")
                tuple7List . append (
                                            (
                                                 stisUpSigmaClippedFrame         , # 0 , frame
                                                 filename                        , # 1 , filename
                                                 ImageHDUIndex                   , # 2 , Index
                                                 frameMeanMedStddev[2]           , # 3 , stddev
                                                 wcsOrientat                     , # 4 , wcsOrientat
                                                 wcsKeywords                     , # 5 , the whole block, until I can figure out an efficient way
                                                 center                          , # 6 , until I get a smarter way of getting just one
                                                 northUpSigmaClippedFrameNPArray   # 7
                                             )
                                        )
                selfDatasetIndex       = selfDatasetIndex      + 1                
                ImageHDUIndex          = ImageHDUIndex         + 3                
                lastFilename           = filename 
                if VL >= 4 : 
                    prntf("outp",20,"selfDatasetIndex = selfDatasetIndex + 1, because STIS SCI ImageHDU Index position has been stripped.")
                    prntf("outp",20,"The frames which used to be separated by ERR and DQ now go SCI, SCI, SCI ... in STISData 'dataset'.")
                    prntf("outp",20,"ImageHDUIndex = ImageHDUIndex + 3, because STIS exposures go SCI+ERR+DQ , SCI+ERR+DQ , SCI+ERR+DQ ...")
                    prntf("outp",20,"lastFilename = filename is to ratchet forward the value of lastFilename...")
                    prntf("outp",20,"...for the 'if filename != lastFilename :' comparator")               
                    
            if VL >= 3 : prntf("outp",20,"STOPPPPP for unSigmaClippedFrame in internalArray :\n")


            if VL >= 3 : prntf("outp",20,"   I N C O M I N G  ", lengthInternalArray, "  F R A M E S")
            if VL >= 3 : prntf("outp",20,"   S O R T E D   B Y   O R D E R")
            if VL >= 3 : prntf("outp",20,"filename\t\t\tImageHDU\tstandard deviation\twcsOrientat")
            for record in tuple7List :
                if VL >= 3 : prntf("outp",20,record[1], "\t", record[2], "\t\t", record[3], "\t", record[4] )
            if VL >= 3 : prntf("outp",20," ")


            if VL >= 3 : prntf("outp",20,"Get a copy of    tuple7List that is sorted by order")
            from operator import itemgetter        
            tuple7ListByOrder                   = sorted ( tuple7List, key = itemgetter ( 1 ) ) # col 1 is the filename
            framesListByOrder                   = [ col[0] for col in tuple7ListByOrder ]       # col 0 is the frame
            framesArrayByOrder                  = np.array ( framesListByOrder )

            stisUpNanMedianFrame = np.nanmedian ( framesArrayByOrder, axis = 0 )

            stisUpNanStdFrame = np.nanstd ( framesArrayByOrder, axis = 0 )

            tuple7ListByStddev                  = sorted ( tuple7List, key = itemgetter ( 3 ) )               # col 3 is the stddev
            
            framesListByStddev                  = [ col[0] for col in tuple7ListByStddev ]                    # col 0 is the frame

            framesArrayByStddev                 = np.array ( framesListByStddev )

            northUpFramesListByStddev           = [ col[7] for col in tuple7ListByStddev ]                    # col 7 is the north up frame
            northUpFramesArrayByStddev          = np.array     ( northUpFramesListByStddev )              
            northUpNanMedianFrame               = np.nanmedian ( northUpFramesArrayByStddev, axis = 0 )
            northUpNanStdFrame                  = np.nanstd    ( northUpFramesArrayByStddev, axis = 0 )

            if VL >= 3 : prntf("outp",20,"sorted list, half / median record, record element #5 = angle -->")
            
            
            # ROTATION HAPPENS HERE AND REQUIRES WCS TO BE ACCESSED
            stisUpWcsBlock                      = tuple7ListByStddev  [ 0 ] [ 5 ]                

            nrthUpWcsBlock = wcsgen.generate_wcs ( 0, center, flipx = False )
            nrthUpWcsBlock.wcs.crval[0] = stisUpWcsBlock.wcs.crval[0]
            nrthUpWcsBlock.wcs.crval[1] = stisUpWcsBlock.wcs.crval[1]
            
            outputPath = "outputs/" + username + "/"

            if radonFlag == False : 
                northUpAllFilename              = runNumberString + "nuAllnrc.fits" # "northUpAllByStddevNORC.fits"
                nrthUpNanMedFilename            = runNumberString + "nuMednrc.fits" # "nrthUpNanMedByStddevNORC.fits"
                nrthUpNanStdFilename            = runNumberString + "nuStdnrc.fits" # "nrthUpNanStdByStddevNORC.fits"
            else : 
                northUpAllFilename              = runNumberString + "nuAll.fits" # "northUpAllByStddevYESRC.fits"
                nrthUpNanMedFilename            = runNumberString + "nuMed.fits" # "nrthUpNanMedByStddevYESRC.fits"
                nrthUpNanStdFilename            = runNumberString + "nuStd.fits" # "nrthUpNanStdByStddevYESRC.fits"
            

            if VL >= 3 : prntf("outp",20,"Make a FITS file for ALL North-Up residuals, sorted by stddev")            
            HeaderDataUnit                      = fits.PrimaryHDU ( northUpFramesArrayByStddev    )
            HeaderDataUnitList                  = fits.HDUList    ( [ HeaderDataUnit ]            )
            try :
                HeaderDataUnitList.writeto          ( outputFolder + northUpAllFilename , overwrite = True)
            except :
                if VL >= 3 : 
                    for i in range(2) :
                        prntf("outp",20,"No file written yet! Please ensure that path exists for: [", outputPath + northUpAllFilename , "]")


            HeaderDataUnit                      = fits.PrimaryHDU ( northUpNanMedianFrame         )
            HeaderDataUnitList                  = fits.HDUList    ( [ HeaderDataUnit ]            )
            HeaderDataUnit.header.update        ( nrthUpWcsBlock.to_header() )
            try : 
                HeaderDataUnitList.writeto          ( outputFolder + nrthUpNanMedFilename , overwrite = True)
            except :
                if VL >= 3 : 
                    for i in range(2) :
                        prntf("outp",20,"No file written yet! Please ensure that path exists for: [", outputPath + nrthUpNanMedFilename , "]\n")


            HeaderDataUnit                      = fits.PrimaryHDU ( northUpNanStdFrame            )
            HeaderDataUnitList                  = fits.HDUList    ( [ HeaderDataUnit ]            )
            HeaderDataUnit.header.update        ( nrthUpWcsBlock.to_header() )
            try : 
                HeaderDataUnitList.writeto          ( outputFolder + nrthUpNanStdFilename , overwrite = True)
            except : 
                if VL >= 3 : 
                    for i in range(2) :
                        prntf("outp",20,"No file written yet! Please ensure that path exists for: [", outputPath + nrthUpNanStdFilename , "]\n")


            if VL >= 3 : prntf("outp",20,"   I N C O M I N G  ", lengthInternalArray, "  F R A M E S")
            if VL >= 3 : prntf("outp",20,"   S O R T E D   B Y   S T D D E V")         
            
            lengthTuple7ListByStddev      = len ( tuple7ListByStddev ) 
            for record in tuple7ListByStddev :
                if VL >= 3 : prntf("outp",20,record[1], "\t", record[2], "\t\t", record[3], "\t", record[4] )
            
            if VL >= 3 : prntf("outp",20,"   G E T   M E D I A N   S T D D E V")
            stddevTuple7List              = [ col[3] for col in tuple7ListByStddev ] # col 3 is stddev
            lengthStddevsList             = len ( stddevTuple7List )
            if VL >= 3 : prntf("outp",20,"lengthStddevsList                         : ", lengthStddevsList)
            
            numpyMean      = np.mean   ( stddevTuple7List )
            if VL >= 3 : prntf("outp",20,"numpyMean                                 : ", numpyMean)
            numpyMedian    = np.median ( stddevTuple7List )
            if VL >= 3 : prntf("outp",20,"numpyMedian                               : ", numpyMedian)
            numpyStd       = np.std    ( stddevTuple7List )
            if VL >= 3 : prntf("outp",20,"numpyStd                                  : ", numpyStd)
            
            if VL >= 3 : prntf("outp",20,"Median stddev of all ", lengthStddevsList, " frames         : ", numpyMedian, "\n" )

            if VL >= 3 : prntf("outp",20,"topTuple7List is the top percentage of tuple7List sorted by stddev.")
            if VL >= 3 : prntf("outp",20,"The word 'top' communicates that it is sorted by stddev (not by STIS order).\n")
            topTuple7List              = tuple7ListByStddev [ 0 : int ( len ( tuple7ListByStddev ) * percentThreshold / 100 ) ]
            lengthTopFrameList         = len ( topTuple7List )
            if VL >= 3 : prntf("outp",20,"lengthTopFrameList                        : ", lengthTopFrameList)

            
            if approvedFlag == False :                
                if VL >= 3 : prntf("outp",20,"   T O P   ", percentThreshold ,"   P E R C E N T   O F   F R A M E S   ( =", lengthTopFrameList, " frames)" )
                if VL >= 3 : prntf("outp",20,"   S O R T E D   B Y   S T D D E V")
                for record in topTuple7List :                
                    if VL >= 3 : prntf("outp",20, "\n", record[1], "\t", record[2], "\t\t", record[3], "\t", record[4] )

                if VL >= 3 : prntf("outp",20,"   G E T   M E D I A N   S T D D E V") 
                stddevTopTuple7List        = [ col [ 3 ]     for col in topTuple7List ] # col 3 is stddev , and topTuple7List is from tuple7ListByStddev
                
                tuple7List.clear()
                
                lengthStddevTopTuple7List  = len ( stddevTopTuple7List )
                if VL >= 3 : prntf("outp",20,"lengthStddevTopTuple7List : ", lengthStddevTopTuple7List)
                numpyMean      = np.mean   ( stddevTopTuple7List )
                if VL >= 3 : prntf("outp",20,"numpyMean                                 : ", numpyMean)
                numpyMedian    = np.median ( stddevTopTuple7List )
                if VL >= 3 : prntf("outp",20,"numpyMedian                               : ", numpyMedian)
                numpyStd       = np.std    ( stddevTopTuple7List )
                if VL >= 3 : prntf("outp",20,"numpyStd                                  : ", numpyStd)
                
                if VL >= 3 : prntf("outp",20,"Median stddev of Top ", lengthTopFrameList, " frames        : ", numpyMedian )

                topTuple7ListToText        = [ col [ 1 : 4 ] for col in topTuple7List ] # col 1 is filename, col2 is index, col 3 is stddev
                lengthTopFrameListToText   = len ( topTuple7ListToText )
                if VL >= 3 : prntf("outp",20,"lengthTopFrameListToText                  : ", lengthTopFrameListToText)
        if VL >= 3 : prntf("outp",20," ")


        if stddevSortFlag == True or stisSortFlag == True :
            if VL >= 3 : prntf("outp",20,"   S A V E   T O P   F R A M E S   T O   T E X T   F I L E")
            if VL >= 3 : prntf("outp",20,"outputFolder                              : ", outputFolder)

            if VL >= 3 : prntf("outp",20,"SAVE topTuple7ListToText")
            outputFilename = '__atf__out' + str ( lengthTuple7ListByStddev ) + 'x' + str ( percentThreshold ) + 'q' + str ( lengthTopFrameListToText ) + '.txt' 
            if VL >= 3 : prntf("outp",20,"outputFilename                            : ", outputFilename)
            dnfn = outputFolder + outputFilename
            if VL >= 3 : prntf("outp",20,"dnfn                                      : ", dnfn)        
        
            with open ( dnfn , 'w' ) as outputFile :
                if stddevSortFlag == True :
                    if VL >= 3 : prntf("outp",20,"Top ", percentThreshold, "% sorted by stddev             was selected to be written to output file")
                    for record in topTuple7ListToText :      
                        outputFile.write ( "%s\n" % [ record[0], record[1], record[2] ] )                    
                    outputFile.write ( "\n" )

                if stisSortFlag == True :
                    if VL >= 3 : prntf("outp",20,"Top ", percentThreshold, "% sorted by STIS numeric order was selected to be written to output file")
                    topTuple7ListToText.sort()
                    for record in topTuple7ListToText :
                        outputFile.write ( "%s\n" % [ record[0], record[1], record[2] ] )
            outputFile.close()
        else :
            if VL >= 3 : prntf("outp",20,"   N O   F R A M E S   W E R E   S A V E D   T O   T E X T   F I L E")
            if VL >= 3 : prntf("outp",20,"No list was selected to be written to output file")
                            

        return self._output


    
    

    def savedata ( 
                  self,
                  filepath, 
                  data, 
                  klipparams    = None,
                  filetype      = "",
                  zaxis         = None,
                  more_keywords = None
                  ):
        hdulist                       = fits.HDUList()
        hdulist.append ( fits.PrimaryHDU ( data = data ) )

        filenames                     = np.unique ( self.filenames )
        nfiles                        = np.size ( filenames )
        hdulist[0].header["DRPNFILE"] = ( nfiles, "Num raw files used in pyKLIP" )
        for i, filename in enumerate ( filenames ):
            hdulist[0].header[ "FILE_{0}".format ( i ) ] = filename + '.fits'

        pykliproot                    = os.path.dirname ( os.path.dirname ( os.path.realpath ( __file__ ) ) )

        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = "unknown"

        hdulist[0].header [ 'PSFSUB' ] = ( "pyKLIP", "PSF Subtraction Algo" )
        hdulist[0].header.add_history ( "Reduced with pyKLIP using commit {0}".format ( pyklipver ) )
        hdulist[0].header [ 'CREATOR' ] = "pyKLIP-{0}".format ( pyklipver )

        hdulist[0].header [ 'pyklipv' ] = ( pyklipver, "pyKLIP version that was used" )

        if klipparams is not None:
            hdulist[0].header [ 'PSFPARAM' ] = ( klipparams, "KLIP parameters" )
            hdulist[0].header.add_history ( "pyKLIP reduction with parameters {0}".format ( klipparams ) )

        if zaxis is not None:
            if "KL Mode" in filetype:
                hdulist[0].header [ 'CTYPE3' ] = 'KLMODES'
                for i, klmode in enumerate ( zaxis ):
                    hdulist[0].header [ 'KLMODE{0}'.format ( i ) ] = ( klmode, "KL Mode of slice {0}".format ( i ) )
                hdulist[0].header [ 'CUNIT3' ] = "N/A"
                hdulist[0].header [ 'CRVAL3' ] = 1
                hdulist[0].header [ 'CRPIX3' ] = 1.
                hdulist[0].header [ 'CD3_3'  ] = 1.

        # if "Spectral" in filetype:
        # < contents >

        wcshdr = self.output_wcs[0].to_header()
        for key in wcshdr.keys():
            hdulist[0].header[key] = wcshdr [ key ] 

        center = self.output_centers[0]
        hdulist[0].header.update ( { 'PSFCENTX' : center[0], 'PSFCENTY' : center[1] } )
        hdulist[0].header.update ( {   'CRPIX1' : center[0],   'CRPIX2' : center[1] } )
        hdulist[0].header.add_history ( "Image recentered to {0}".format ( str ( center ) ) )
        
        if more_keywords is not None:
            hdulist[0].header.update ( more_keywords )

        try:
            hdulist.writeto ( filepath, overwrite = True )
        except TypeError:
            hdulist.writeto ( filepath, clobber = True )
            hdulist.close()

# def calibrate_output
# < contents >


                

