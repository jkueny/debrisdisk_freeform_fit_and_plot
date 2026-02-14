import logging
import time
import re 
import os
import sys
import copy

ansi_re             = re.compile(r'\x1b\[[0-9;]*m')

def prntf(sender, incomingLevel = 20, message1 = "", message2 = "", message3 = "", message4 = "", message5 = "", message6 = "", message7 = "", message8 = "", message9 = "", message10 = "", message11 = "", message12 = "", message13 = "", message14 = "" ) :
    """
        The TOP print statement offers no line numbers.
            HPC allows it.
            It has to be actively uncommented, when on HPC.
        
        The BOTTOM print statement offers LINE NUMBERS, and is used for diagnostics / development.
            HPC does not appear to allow f-strings.
            It has to be actively commented out, when on HPC.
    """
    
    lineNumberString = str(sys._getframe().f_back.f_lineno).rjust(4) + " "
    print(sender, " ", lineNumberString, message1, message2, message3, message4, message5, message6, message7, message8, message9, message10, message11, message12, message13, message14)
    printString1 = sender + " " + str(lineNumberString) + str(message1) + str(message2) + str(message3) + str(message4) + str(message5) + str(message6) + str(message7) + str(message8) + str(message9) + str(message10) + str(message11) + str(message12) + str(message13) + str(message14)

    
    home              = os.path.expanduser ( "~" )
    logpath           = home + "/Hubble/STIS/logsFolder/"
    logFilename       = "logfile.txt" # This will need to be revisited when I get logging for nmf_imaging...
    fqpn              = logpath + logFilename
    t                 = time.localtime()
    
    
    with open ( fqpn, 'a' ) as outputFile :
        
        #    logging.basicConfig(level=logging.DEBUG, filename='logfile.log', filemode='a', format='%(asctime)s:%(levelname)s:%(message)s' )
        
        if incomingLevel       == 20 :
            logging.basicConfig ( level=logging.INFO    , filename='logsFolder/logfile.log', filemode='a', format='%(asctime)s:%(levelname)s:%(message)s' )
            logging.info        ( printString1 )
            printString2        = time.asctime(t) + " INFO     " + printString1 + "\n"

        if incomingLevel       == 30 :
            logging.basicConfig ( level=logging.WARNING , filename='logsFolder/logfile.log', filemode='a', format='%(asctime)s:%(levelname)s:%(message)s' )
            logging.warning     ( printString1 )
            printString2        = time.asctime(t) + " WARNING  " + printString1 + "\n"
            #    This will be for radonCenter determinations that are shorter or narrower than usually expected
                
        if incomingLevel       == 40 :
            logging.basicConfig ( level=logging.ERROR   , filename='logsFolder/logfile.log', filemode='a', format='%(asctime)s:%(levelname)s:%(message)s' )
            logging.error       ( printString1 )
            printString2        = time.asctime(t) + " ERROR    " + printString1 + "\n"
        
        if incomingLevel       == 50 :
            logging.basicConfig ( level=logging.CRITICAL, filename='logsFolder/logfile.log', filemode='a', format='%(asctime)s:%(levelname)s:%(message)s' )
            ansiRemoved         = re.sub ( ansi_re, '', str ( copy.copy ( printString1 ) ) )
            logging.critical    ( ansiRemoved )
            printString2        = time.asctime(t) + " CRITICAL " + ansiRemoved + "\n"
            
        outputFile.write  ( printString2 )
