'''
Initial conditions:
- Existing reduced data where each frame has a header value for "PARANG".
- Existing configuration YAML file where the desired disk model parameters are defined.
- 
'''

import os
import yaml
import numpy as np
from astropy.io import fits
import argparse




def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param-file", type=str,
                        help="Name of YAML config/parameter file in initialization_files.")
    parser.add_argument("-d", "--data-dir", type=str,
                        help="Directory containing the dataset to modify.",
                        )
    args = parser.parse_args()

    # Read in the YAML param file
    # but first make sure the path is right...
    str_yaml = f"initialization_files/{args.param_file}"
    if not os.path.exists(args.param_file):
        print(f"YAML param file not found: {str_yaml}")
        return
    with open(str_yaml, "r") as yaml_file:
        params_yaml = yaml.safe_load(yaml_file)
    
    



if __name__ == "__main__":
    main()

