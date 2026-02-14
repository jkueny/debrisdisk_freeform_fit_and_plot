# FFortissiMo

**FFortissiMo** (Pixel-based Freeform Forward Modeling) is a Python package for fitting pixel-by-pixel freeform models to KLIP-reduced astronomical images of extended objects, with a focus on debris disk observations.

## Overview

FFortissiMo provides a flexible framework for forward modeling extended astronomical sources (primarily debris disks) in high-contrast imaging data. The package leverages JAX for efficient gradient-based optimization and integrates with the KLIP (Karhunen-Loève Image Processing) pipeline for PSF subtraction.

### Key Features

- **Pixel-by-pixel freeform modeling** of extended sources in high-contrast imaging data
- **JAX-based optimization** for efficient gradient computation and optimization
- **KLIP integration** with forward modeling for accurate PSF subtraction
- **Support for ADI and RDI modes** (Angular Differential Imaging and Reference Differential Imaging)
- **Flexible regularization** including spatial frequency penalties and radial profile constraints
- **Physical model fitting** to freeform results for parameter extraction

## Installation

### Requirements

- Python >= 3.8
- numpy == 1.26.4
- jax >= 0.4.0
- jaxlib >= 0.4.0
- astropy >= 5.0.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- pyyaml >= 6.0
- emcee >= 3.1.0
- numba >= 0.56.0
- optax >= 0.1.0

### Installation from Source

```bash
# Clone the repository
git clone <repository-url>
cd debrisdisk_freeform_fit_and_plot

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Configuration File

Create or use an existing YAML configuration file in `initialization_files/`. See existing examples for reference (e.g., `HR4796_g_camsci1_20250408_09.yaml`).

### 2. Run KLIP Reduction

```bash
python src/ffortissimo/ff_klip.py -p initialization_files/your_config.yaml
```

This step:
- Loads and prepares the dataset
- Creates binary masks for disk generation and noise mapping
- Runs KLIP reduction with forward modeling
- Generates KLIP basis file (`*_klbasis.h5`)
- Creates noise maps and other necessary files

### 3. Create Masks and Estimate Noise

```bash
python src/ffortissimo/ff_mask.py -p initialization_files/your_config.yaml
```

This step creates optimization masks and noise maps used during fitting.

### 4. Fit Freeform Model

```bash
python src/ffortissimo/diskfit_freeform.py -p initialization_files/your_config.yaml --num-iterations 50000
```

This performs the freeform model optimization to fit the KLIP-reduced data.

### 5. (Optional) Fit Physical Model

```bash
python scripts/modelfit_physical_to_freeform.py -p initialization_files/your_config.yaml
```

Fit a physics-informed parametric disk model to the freeform results using MCMC.

## Project Structure

```
debrisdisk_freeform_fit_and_plot/
├── src/
│   └── ffortissimo/          # Main package
│       ├── diskfit_freeform.py      # Main freeform fitting script
│       ├── ff_klip.py               # KLIP reduction pipeline
│       ├── ff_mask.py               # Mask and noise map generation
│       ├── modeling/                # Disk model classes
│       ├── utils/                   # Utility functions
│       └── dev/pyklip/              # Modified pyklip package
├── scripts/                   # Analysis scripts
├── initialization_files/      # YAML configuration files
├── starting_models/           # Initial model FITS files
└── tests/                     # Test scripts
```

## Configuration

The pipeline is configured via YAML files in `initialization_files/`. Key configuration sections include:

- **Data paths**: Directories for input data, KLIP outputs, and results
- **Mask parameters**: Inner/outer working angles, optimization regions
- **KLIP settings**: Number of KL modes, annuli, subsections
- **Optimization parameters**: Learning rate, regularization strength, iteration count
- **Instrument settings**: PSF information, pixel scales, centers

See `initialization_files/HR4796_g_camsci1_20250408_09.yaml` for a detailed example.

## Usage Examples

### Basic Freeform Fitting

```python
from ffortissimo.diskfit_freeform import main

config_file = "initialization_files/your_config.yaml"
main(config=config_file,
     num_iterations=50000,
     init_model="starting_models/model.fits",
     reg_lambda=1.0,
     learning_rate=0.01)
```

### Custom Optimization Settings

```bash
python src/ffortissimo/diskfit_freeform.py \
    -p initialization_files/config.yaml \
    --num-iterations 100000 \
    --reg-lambda 0.5 \
    --learning-rate 0.005 \
    --delta 0.1
```

## Output Files

The pipeline generates several output files:

- **Freeform model FITS files**: Best-fit freeform disk model
- **Forward modeled images**: PSF-subtracted images using the freeform model
- **Optimization history**: Loss values and convergence metrics
- **Visualization plots**: Comparison plots showing data, model, and residuals

Outputs are saved in the directory specified by `resultsdir` in the configuration file.

## Citation

If you use FFortissiMo in your research, please cite:

```bibtex
@software{ffortissimo2024,
  author = {Kueny, Jay},
  title = {FFortissiMo: Pixel-based Freeform Forward Modeling},
  year = {2024},
  version = {0.1.0}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Contact

- **Author**: Jay Kueny
- **Email**: jkueny@arizona.edu
- **Institution**: University of Arizona

## Acknowledgments

This package builds upon and extends functionality from:
- [pyKLIP](https://bitbucket.org/pyKLIP/pyklip) for KLIP reduction
- JAX for automatic differentiation and optimization
- The high-contrast imaging community for methods and inspiration
