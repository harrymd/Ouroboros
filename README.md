# Ouroboros

A *Python* package which calculates the seismic normal modes of spherically-symmetric planets using a finite-element method, and does other related calculations.

Normal-mode calculations can be run without gravity, with only background gravity (Cowling approximation), or include background and self-gravity. Earth models can contain any number of fluid and solid regions. Currently the code does not handle anisotropy, anelasticity, or rotation. Post-processing tools are provided for mode summation (to generate synthetic seismograms) and calculation of sensitivity kernels. Scripts are included for plotting of mode frequencies, eigenfunctions, sensitivity kernels and synthetic seismograms.

## Contents
  * [Using the code](#usage)
  * [Method](#method)
  * [Structure of code](#structure)
  * [Benchmarking and performance](#benchmarking)
  * [Examples](#examples)
  * [History and contributors](#history)
  * [How to contribute](#contribute)
  * [How to cite](#attribution)
  * [Related repositories](#related)
  * [References](#references)

<a style="color: #000000" name="usage"/>

## Using the code

### Installation

You must have Python3 installed, including the packages NumPy and SciPy (and MatPlotLib if you wish to use the plotting scripts). We recommend using the Python environment manager [Anaconda](https://docs.anaconda.com/anaconda/install/) to install Python and manage Python packages. A suitable environment for the Ouroboros code can be created and activated with the commands

```bash
conda create --name Ouroboros python=3 numpy scipy matplotlib
conda activate Ouroboros
```
Then, download this repository and it will be ready to use.

### Running the codes

Instructions for using the various codes are given in separate `README.md` files in the following directories:

 * `modes/` Calculating the normal modes of a given model.
 * `kernels/` Calculating the sensitivity kernels of modes.
 * `summation/` Calculating synthetic seismograms with mode summation.
 * `mineos/` Python wrappers for the [*Mineos*](https://geodynamics.org/cig/software/mineos/) normal-mode library, useful for comparison with *Ouroboros*.



<a href="#top">Back to top</a>

<a style="color: #000000" name="examples"/>
## Examples

<a href="#top">Back to top</a>

<a style="color: #000000" name="history"/>
## History and contributors

This code has been developed by members of the [GMIG group](http://gmig.blogs.rice.edu/) at Rice University. The code was first developed in Matlab by Jingchen Ye and [Jia Shi](https://sites.google.com/view/jiashi/) around 2017. In 2019 it was translated to Python by [Jiayuan Han](https://github.com/hanjiayuan236), who made the improvements described below in versions 1, 2, 3 and 'Anelasticity'. Currently the code is maintained by [Harry Matchette-Downes](http://web.mit.edu/hrmd/www/home.html) and Jia Shi.

#### Version 1

Initial translation from Matlab. Can calculate the modes of an SNREI (spherically-symmetric, non-rotating, elastic, isotropic) Earth, with a solid inner core, fluid outer core, and solid mantle.

#### Version 2

The treatment of fluid regions was made more general, allowing any number of fluid regions to be included simply by altering the input file.

#### Version 3

Added code for the generation of synthetic seismograms by mode summation. Some small bugs were fixed relating to the units of different terms.

##### Version 3.h

This version contains small changes made by: code reorganisation to avoid duplication, more detailed documentation (including this Readme) and a user-friendly interface, and testing against Mineos, calculation of sensitivity kernels, and tools for plotting. It was used in our paper about mixed Stoneley-Rayleigh modes (Matchette-Downes et al., in prep.). Mode summation is not included in this version.

##### Version 'Anelasticity'

Work on anelasticity was started, but is not currently active.

<a href="#top">Back to top</a>

<a style="color: #000000" name="contribute"/>

## How to contribute

We welcome any form of contributions, such as bug reports, new code, feature requests, or comments.

<a href="#top">Back to top</a>

<a style="color: #000000" name="attribution"/>

## How to cite

If you use this repository for published research, please cite this work, for example 'we calculated modes using the *Ouroboros* code (Ye, 2018; Shi et al. 2020; <https://github.com/hanjiayuan236/RadialPNM_py>)'.

<a href="#top">Back to top</a>

<a style="color: #000000" name="related"/>

## Related repositories

* The [Matlab version](https://github.com/js1019/RadialPNM) of Ouroboros, which predates version 1 of the Python code, so it lacks some functionality and may contain small bugs.
* [NormalModes](https://github.com/js1019/NormalModes) and [PlanetaryModels](https://github.com/js1019/PlanetaryModels): implementation of the same method to non-spherically-symmetric planets. Mode summation for this 3D case is a work in progress, found [here](https://github.com/harrymd/NMSummation).
* Repository for the paper Matchette-Downes et al. (2020) (a work in progress).

<a href="#top">Back to top</a>

<a style="color: #000000" name="references"/>

## References

* [Buland and Gilbert (1984)](https://doi.org/10.1016/0021-9991(84)90141-4). *Computation of free oscillations of the Earth*. Journal of Computational Physics 54.1, pp. 95-114.
* Dahlen and Tromp (1998) *Theoretical Global Seismology*. Princeton University Press.
* [Masters, Woodhouse, and Freeman (2011)](https://geodynamics.org/cig/software/mineos/). *Mineos*. Version 1.0.2.
* Matchette-Downes, Shi, Ye, Han, de Hoop and van der Hilst (in prep.) *Mixed Rayleigh-Stoneley modes: A new probe for Earth's core-mantle boundary*.
* [Shi, Li, Xi, Saad, de Hoop (2020)](https://arxiv.org/abs/1906.11082) *A Rayleigh-Ritz method based approach tocomputing seismic normal modes in the presence of an essential spectrum*.
* [Ye (2018)](https://scholarship.rice.edu/handle/1911/104942) *Revisiting the computation of normal modes in SNREI models of planets - close eigenfrequencies* M.Sc. thesis, Rice University.

<a href="#top">Back to top</a>