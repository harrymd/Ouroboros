# Python wrappers for *Mineos*

[*Mineos*](https://geodynamics.org/cig/software/mineos/) is a widely-used Fortran code for calculating planetary normal modes via numerical integration of the radial scalar equations (see Dahlen and Tromp [1998], section 8). We use *Mineos* for benchmarking our codes. For convenience, we have written Python wrappers to prepare *Mineos* input files, and call the *Mineos* command-line codes using the Python standard module `subprocess`.

## Installation

*Mineos* must be installed (i.e., you must be able to run the *Mineos* commands from your command line, such as `minos_bran`, `eigcon`, `green` and `syndat`). Follow the installation instructions in the *Mineos* manual.

## Calculating mode eigenfrequencies and eigenfunctions

```
python3 mineos/calculate_modes.py inputs/example_input_Mineos.txt
```

with input file

```
path_model models/prem_noocean.txt
path_out_dir ../../output/mineos
grav_switch 2
mode_types S
n_limits 0 15 
l_limits 0 30 
f_limits 0.0 5.5
eps 1.0E-10
max_depth all
```

Warning: *Mineos* has a bug when doing mode summation including radial modes if the eigenfunctions are saved with a large value of `max_depth` (including `max_depth = 'all'`).

### Plotting mode eigenfrequencies

```
python3 plot/plot_dispersion.py inputs/example_input_Mineos.txt --use_mineos
```

### Plotting mode eigenfunctions

## Mode summation

```
python3 mineos/summation.py inputs/example_input_Mineos.txt
inputs/example_input_Mineos_summation.txt
```

with input file

```
path_channels inputs/example_station_list.txt 
path_cmt inputs/example_cmt_china.txt
f_lims same
n_samples 1000
data_type 0
plane 0
```

### The channels file

The channels file (`path_channels`) describes the observer properties, i.e. the seismometer location and the orientation of the sensors. The channels file can list multiple stations and each station can have multiple sensors. The format of the channels file is described in the *Mineos* manual (section 4.4). It is passed to the *Mineos* utility `simpledit` for conversion into a CSS database before summation.

### The CMT file

The centroid moment tensor (CMT) file (`path_cmt`) describes the earthquake source. It must be in the format used by *Mineos*. If you have downloaded a moment-tensor solution from the [Global CMT](https://www.globalcmt.org/) project, it will be written in [NDK](https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained) format. To convert from *Mineos* to NDK, use a command such as

```
python3 misc/cmt_io.py --path_ndk inputs/CMTs/tohoku.ndk
	--path_mineos inputs/CMTs/tohoku.txt --mineos_dt 10.0
```

The time interval between samples in the synthetic seismograms is also controlled by the *Mineos* CMT file (it is the 10th argument), but of course it is not part of the NDK format. Use the `--mineos_dt` flag to specify the time interval (in seconds).

## Things to add

* Plotting eigenfunctions.
* Plotting synthetic seismograms and Green's functions.
* Maybe re-grid comparison trace to assure phase alignment

```
python3 plot/plot_summation.py --use_mineos ../../input/mineos/input_Mineos_stoneley.txt ../../input/mineos/input_Mineos_summation_stoneley.txt PAYG VHZ
```