# Python wrappers for *Mineos*

[*Mineos*](https://geodynamics.org/cig/software/mineos/) is a widely-used Fortran code for calculating planetary normal modes via numerical integration of the radial scalar equations (see Dahlen and Tromp [1998], section 8). We use *Mineos* for benchmarking our codes. For convenience, we have written Python wrappers to prepare *Mineos* input files, and call the *Mineos* command-line codes using the Python standard module `subprocess`.

## Installation

*Mineos* must be installed (i.e., you must be able to run the *Mineos* commands from your command line, such as `minos_bran`, `eigcon`, `green` and `syndat`). Follow the installation instructions in the *Mineos* manual.

## Using the programs

### Calculating mode eigenfrequencies and eigenfunctions
```
python3 mineos/calculate_modes.py inputs/example_input_Mineos.txt
```

#### Plotting mode eigenfrequencies

```
python3 plot/plot_dispersion.py inputs/example_input_Mineos.txt --use_mineos
```

#### Plotting mode eigenfunctions