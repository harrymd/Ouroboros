# Sensitivity kernels

Sensitivity kernels are calculated using the formulae from Dahlen and Tromp (1998, section 9.3) derived using Rayleigh's principle.

## Running the code

First, you must calculate the normal modes (see `modes/README.md`) based on an input file such as `inputs/example_input_Ourorobos.txt`. Next, to calculate the sensitivity kernels for all of the modes specified in the input file, run a command such as

```
python3 kernels/run_kernels.py inputs/example_input_Ouroboros.txt
```

### The format of the output files

There are five relevant variables

 * *κ*, bulk modulus;
 * *μ*, shear modulus;
 * *ρ*, density;
 * *α*, P-wave speed; and
 * *β*, S-wave speed.

If *κ*, *μ*, and *ρ* are given, then *α* and *β* are uniquely determined. Similarly, if *α*, *β* and *ρ* are given, then *κ* and *μ* are uniquely determined. Therefore, it is natural to consider two equivalent of groups of three sensitivity kernels. The first group is

 * *K<sub>κ</sub>*, sensitivity to *κ* with fixed *μ* and *ρ*;
 * *K<sub>μ</sub>*, sensitivity to *μ* with fixed *κ* and *ρ*; and
 * *K<sub>ρ</sub>*, sensitivity to *ρ* with fixed *κ* and *μ*.
 
The second group is 

 * *K<sub>α</sub>*, sensitivity to *α* with fixed *β* and *ρ*;
 * *K<sub>β</sub>*, sensitivity to *β* with fixed *α* and *ρ*; and
 * *K<sub>ρ'</sub>*, sensitivity to *ρ* with fixed *α* and *β*.


### Plotting kernels

The syntax is very similar to plotting the mode eigenfunctions (see `modes/README.md`, or try `python3 plot/plot_kernels.py -h`). A command such as

```
python3 plot/plot_kernels.py inputs/example_input_Ouroboros.txt R 0 0
```

will produce a plot like

<img src="../docs/example_sensitivity_kernel.png" width="80%" title ="Example of mode sensitivty of mode 0R0 from example input file."/>

## Other functions

The `kernels/` directory contains a few other functions which are used for development and not written for public use:

 * `kernels_brute.py` Calculating kernels by brute force (i.e., varying one layer at a time) for benchmarking. This is extremely computationally slow.
 * `test_kernels.py` Plotting kernels against brute-force kernels.

## Issues

### Density kernels including gravity effect

Currently, the full form of the sensitivity to density has not been implemented in the cases involving background gravity or self-gravity. The extra terms are ignored (with a warning), although the eigenvalues and eigenfunctions include the effect of gravity, which reduces the error in this approximation. Ignoring the extra terms is a reasonable approximation at higher frequencies.

### Some kernels not benchmarked

So far, the sensitivity kernels have only been benchmarked against brute-force calculations for radial-mode sensitivity to bulk modulus, shear modulus and density. Future tests will benchmark the spheroidal and toroidal modes and the composition kernels which give sensitivity to P-wave speed, S-wave speed and density when these two are fixed. Until this is done, it is likely that there are errors in the units of the sensitivity kernels.