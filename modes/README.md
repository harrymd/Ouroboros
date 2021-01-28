# Calculating normal modes

## Running the code

Call the main script from the command line:

```bash
python3 modes/calculate_modes.py inputs/example_input_Ouroboros.txt
```

The only argument is the path to the input file. The default input file (shown here) should run sucessfully without any other changes, and provide detailed output messages.

### Changing the input variables

The example input file is

```
path_to_model models/prem_noq_noocean.txt
path_to_outdir ../../output/Ouroboros
gravity_switch 0 
mode_types R S T 
l_limits 0 5 
n_limits 0 10 
n_layers 700
```

Each line has a string descriptor followed by one or more arguments. The order of the lines must not be changed. The lines are

* `path_to_model`: The path to the model file (see note* below).
* `path_to_outdir`: The path to the desired output directory (see note* below).
* `gravity_switch`: Controls gravity. Ignored for toroidal modes. For radial and spheroidal modes, should be one of
 *  `0` (no gravity);
 * `1` (background gravity but no perturbation; Cowling approximation); or
 * `2` (background gravity and perturbation).
* `mode_types`: A list of mode types (separated by spaces). Should be one or more of
 * `R` (radial modes);
 * `S` (spheroidal modes); or
 * `T` (toroidal modes).
* `l_limits`: Two integers separated by a space, setting the lower and upper value of *ℓ*. These are ignored for radial modes (which have *ℓ* = 0). The computation time is proportional to the number of *ℓ* values.
* `n_limits`: Two integers separated by a space, setting the lower and upper value of *n*.
* `n_layers`: The number of layers in the finite-element model. The user should look for an optimal number of layers that is neither too small (this leads to inaccurate calculations for higher-frequency modes) nor too large (this leads to a very large, time-consuming eigenvalue problem). The details of the computational grid are handled internally by `modes.lib.mantlePoint_equalEnd()`, taking into account discontinuities and using a graded mesh with finer spacing near interfaces.

Modes with certain combinations of *n* and *ℓ* cannot exist without external forcing and are skipped by *Ouroboros*. These are:

* Toroidal modes with *ℓ* = 0;
* All modes with *n* = 0, *ℓ* = 0;
* All modes with *n* = 0, *ℓ* = 1.

*Note: the paths can be either absolute paths (e.g. `/usr/files/model.txt`) or relative paths (e.g. `../../output`).

### Format of the model file

The model file can be specified in one of two formats. The first option is the tabular format used by the *Mineos* code (see the *Mineos* manual, Masters et al. 2011, section 3.1.2.1), although anisotropic wavespeeds are silently converted to the isotropic mean, and attenuation is currently ignored. The second option is a simple space-separated text file with four columns: radius (increasing from 0, in km), density (in g/cm3), P-wave speed and S-wave speed (both in km/s).

### The format of the output files

The path to the output is determined by the input parameters. For example, if `path_to_outdir = ../output`, `path_to_model = model/my_model.txt`, `n_layers = 700`, `n_max = 5`, `l_max = 10`, `grav_switch = 0`, and `mode_type = S`, then the output will be saved in `../output/my_model_00700/00005_00010_0/grav_0/S`. The output consists of the two parts: eigenvalues and eigenvectors (also called eigenfunctions).

For a planet with multiple solid regions separate by fluid regions (for example, the Earth), the toroidal modes in each solid region are completely decoupled from the toroidal modes of other solid regions. Therefore, the eigenvalues for each solid region are saved in separate files `eigenvalues_000.txt`, `eigevalues_001.txt`, ..., labelling from the centre of the planet outwards, and similarly for the eigenfunctions. As an example, for Earth, `000` corresponds to inner-core toroidal modes and `001` corresponds to mantle toroidal modes. Fluid regions do not have toroidal modes (therefore entirely fluid planets do not have any toroidal modes).

Within the output directory, the eigenvalues are saved in the file `eigenvalues.txt`. Each line has the value of *n*, *ℓ* and the frequency (in mHz) for a given mode. The modes are listed in order of increasing *n*, then increasing *ℓ*.

The eigenvectors are stored in the subdirectory `eigenfunctions`. There is one file per mode. The files are stored in NumPy binary format to save space. They can be read with the `numpy.load` function. As an example, the mode with *n* = 3 and *ℓ* = 5 will be saved as `00003_00005.npy`. For spheroidal modes, the output is an array with three rows, corresponding to *r* (radial coordinate in metres), *U* (radial eigenfunction) and *V* (consoidal eigenfunction). For toroidal modes, there are just two rows, corresponding to *r* and *W* (toroidal eigenfunction). For radial modes, there are two rows, corresponding to *r* and *U*. For definitions of *U*, *V* and *W*, see Dahlen and Tromp (1998, section 8.6.1).

The normalisation of the eigenfunctions differs from the normalisation used in *Mineos*. Although we use the same normalisation formulae (specifically, the formulae given in section 3.2.3.2 of the *Mineos* manual), we use different units, so the results differ by a scalar factor. To match the eigenfunctions from the two codes, the eigenfunctions from *Ouroboros* must be multiplied by *R*<sup>2</sup>/1000, where *R* is the radius of the planet (in km). They might also have different signs (the sign of the eigenfunctions is arbitrary). Note that both of these normalisation conventions differ from that used in Dahlen and Tromp (1998, equation 8.107) by a factor of the angular frequency squared.

 <a href="#top">Back to top</a>

### Plotting the output

#### Viewing the mode frequencies

To plot a mode diagram, try

```
# Plots spheroidal and radial modes.
python3 plot/plot_dispersion.py inputs/example_input_Ouroboros.txt
```

The plot will appear on your screen, and will also be saved in `dir_output`, in a subdirectory called `plots/`. By default, the spheroidal modes are plotted, and the radial modes, with *ℓ* = 0, are added automatically if they are found in the output directory:

<img src="../docs/example_dispersion.png" width="90%" title = "Angular-order--frequency diagram for spheroidal modes using example input file.">

For toroidal modes, you must specify the solid region whose modes you wish to plot (see discussion in section *The format of the output files*, above). For example, to plot the  Earth's mantle toroidal modes you would use the command

```bash
# Plot toroidal modes from second solid region.
python3 plot/plot_dispersion.py inputs/example_input_Ouroboros.txt --toroidal 1
```

To plot dispersion from *Mineos* output, use the same syntax with the `--mineos` flag.

#### Viewing the mode displacement patterns

Similarly, eigenfunctions can be plotted from the command line, specifying the mode type (R, S or T) as well as *n* and *ℓ*, for example

```bash
# Plot spheroidal mode with n = 2, l = 4.
python3 plot/plot_eigenfunctions.py inputs/example_input_Ouroboros.txt S 2 4
```

which yields the following figure:

<img src="../docs/example_eigenfunction.png" width="40%" title ="Example of mode 2S4 from example input file."/>

Once again, for toroidal modes you must also specify the index of the solid region (see *Viewing the mode frequencies*, above) as follows:

```bash
# Plot toroidal mode with n = 2, l = 4, from the second solid region.
python3 plot/plot_eigenfunctions.py inputs/example_input_Ouroboros.txt
			T 2 4 --toroidal 1
```

For radial modes, *ℓ* must be 0. To plot modes from *Mineos* output, use the same syntax with the `--mineos` flag.



<a style="color: #000000" name="method"/>

## Method

To calculate the modes of a spherically-symmetric planet, we apply the Rayleigh-Ritz method (Dahlen and Tromp, 1998, section 7), refining the approach of Buland and Gilbert (1984). Our implementation is described fully by Ye (2018) and Shi et al. (2020), and summarised here. We use a weak form of the governing equations. We remove the undertones (modes associated with fluid flow, also known as the essential spectrum) from the solution space by introducing a variable related to pressure. We discretise this weak form with continuous mixed Galerkin finite elements to give a matrix eigenvalue problem. In the spherically-symmetric case, this eigenvalue problem can be solved using standard methods. The eigenvalues are the (squared) frequencies of the modes, and the eigenvectors are their displacement patterns.

<a href="#top">Back to top</a>

<a style="color: #000000" name="structure"/>

## Structure of code

The structure of the code is described by the following flowchart:

![](../docs/flowchart.png "Flowchart for Ouroboros code")

<a href="#top">Back to top</a>

<a style="color: #000000" name="benchmarking"/>

## Benchmarking and performance

### Testing against *Mineos*

The *Mineos* code (Masters et al., 2011) is the *de facto* standard for calculation of Earth's normal modes. Here we present a comparison between *Ouroboros* (version 3.s) and *Mineos* (version 1.0.2). We calculated all of the spheroidal modes with *n* < 50, *ℓ* < 60 and *f* < 15 mHz. We used the `demos/prem_noocean.txt` model from *Mineos*, modified by setting attenuation to 0. For *Ouroboros*, we used `n_layers = 700`. We made comparisons with only gravity (`g_switch = 1`, *Mineos* gravity cut off of 0 mHz) and gravity with perturbation (`g_switch = 2`, *Mineos* gravity cut off at arbitrarily high frequency, e.g. 50 mHz). In *Mineos*, gravity cannot be neglected altogether (`g_switch = 0` in *Ouroboros*), so we did not test this case (although all three cases converge for higher-frequency modes). Comparison of the frequencies (shown for the case `g_switch = 1` below) shows that frequency differences are small: less than 0.5 % for all modes except for <sub>2</sub>S<sub>1</sub>. We are not sure the cause of the discrepancy for this mode, which vanishes in the case `g_switch = 2`. The figure shows that the frequencies calculated with *Ouroboros* are systematically higher than the frequencies from *Mineos*, and the discrepancies are largest for modes with low *n* and high *ℓ*. We do not know what causes these systematic differences.

![](../docs/frac_freq_diff_Ouroboros_Mineos_1.png "Comparison of frequencies calculated with Ouroboros and Mineos")

We can also compare the eigenfunctions, as shown in the figure below. The agreement is within 0.5 % in most cases, but significant differences are observed for the mode <sub>2</sub>S<sub>1</sub> (discussed above) and modes near occurring near the intersections of branches. We discuss this latter case in detail in Ye (2018) and Matchette-Downes et al. (in prep.). In short, we believe that the discrepancy is due to the failure of the numerical integration approach of *Mineos* to guarantee the orthogonality of the eigenfunctions, especially for the Stoneley-type modes (solid-fluid interface modes) for which it is difficult to enforce the boundary conditions. Apart from near-intersection modes, discrepancies are also found for modes with higher *ℓ*. We do not think these differences are intrinsic, but probably just due to the coarse grid in the default *Mineos* model (*Mineos* models have a hard-coded limit of 350 nodes, although this could easily be changed and re-compiled). For most practical applications, the differences between *Mineos* and *Ouroboros* will probably not be significant.

![](../docs/eigfunc_diff_Ouroboros_Mineos_1.png "Comparison of eigenfunctions calculated with Ouroboros and Mineos")

### Computational cost

The code is not optimised for speed, and tends to be slower than *Mineos* for calculating a similar number of modes. This is probably due to use of double-precision variables, initialisation of many variables including complicated objects (*Mineos* is written in Fortran, which promotes very lean code) and intrinsic differences between our method (FEM) and the integration method used in *Mineos*. Nonetheless, the modes required for most Earth-science and planetary-science applications can be calculated on a laptop in a reasonable amount of time. 

spheroidal_modes (switch = S_G): l =    50 (from     0 to    50)
Total time used: 2970.022 s

For example, the spheroidal modes of an Earth model with 700 layers can be calculated for *ℓ* up to 50 and *n* up to 60 in about X seconds without gravity, Y seconds with gravity, and Z seconds with gravity and perturbation using a single core of an [SKX compute node on the Stampede2 cluster](https://portal.tacc.utexas.edu/user-guides/stampede2#overview-skxcomputenodes). The eigenvalue problem at each value of *ℓ* is independent of the other *ℓ*-values, so it is trivial to parallelise the loop over *ℓ* ('embarrassingly parallel') if faster calculation is necessary.

### Computational limitations

We have not explored the limits of the code for high frequencies, large values of *ℓ*, or very complicated models. Calculations are performed at double precision, so numerical errors are probably small compared to typical model or data uncertainties in geophysics.