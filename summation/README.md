# Mode summation

The response of a planet to an earthquake source can be calculated by mode summation, as given by the formulae from Dahlen and Tromp (1998, section 10.3).

## Using the code

### Preparing mode input

For more information on these steps, see the `README.md` files in the relevant directories (`modes/`, `kernels/`).

1. Calculate the mode frequencies and eigenfunctions:

    ```
    python modes/calculate_modes.py inputs/example_input_Ouroboros.txt
    ```

2. If using attenuation, calculate the sensitivity kernels and apply attenuation correction

    ```
    python kernels/run_kernels.py inputs/example_input_Ouroboros.txt
    python modes/attenuation_correction.py inputs/example_input_Ouroboros.txt
    ```
    
3. Calculate the gradients and potential of the eigenfunctions (see `modes/README.md` for more information):
    
    ```
    python modes/calculate_potential.py inputs/example_input_Ouroboros.txt
    python modes/calculate_gradients.py inputs/example_input_Ouroboros.txt
    ```
    
### Summing modes

```
python summation/run_summation.py inputs/example_input_Ouroboros.txt
	inputs/example_input_Ouroboros_summation.txt
```

The summation input file looks like

```
mode_types S 
f_lims 0.0 5.0
path_channels inputs/example_station_list.txt
path_cmt inputs/example_cmt_china.txt
time_step 10.0
n_samples 8000
pulse triangle
output_type acceleration
attenuation full
correct_response 1 
```

where the lines are ... 




