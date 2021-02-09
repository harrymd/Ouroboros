# Miscellaneous scripts

Sundry codes, mostly for developer use.

## Manipulating CMT files

Instructions for converting from NDK format to Mineos CMT format are given in `mineos/README.md`.

## Counting modes in a given interval

```
python3 mode_count.py [-h] [--use_mineos]
	path_to_input_file freq_lower freq_upper

```

gives e.g.

```
Searching for modes in the frequency range   0.000000 to   0.500000 mHz.
Mode type  R:     0 modes (multiplicity     0)
Mode type  S:     3 modes (multiplicity    15)
Mode type T0:     0 modes (multiplicity     0)
Mode type T1:     1 modes (multiplicity     5)
Totals:           4 modes (multiplicity    20)
```

## Checking normalisation

The eigenfunctions are normalised as described in `modes/README.md`. This can be checked with numerical integration. Currently only implemented for radial modes. See `-h` for usage. Example

`python3 misc/check_normalisation.py inputs/example_input_Ouroboros.txt R 0 0`