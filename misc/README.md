# Miscellaneous scripts

Sundry codes, mostly for developer use.

## Checking normalisation

The eigenfunctions are normalised as described in `modes/README.md`. This can be checked with numerical integration. Currently only implemented for radial modes. See `-h` for usage. Example

`python3 misc/check_normalisation.py inputs/example_input_Ouroboros.txt R 0 0`