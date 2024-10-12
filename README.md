## FracNum: fractional differential equations made fun!

Fracnum is the working title of implemented methods to approximate fractional-order differential equations.

As of current (October 11 2024), the main implementation and feature lies in `fracnum.BernsteinSplines.solve_ivp_local`, an efficient way to approximate fractional ordinary differential equations using Bernstein-Splines. An example on the fractional-order Van der Pol oscillator is provided in `examples`.

The other Splines-solving method (`fracnum.splines.BernsteinSplines.solve_ivp_global` and all implementation in `fracnum.numerical` except for `ivp_diethelm`) need to be re-tested and can be considered under construction for now.

## Installation Windows:

```
python -m venv .venv ; .venv/Scripts/activate.bat ; python -m pip install -e .
```

## Installation Linux:

```
python -m venv .venv ; source .venv/bin/activate ; python -m pip install -e .
```

## Example run for some pretty pictures:

```
python examples/VdP/frac_vdp_oscillator_example.py
```
