## FracNum: fractional differential equations made fun!

![A phase portrait heart<3](docs/pictures/stable_heart_empty.png)

Welcome! Fracnum is the working title of implemented methods to approximate fractional-order differential equations, as used in my thesis on fractional-order differential equations, in particular applied to the Van der Pol equation. Both a "local" and "global" setup are implemented (```fr.splines.SplineSolver```), where the local setup iterates knot for knot and the global method tries to satisfy the equation on the whole time domain directly. The global setup has as advantage that BVP-type problems can be satisfied. However, convergence is much more harder to reach, which is near-unconditionally satisfied in the local approach for most knot sizes.

More documentation, examples and features will be added later. For now, the example file is the place to start out. Enjoy!

## Installation Windows:

```
python -m venv .venv ; ".venv/Scripts/activate.bat" ; python -m pip install -e .
```

## Installation Linux:

```
python -m venv .venv ; source .venv/bin/activate ; python -m pip install -e .
```

## Example file for pretty pictures:

```
python examples/VdP/frac_vdp_oscillator_example.py
```
