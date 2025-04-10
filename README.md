## FracNum: fractional differential equations made fun!

![A phase portrait heart<3](docs/pictures/stable_heart_empty.png)

[Update: check out the arXiv preprint outlining the method for IVP's!](https://doi.org/10.48550/arXiv.2503.22335)

Welcome! Fracnum is the working title of implemented methods to approximate fractional-order differential equations, as used in my thesis on fractional-order differential equations, in particular applied to the Van der Pol equation. Both a "local" and "global" setup are implemented (```fr.splines.SplineSolver```), where the local setup iterates knot for knot and the global method tries to satisfy the equation on the whole time domain directly. The global setup has as advantage that BVP-type problems can be satisfied. However, convergence is much more harder to reach, which is near-unconditionally satisfied in the local approach for most knot sizes. Furthermore, the general Hilfer derivative can be taken, for which parameter $\beta$ can be provided through ```beta_vals```. For more information on the Hilfer derivative and its connection to IVP-solutions, see for instance [(Furati, 2012)](https://www.sciencedirect.com/science/article/pii/S0898122112000193).

More documentation, examples and features will be added later. For now, the example file is the place to start out. Enjoy!

## Installation Windows:

```
python -m venv .venv ; .venv\Scripts\activate.bat ; python -m pip install -e .
```

## Installation Linux:

```
python -m venv .venv ; source .venv/bin/activate ; python -m pip install -e .
```

## Optional: CUDA acceleration
Fracnum supports using cupy as a drop-in replacement for numpy. To enable this, install optional dependences with `pip install -e .[gpu]`, and change `BACKEND_ENGINE=cupy` in `environment.env`.  

`cupy` is not faster than `numpy` for smaller problems. But [experimentally](experiments/performance/running_times.txt), at a simulation of about 750.000 iterations, `cupy` saves significant time.

NOTE: `cupy` support with analytical `mpmath` sinusoidal forcing through `forcing_params` is not yet implemented. Best is to pass this to the function itself using splines.

## Example file for pretty pictures:

```
python examples/VdP/frac_vdp_oscillator_example.py
```

## Example experimentation file:

```
python examples/VdP/frac_vdp_oscillator_experimentation.py
```
