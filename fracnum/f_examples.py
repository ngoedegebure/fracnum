
import numpy as np

def f(self, t_vals, x_vals, params, bernstein=False):
    if 'a' in params.keys() and 'omega' in params.keys():
        forcing = params['a'] * np.sin(params['omega']*t_vals)
    else:
        forcing = 0
        
    mu = params['mu']
    if bernstein:
        A_x_in, A_y_in = x_vals
        # breakpoint()

        # x' = y
        A_x_out = self.upscale(A_y_in, 3)

        # y' = mu * (1-x^2)y - x

        A_y_out = mu * self.mult(1 - self.mult(A_x_in, A_x_in), A_y_in) - self.upscale(forcing + A_x_in, 3)

        return np.array([A_x_out, A_y_out])
    else:
        x, y = x_vals[:, 0], x_vals[:, 1]

        # x' = y
        x_out = y
        # y' = mu * (1-x^2)y - x
        y_out = -x - mu * (x**2 -1)*y

        return np.array([x_out, y_out]).T
    