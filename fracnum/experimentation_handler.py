import numpy as np
import xarray as xr
import copy
import itertools

class ExperimentationHandler():
    def __init__(self, bs, f_object, vary_params, alpha_fun = None):
        self.bs = bs
        self.f_object = f_object

        self.vary_params_input = vary_params

        if alpha_fun is None:
            self.alpha_fun = lambda alpha:alpha # Just return alpha
        else:
            self.alpha_fun = alpha_fun

        self.n_vary = len(vary_params.keys())

        self.param_space = ExperimentationHandler.build_param_space(vary_params)
        self.param_types = ExperimentationHandler.get_param_types(vary_params)

    @staticmethod 
    def get_param_types(vary_dict):
        return [vary_dict[key]['type'] for key in vary_dict.keys()]

    @staticmethod
    def build_param_space(vary_dict):
        space_dict = {}
        for key in vary_dict.keys():
            if 'values' in vary_dict[key].keys():
                # If values passed directly
                space_dict[key]=vary_dict[key]['values']
            else:
                # If bounds passed to make linear step size space
                a, b = vary_dict[key]['bounds']
                step_size = vary_dict[key]['step_size']
                N = int(np.abs(b-a)/step_size) + 1
                space_dict[key] = np.linspace(a, b, N)

        return space_dict
    
    def set_defaults(self, alpha = None, x_0 = None, params = None, forcing_params = None):
        self.default_alpha = alpha
        self.default_x_0 = x_0
        self.default_params = params
        self.default_forcing_params = forcing_params

    def get_values_for_run(self, param_types, keys, vals):
        alpha = self.default_alpha
        x_0 = self.default_x_0
        params = self.default_params
        forcing_params = self.default_forcing_params

        for param_type, key, val in zip(param_types, keys, vals):
            if param_type == 'alpha':
                alpha = val
            if param_type == 'x_0':
                x_0 = val
            if param_type == 'param' or param_type == 'parameter':
                params = copy.copy(self.default_params)
                params[key] = val
            if param_type == 'forcing':
                forcing_params = copy.copy(self.default_forcing_params)
                forcing_index = self.vary_params_input[key]['forcing_index']
                forcing_params[forcing_index][key] = val

        return self.alpha_fun(alpha), x_0, params, forcing_params
    
    @staticmethod
    def parse_storage(storage_settings):
        save_ts, save_ts_der = [0], [0] # defaults: save first dimension and its derivative
        if storage_settings is not None:
            if 'save_ts' in storage_settings.keys():
                save_ts = storage_settings['save_ts']
            if 'save_ts_der' in storage_settings.keys():
                save_ts_der = storage_settings['save_ts_der']
        return save_ts, save_ts_der

    @staticmethod
    def initialize_ds(storage_settings, param_space, t_vals):
        t_key = 't'
        x_comp_string = "x_component"

        save_ts, save_ts_der = ExperimentationHandler.parse_storage(storage_settings)
                
        x_names = [f"x{i}" for i in save_ts]
        [x_names.append(f"x{i}_der") for i in save_ts_der]
        n_x = len(x_names)

        full_keys = copy.copy(list(param_space.keys()))
        full_keys.append(t_key)
        full_keys.append(x_comp_string)

        full_size = [len(param) for param in list(param_space.values())]
        full_size.append(len(t_vals))
        full_size.append(n_x)

        coord_dict = copy.copy(param_space)
        coord_dict[t_key] = t_vals
        coord_dict[x_comp_string] = x_names
        ds = xr.Dataset({"x": (full_keys, np.zeros(full_size))},
            coords = coord_dict
        )
        
        return ds
    
    @staticmethod
    def update_ds(ds, results, param_names, param_vals, storage_settings):
        x = results['x']

        save_ts, save_ts_der = ExperimentationHandler.parse_storage(storage_settings)

        indices = {}
        for i in range(len(param_names)):
            indices[param_names[i]] = param_vals[i]

        i_x = 0
        for i_ts in save_ts:
            # All the values to save normally
            x_select = x[:, i_ts]

            indices['x_component'] = f"x{i_ts}"
            slices = [indices.get(dim, slice(None)) for dim in ds.dims]

            ds['x'].loc[tuple(slices)] = x_select
            i_x+=1
        
        for i_ts in save_ts_der:
            # All the values to save normally
            x_select = x[:, i_ts]

            indices['x_component'] = f"x{i_ts}_der"
            slices = [indices.get(dim, slice(None)) for dim in ds.dims]

            der_vals = np.squeeze(np.diff(x_select)/np.diff(results['t']))
            ds['x'].loc[tuple(slices)] = np.append(der_vals, float('nan'))
            i_x+=1

        if 'path' in storage_settings.keys() and 'autosave' in storage_settings.keys():
            if storage_settings['autosave'] is True:
                ds.to_netcdf(storage_settings['path'])

        return ds
    
    def run_experiment(self, storage_settings = None, verbose = True):
        ds = ExperimentationHandler.initialize_ds(storage_settings, self.param_space, self.bs.t_eval_vals_list)
        print(ds)
        param_space_vals = list(self.param_space.values())

        param_names = list(self.param_space.keys())
        param_types = self.param_types

        i_run = 0
        combinations = list(itertools.product(*param_space_vals))
        n_runs = len(combinations)
        for new_inputs in combinations:
            # Main loop
            alpha, x_0, params, forcing_params = self.get_values_for_run(param_types, param_names, new_inputs)
            
            if verbose:
                print(f"\n--- Experiment {i_run+1}/{n_runs} ---")
                print(f" ~ alpha: {np.round(alpha,3)}")
                print(f" ~ x_0: {x_0}")
                print(f" ~ params: {params}")
                print(f" ~ forcing: {forcing_params}")
            
            # TODO: this is maybe not so beautiful, change?
            self.f_object.params = params

            # TODO: don't initialize a new one everytime? Saves some comp. time
            solver = self.bs.initialize_solver(self.f_object.f, x_0, alpha, forcing_params = forcing_params) 

            # TODO: add kwargs here
            results = solver.run()

            ds = ExperimentationHandler.update_ds(ds, results, param_names, new_inputs, storage_settings)

            i_run += 1

        if 'path' in storage_settings.keys():
            ds.to_netcdf(storage_settings['path'])