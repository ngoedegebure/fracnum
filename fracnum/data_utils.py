import xarray as xr
import copy
import numpy as np
import os

class DataHandler():
    def __init__(self, storage_settings, param_space, t_vals, t_key = 't', x_key = 'x_dim'):
        self.storage_settings = storage_settings
        self.param_space = param_space
        self.t_key, self.x_key = t_key, x_key
        self.save_ts, self.save_ts_der = DataHandler.parse_storage(storage_settings)

        self.ds = self.initialize_ds(param_space, t_vals)

    @staticmethod
    def parse_storage(storage_settings):
        save_ts, save_ts_der = [0], [0] # defaults: save first dimension and its derivative
        if storage_settings is not None:
            if 'save_ts' in storage_settings.keys():
                save_ts = storage_settings['save_ts']
            if 'save_ts_der' in storage_settings.keys():
                save_ts_der = storage_settings['save_ts_der']
        return save_ts, save_ts_der
    
    def initialize_ds(self, param_space, t_vals):
        save_ts, save_ts_der = self.save_ts, self.save_ts_der
        t_key, x_comp_string = self.t_key, self.x_key

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
        self.ds = xr.Dataset({"x": (full_keys, np.zeros(full_size))},
            coords = coord_dict
        )
        
        return self.ds
    
    def update_ds(self, param_vals, results):
        x = results['x']
        param_names = list(self.param_space.keys())
        save_ts, save_ts_der = self.save_ts, self.save_ts_der

        indices = {}
        for i in range(len(param_names)):
            indices[param_names[i]] = param_vals[i]

        i_x = 0
        for i_ts in save_ts:
            # All the values to save normally
            x_select = x[:, i_ts]

            indices[self.x_key] = f"x{i_ts}"
            slices = [indices.get(dim, slice(None)) for dim in self.ds.dims]
            
            self.ds['x'].loc[tuple(slices)] = x_select
            i_x+=1
        
        for i_ts in save_ts_der:
            # All the values to save normally
            x_select = x[:, i_ts]

            indices[self.x_key] = f"x{i_ts}_der"
            slices = [indices.get(dim, slice(None)) for dim in self.ds.dims]

            der_vals = np.squeeze(np.diff(x_select)/np.diff(results['t']))
            self.ds['x'].loc[tuple(slices)] = np.append(der_vals, float('nan'))
            i_x+=1

        if 'autosave' in self.storage_settings.keys():
            if self.storage_settings['autosave'] is True:
                self.store_if_needed()

        return self.ds
    
    @staticmethod 
    def folder_from_path(filepath):
        slash_indices = [filepath[i] == "/" for i in range(len(filepath))]
        last_slash = int(np.max(np.asarray(slash_indices).nonzero()))
        return filepath[:last_slash]
    
    def store_if_needed(self):
        if 'path' in self.storage_settings.keys():
            path = self.storage_settings['path']
            folder_path = DataHandler.folder_from_path(path)
            try:
                os.makedirs(folder_path)
            except FileExistsError:
                # Directory already exists
                pass
            self.ds.to_netcdf(path)