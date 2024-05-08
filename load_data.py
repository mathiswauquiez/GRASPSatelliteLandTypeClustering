import netCDF4 as nc
import numpy as np
import pandas as pd

def normalize(data):
  return (data - data.mean()) / data.std()

def get_data(selection, clustering_variables, filename='data/GRASP_POLDER_L3_climatological_S1.1degree.nc', fill_value=-1000):
    """ Loads the data from the GRASP netCDF file and returns a pandas dataframe with the selected variables and the latitude and longitude """
    
    original_dataset = nc.Dataset(filename)
    img_shape = original_dataset.variables['DHR1020'][:].shape # we will use this to flatten the variables and get the shape back later

    # mask creation, to select the land or the sea or both
    if selection == 'sea':
        mask = (original_dataset.variables['LandPercentage'][:] == 100).filled(True)
    elif selection == 'land':
        mask = (original_dataset.variables['LandPercentage'][:] != 100).filled(True)
    else:
        mask = original_dataset.variables['DHR1020'][:].mask

    # flattening the variables / masking
    data = {}
    for var in original_dataset.variables:
        if original_dataset[var][:].shape == img_shape: # we only want the variables with the same shape as the image
            data[var] = original_dataset[var][:].flatten()
            data[var][mask.flatten()] = fill_value

    # creating the dataframe
    df = pd.DataFrame(data)
    #del data, original_dataset # free some memory

    # we explicitly set the fill_value to nan
    df = df.replace(fill_value, np.nan)
    df = df[clustering_variables + ['Latitude', 'Longitude']]
    df.dropna(inplace=True)
    data = df[clustering_variables].values
    data = normalize(data)
    return data, df