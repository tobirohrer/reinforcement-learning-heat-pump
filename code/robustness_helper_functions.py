import numpy as np

def randomize_forecast(forecast):
    for i in range(len(forecast)):
        std = 1.1 - 0.9 ** i
        forecast[i] = forecast[i] + np.random.normal(0, std)
    return forecast

def randomize_building_params(building_params):
    """Apply Domain Randomization"""
    params = building_params.copy()
    params['H_ve'] = building_params['H_ve'] * np.random.normal(1, 0.2)
    params['H_tr'] = building_params['H_tr'] * np.random.normal(1, 0.2)
    params['area_floor'] = building_params['area_floor'] * np.random.normal(1, 0.2)
    params['height_room'] = building_params['height_room'] * np.random.normal(1, 0.2)
    params['c_bldg'] = building_params['c_bldg'] * np.random.normal(1, 0.2)
    return params