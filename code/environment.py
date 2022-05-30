from abc import ABC

import numpy as np
import sys

import pandas as pd
import gym
import matplotlib.pyplot as plt
import random
import datetime
import logging
import sys
from robustness_helper_functions import randomize_forecast, randomize_building_params
######
# Start of imports of the simulation framework which is not open source
######
sys.path.append('../../../i4c/')
import model_hvac
import model_buildings
import simulator
# CAUTION & ToDo ! The paths were changed from the original i4c repo (Because of test with train test split for buildings). Please change paths back to original building files.
from data.buildings.test.i4c_building import i4c as i4c_test
from data.buildings.test.sfh_58_68_geg import sfh_58_68_geg as sfh_58_68_geg_test
from data.buildings.test.sfh_84_94_soc import sfh_84_94_soc as sfh_84_94_soc_test
from data.buildings.test.sfh_84_94_ad_ref import sfh_84_94_ad_ref as sfh_84_94_ad_ref_test
from data.buildings.test.sfh_84_94_ref import sfh_84_94_ref as sfh_84_94_ref_test
from data.buildings.train.i4c_building import i4c as i4c_train
from data.buildings.train.sfh_58_68_geg import sfh_58_68_geg as sfh_58_68_geg_train
from data.buildings.train.sfh_84_94_soc import sfh_84_94_soc as sfh_84_94_soc_train

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

MDOT_HP = 0.25  # fixed mass flow rate of heat pump
MAX_FORECAST_STEPS = 96  # Used to make datasets same length during eval

def label_datasets(row):
    year = row['Time'].year
    if row['Time'].month in [1, 2, 3]:
        return 'jan_mar_' + str(year)
    if row['Time'].month in [10, 11, 12]:
        return 'oct_dec_' + str(year)


def preprocess_data(raw_data, resample_interval, evaluation):
    """
    G(i) [W/m2] - Global in-plane irradiance (if radiation components are not requested
    Gb(i) [W/m2] - Direct in-plane irradiance (if radiation components are requested)
    Gd(i) [W/m2] - Diffuse in-plane irradiance (if radiation components are requested)
    Gr(i) [W/m2] - Reflected in-plane irradiance (if radiation components are requested)
    """
    data = raw_data
    if resample_interval is not None:
        data = data.set_index('Time').resample(resample_interval).interpolate(method='linear').reset_index()

    data['dataset_id'] = data.apply(lambda row: label_datasets(row), axis=1)

    if evaluation:
        if len(data['dataset_id'].unique()) > 1:
            raise Exception('Evaluation only works on single months !')
    return data


def select_random_data(all_data):
    # In case of evaluation, we only look at one season. We just take this one :-)
    datasets = all_data['dataset_id'].unique()
    datasets = list(filter(None,datasets)) # remove None from list
    random_dataset = np.random.choice(datasets)
    print('selecting random data: ', random_dataset)

    return all_data[all_data['dataset_id'] == random_dataset]


def get_simulator(heatpump_type, building_type, building_model_class, resample_interval, train_test, randomize_building=False):
    interval = 3600  # in sec
    if resample_interval == '900S':
        interval = 900

    # CAUTION & ToDo ! The paths were changed from the original i4c repo (Because of test with train test split for buildings). Please change paths back to original building files.
    building_config_map = {'train': {'i4c': i4c_train,
                                     '84': sfh_84_94_soc_train,
                                     '58': sfh_58_68_geg_train},
                           'test': {'i4c': i4c_test,
                                    '84': sfh_84_94_soc_test,
                                    '58': sfh_58_68_geg_test,
                                    '84_ad_ref': sfh_84_94_ad_ref_test,
                                    '84_ref': sfh_84_94_ref_test}}
    building_parameters = building_config_map[train_test][building_type]
    # In case of domain randomization is to be applied
    if randomize_building == True:
        building_parameters = randomize_building_params(building_parameters)

    print('initializing building with parameters:')
    print(building_parameters)

    building_model = model_buildings.Building(params=building_parameters,
                                              T_room_set_lower=21,
                                              mdot_hp=MDOT_HP,
                                              method=building_model_class
                                              )

    hp_model = None
    if heatpump_type == 'vitocal':
        hp_model = model_hvac.Heatpump_Vitocal(mdot_HP=MDOT_HP)
    if heatpump_type == 'aw':
        hp_model = model_hvac.Heatpump_AW(mdot_HP=MDOT_HP)  # air water heat pump
    return simulator.Model_simulator(bldg_model=building_model,
                                     hp_model=hp_model,
                                     timestep=interval)

def calculate_reward(electricity_used, comfort_deviation, action, price, use_price):
    if action == -1:
        cost = 0
    else:
        cost = electricity_used
        if use_price:
            cost = cost * price

    if comfort_deviation != 0:
        comfort_penalty = comfort_deviation
    else:
        comfort_penalty = 0

    if use_price:
        return np.float64(-0.07 * cost + -1 * comfort_penalty)
    else:
        return np.float64(-1 * cost - 0.5 * comfort_penalty)


def get_observation_dimension(forecast_downsample_rate, use_price, num_forecast_steps):
    """Calculate observation/state dimension which is required for openAI Gym"""
    temperature_state_dimension = 2
    data_per_forecast_step = 1
    if use_price:
        data_per_forecast_step = 2
    return int(temperature_state_dimension + data_per_forecast_step/forecast_downsample_rate * num_forecast_steps)


class Environment(gym.Env):
    def __init__(self, data_path=None,
                 action_type='continuous', # Environment supports continuous and discrete action types
                 evaluation=False,  # Build one environment build on the data given. Dont sample at random.
                 data=None,  # If evaluation is True, proide dedicated Dataframe, not file path.
                 heatpump_type='aw',
                 building_type='i4c', #Define Building type. Caution, types are named different than in the master thesis. (i4c=efficient, 84=old)
                 train_test='test',  # ToDo: Change to train... makes more sense.
                 building_model_class=None, # 2R2C or 4R3C? Both are supported
                 sensor_noise=False, #If set to True, noise will be added to the sensor meassurements
                 forecast_noise=False, #If set to True, noise will be added to the forecast
                 randomize_building=False, #If set to True, Domain Randomization will be applied during training
                 disturbances=False, #If set to True, data from solar irradiation and random opening of windows will be simulated during training
                 forecast_downsampling_rate=4, #Downsampling of the forecast, if a really long forecast is included which would cause a too big state space
                 debug=False,
                 num_forecast_steps=4, #How many forecast steps to include into the state
                 seed=42,
                 resample_interval=None, #How many seconds correspond to one time step ?
                 use_price=False # True if Demand Response scenario
                 ):
        self.forecast_downsampling_rate = forecast_downsampling_rate
        self.sensor_noise = sensor_noise
        self.forecast_noise = forecast_noise
        self.disturbances = disturbances
        self.randomize_building = randomize_building
        self.evaluation = evaluation
        self.building_model_class = building_model_class
        self.building_type = building_type
        np.random.seed(seed)
        random.seed(seed)
        self.simulator = get_simulator(heatpump_type, self.building_type, self.building_model_class, resample_interval,
                                       train_test)
        # all_weather_data contains information from multiple seasons. During training we draw a season at random for each episode
        if evaluation:
            self.all_data = preprocess_data(data,
                                            resample_interval,
                                            self.evaluation)
            self.max_episode_length = len(self.all_data) - MAX_FORECAST_STEPS
            print("Evaluating: Episode length is {} timesteps".format(self.max_episode_length))
        else:
            self.all_data = preprocess_data(pd.read_csv(data_path, parse_dates=['Time']),
                                            resample_interval,
                                            self.evaluation)
            self.max_episode_length = 720 * 4 # ToDo: This ony applies if resample interval is 900s

        # Current weather data used for training or evaluation is stored in weather_data
        self.episode_data = None
        self.num_forecast_steps = num_forecast_steps
        self.step_count = 0
        self.start_timestep = None
        self.temp_state = None
        self.debug = debug
        self.use_price = use_price
        self.action_type = action_type
        if action_type == 'discrete':
            self.action_space = gym.spaces.Discrete(4)
        if action_type == 'continuous':
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32
                                               )

        self.observation_space = gym.spaces.Box(shape=(get_observation_dimension(self.forecast_downsampling_rate,
                                                                                 self.use_price,
                                                                                 self.num_forecast_steps),),
                                                low=-np.inf,
                                                high=np.inf,
                                                dtype=np.float64)

    def step(self, action):
        """This function defines what happens in one ineraction step between agent and environment"""
        
        # action represents Q_dot_HP
        if self.action_type == 'discrete':
            # ToDo: also rescale to [-1,1] for discrete actions !
            q_dot_hp = action * 4000
        if self.action_type == 'continuous':
            # scale to range [-1,1]
            q_dot_hp = (action + 1) * 6000  # rescale from [-1, 1] to [0, 12000 W]
        # u is the controll action for the simulation framework, which is the supply temperature in this case !    
        u = q_dot_hp / (MDOT_HP * 4181) + self.temp_state['T_hp_ret']

        data_at_this_timestep = self.episode_data.iloc[self.start_timestep + self.step_count]

        # Gains set to 5 Watt per m^2 (according to DIN4108-6) 
        gains = self.simulator.bldg_model.params['area_floor'] * 5

        # Disturbances are included for future works !
        if self.disturbances == True:
            gains = data_at_this_timestep['gains']
            if data_at_this_timestep['blow_air'] == True:
                self.temp_state['T_room'] = self.temp_state['T_room'] - 1


        p = {'T_amb': data_at_this_timestep['T_amb'],
             'Qdot_gains': gains}  # Setting all gains statically

        results = self.simulator.get_next_state(self.temp_state,
                                                u,
                                                p)

        self.temp_state = results['state']
        self.step_count = self.step_count + 1

        comfort_deviation_max = results['cost']['dev_neg_max'] + results['cost']['dev_pos_max']
        comfort_deviation_sum = results['cost']['dev_neg_sum'] + results['cost']['dev_pos_sum']

        reward = calculate_reward(results['cost']['E_el'] / 1000,
                                  comfort_deviation_max,
                                  action,
                                  data_at_this_timestep['EEX'],
                                  self.use_price)
        done = self.is_done()

        return self.get_state(), reward, done, {'electricity_used': results['cost']['E_el'],
                                                'comfort_deviation': comfort_deviation_max,
                                                'comfort_deviation_sum': comfort_deviation_sum,
                                                'temperatures': np.array(list(self.temp_state.values())),
                                                'T_HP': u,
                                                'outside_temp': data_at_this_timestep['T_amb'],
                                                'price': data_at_this_timestep['EEX'],
                                                }

    def reset(self):
        self.step_count = 0
        # Select random weather profile
        self.episode_data = select_random_data(self.all_data)

        if self.evaluation:
            self.start_timestep = 0
        else:
            # Select random start date in the selected weather profile
            self.start_timestep = int(
                np.random.uniform(0, len(self.episode_data) - self.max_episode_length - self.num_forecast_steps))

        self.temp_state = {key: 21 for key in self.simulator.bldg_model.state_keys}
        self.temp_state['T_hp_ret'] = 23
        
        # get new simulator with new random building params
        # This only applies for domain randomization !
        if self.randomize_building == True: 
            self.simulator = get_simulator('aw', self.building_type, self.building_model_class, '900S',
                                           'test',  randomize_building=True)

        return self.get_state()

    def get_state(self):
        forecast_data = self.episode_data[self.start_timestep + self.step_count:
                                          self.start_timestep + self.step_count + self.num_forecast_steps]
        # Downsample forecast to reduce dimensionality
        forecast_data = forecast_data[::self.forecast_downsampling_rate]

        weather_forecast = forecast_data['T_amb'].copy()

        price_forecast = []
        if self.use_price:
            price_forecast = forecast_data['EEX']

        temp_state = np.array([self.temp_state['T_room'], self.temp_state['T_hp_ret']])
        if self.sensor_noise:
            temp_state = np.array([np.random.normal(self.temp_state['T_room'], 0.5),
                                   np.random.normal(self.temp_state['T_hp_ret'], 0.1)])
        if self.forecast_noise:
            weather_forecast = randomize_forecast(weather_forecast.values)
        return np.concatenate((temp_state,
                               weather_forecast,
                               price_forecast),
                              axis=0)

    def is_done(self):
        if self.step_count >= self.max_episode_length:
            return True
        return False
