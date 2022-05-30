import pandas as pd
from environments.i4c.env import Environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from timeit import default_timer as timer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np


def get_statistics(agent,
                   action_type,
                   num_forecast_steps,
                   building_type,
                   building_model_class,
                   eval_data_path,
                   sensor_noise=False,
                   forecast_noise=False,
                   disturbances=False,
                   resample_interval=None,
                   env_path=None,
                   use_price=False,
                   train_test='test',
                   seed=42):
    """This function is used to get the trajectory statistics from the trained models"""
    eval_data = pd.read_csv(eval_data_path, parse_dates=['Time'])

    print("running evaluation: ")

    # Only use months where heating is necessary !
    eval_data = eval_data[eval_data['Time'].dt.month.isin([1, 2, 3, 10, 11, 12])]

    years = eval_data['Time'].dt.year.unique()
    months = eval_data['Time'].dt.month.unique()

    print(months)

    start = timer()
    dataset_ids, comfort_deviation, Q_el_HP, Q_HP, t_amb, t_indoor, t_hp, R, original_rs, prices = (
    [], [], [], [], [], [], [], [], [], []) # using plain lists instead of pandas. Makes it 4 times faster...
    i = 0
    for year in years:
        for month in months:
            monthly_data = eval_data[eval_data['Time'].dt.year == year]
            monthly_data = monthly_data[monthly_data['Time'].dt.month == month]

            dataset_id = str(month) + '/' + str(year)

            print("running evaluation for: ", dataset_id)

            eval_env = Environment(data=monthly_data,
                                   action_type=action_type,
                                   building_type=building_type,
                                   building_model_class=building_model_class,
                                   evaluation=True,
                                   sensor_noise=sensor_noise,
                                   forecast_noise=forecast_noise,
                                   disturbances=disturbances,
                                   resample_interval=resample_interval,
                                   num_forecast_steps=num_forecast_steps,
                                   heatpump_type='aw',
                                   use_price=use_price,
                                   train_test=train_test,
                                   seed=seed)

            eval_env = DummyVecEnv([lambda: eval_env])
            # It is important to load the environmental statistics here as we use a rolling mean calculation !
            eval_env = VecNormalize.load(env_path, eval_env)       
            eval_env.training = False

            obs = eval_env.reset()
            done = False
            t = 0

            # Now let the agent interact with the environment and store everything that happens !
            while not done:
                old_obs = obs.copy()
                action = agent.predict(obs, deterministic=True)[0]
                obs, r, done, info = eval_env.step(action)
                original_r = eval_env.get_original_reward()
                if isinstance(info, list):
                    info = info[0]
                    action = action[0]
                    r = r[0]
                    original_r = original_r[0]

                if action_type == 'continuous':
                    info['electricity_used'] = info['electricity_used'][0]
                    action = (action[0] + 1) * 6  # rescale from [-1, 1] to [0, 12]
                    info['T_HP'] = info['T_HP'][0]
                else:
                    action = action * 2 #ToDo: This should happen automatically w.r.t. the size of the discrete action space
                # Store the current interaction in lists.
                dataset_ids.append(dataset_id);
                comfort_deviation.append(info['comfort_deviation'])
                Q_el_HP.append(info['electricity_used']);
                Q_HP.append(action);
                prices.append(info['price'])
                t_amb.append(info['outside_temp'])
                t_indoor.append(info['temperatures'][0]);
                t_hp.append(info['T_HP']);
                R.append(r)
                original_rs.append(original_r)
                t += 1
                i += 1
    # Dump list as .csv file 
    trajectory_stats = pd.DataFrame({'dataset_id': dataset_ids,
                                     'comfort_deviation': comfort_deviation,
                                     'Q_el_HP': Q_el_HP,
                                     'Q_HP': Q_HP,
                                     't_amb': t_amb,
                                     't_indoor': t_indoor,
                                     't_hp': t_hp,
                                     'R': R,
                                     'original_r': original_rs,
                                     'price': prices
                                     })
    print ('Elapsed time for creating stats %.2fs.' % (timer() - start))
    return trajectory_stats

class SaveOnBestTrainingRewardCallback_v2(BaseCallback):
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html for more information about Callbacks in SB3
    def __init__(self, check_freq, log_dir, action_type, use_price, num_forecast_steps, seed, building_model_class, building_type, sensor_noise=False, forecast_noise=False, disturbances=False, resample_interval=None, verbose=1, transfer_learning = False):
        super(SaveOnBestTrainingRewardCallback_v2, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        if transfer_learning:
            self.log_dir = self.log_dir + 'transfer_learning/'
        self.save_path = os.path.join(self.log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.action_type = action_type
        self.resample_interval = resample_interval
        self.use_price = use_price
        self.num_forecast_steps = num_forecast_steps
        self.summary = pd.DataFrame()
        self.seed = seed
        self.building_model_class = building_model_class
        self.building_type = building_type
        self.sensor_noise = sensor_noise
        self.forecast_noise = forecast_noise
        self.disturbances = disturbances

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls + 100) % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if self.n_calls > 1:  # start only after 1 episodes
                start = timer()
                # Save current training env to load running mean and std for evaluation
                self.training_env.save(self.log_dir + 'saved_env_temp.pkl')

                stats = get_statistics(self.model,
                                       action_type=self.action_type,
                                       num_forecast_steps=self.num_forecast_steps,
                                       eval_data_path='data/preprocessed_v2/eval_gains.csv',
                                       env_path=self.log_dir + 'saved_env_temp.pkl',
                                       resample_interval=self.resample_interval,
                                       use_price=self.use_price,
                                       building_model_class=self.building_model_class,
                                       building_type=self.building_type,
                                       seed = self.seed, #Passing seed, so eval env does not overwrite seed by calling np.random.seed() ... Life -2 days oO.
                                       sensor_noise = self.sensor_noise,
                                       forecast_noise = self.forecast_noise,
                                       disturbances = self.disturbances
                                       )
                print ('Elapsed time for evaluation %.2fs.' % (timer() - start))

                stats_summary = stats.mean(numeric_only=True)
                stats_summary['comfort_deviation_max'] = stats['comfort_deviation'].max()

                stats_summary['episode'] = len(x)
                print("summary is: ")
                print(stats_summary)
                self.summary = self.summary.append(stats_summary, ignore_index=True)
                self.summary.to_csv(self.log_dir + 'evaluation_summary.csv')

                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 stats_summary['original_r']))

                # New best model, saving agent here
                if stats_summary['original_r'] > self.best_mean_reward:
                    self.best_mean_reward = stats_summary['original_r']
                    if self.verbose > 0:
                        print("Saving new best model")
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
                    self.training_env.save(self.log_dir + 'saved_env_best.pkl')

        return True