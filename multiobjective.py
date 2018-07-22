import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import gym
import baselines.common.tf_util as U
from baselines import logger
from baselines.common.schedules import LinearSchedule
import build_graph as dqn_dv
from multiobjective_replay_buffer import MultiObjectiveReplayBuffer


GPU_FRACTION = 0.1


class MultiActWrapper(object):
    def __init__(self, act, act_params, priorities, actions_num, flat_dv=True, weights=None, disable_dv=False):
        self.actions_num = actions_num
        self.flat_decision_values = flat_dv
        self._act = act
        self._act_params = act_params
        self.priorities = priorities
        self.disable_dv = disable_dv
        if weights is None:
            weights = dict([(ob, 1.0) for ob in priorities])

        self.weights = weights

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params, priorities, actions_num, flat_dv = cloudpickle.load(f)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRACTION)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.__enter__()
        
        act = dict()
        for ob in priorities:
            act[ob] = dqn_dv.build_act(**act_params[ob])

        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return MultiActWrapper(act, act_params, priorities, actions_num, flat_dv)

    def __call__(self, *args, **kwargs):
        decision_values = dict()
        selected_dvs = dict()
        q_vals = dict()
        extra_indicators = []

        for ob in self.priorities:
            action, _q_val, _dv = self._act[ob](*args, **kwargs)

            q_val_scaled = _q_val - _q_val.min()
            q_val_scaled = q_val_scaled / q_val_scaled.max()

            _dv_decision = np.squeeze(_dv)

            decision_values[ob] = _dv_decision
            selected_dvs[ob] = decision_values[ob]
            q_vals[ob] = q_val_scaled
            extra_indicators.append(decision_values[ob])

        random_q_vals = np.random.rand(self.actions_num)

        if self.flat_decision_values:
            if self.disable_dv:
                policies = [q_vals[ob] * self.weights[ob] for ob in self.priorities]
            else:  # subsumption
                policies = [q_vals[ob] * decision_values[ob] * self.weights[ob] for ob in self.priorities]

            q_vals_sum = random_q_vals * 0.01 + np.sum(policies, axis=0)
        else:
            q_vals_sum = random_q_vals * 0.01
            for ob in reversed(self.priorities):
                q_vals_sum = q_vals[ob] * (decision_values[ob]) + q_vals_sum * (1.0 - decision_values[ob])

        action = np.argmax(q_vals_sum)

        return action, q_vals_sum, decision_values, selected_dvs, extra_indicators

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params, self.priorities, self.actions_num, self.flat_decision_values), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return MultiActWrapper.load(path)


def learn(env,
          q_func_dict,
          priorities,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          flat_decision_values=False,
          disable_dv=False,
          callback=None):

    # Create all the functions necessary to train the model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # sess = tf.Session()
    sess.__enter__()

    executor = ThreadPoolExecutor(max_workers=3)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    objectives = env.env.get_objectives()

    act = {}
    train = {}
    update_target = {}
    debug = {}
    act_params = {}

    for ob in priorities:
        q_func = q_func_dict[ob]

        act[ob], train[ob], update_target[ob], debug[ob] = dqn_dv.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            double_q=True,
            grad_norm_clipping=10,
            scope=ob
        )

        act_params[ob] = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
            'scope': ob,
        }

    multi_act = MultiActWrapper(act, act_params, priorities, env.action_space.n, disable_dv=disable_dv)

    replay_buffer = MultiObjectiveReplayBuffer(buffer_size, objectives)
    beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    [update_target_fn() for update_target_fn in update_target.values()]

    episode_rewards = [0.0]
    objective_rewards = dict((k, [0.0]) for k in priorities)
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}

            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.

            action, q_vals_sum, dvs, selected_dvs, extra_indicators = multi_act(np.array(obs)[None], update_eps=update_eps, **kwargs)

            if isinstance(env.action_space, gym.spaces.MultiBinary):
                env_action = np.zeros(env.action_space.n)
                env_action[action] = 1
            else:
                env_action = action
            reset = False

            env.env.set_extra_indicators(extra_indicators)
            new_obs, rew, done, _ = env.step(env_action)

            rew_sum = sum(rew.values())
            dv_rew = dict([(k, abs(v)) for k, v in rew.items()])

            rew_with_bias = dict([(k, v + 0.1*rew_sum) for k, v in rew.items()])

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, selected_dvs, rew_with_bias, dv_rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += np.sum(list(rew.values()))
            for ob in priorities:
                objective_rewards[ob][-1] += rew[ob]

            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                for ob in priorities:
                    objective_rewards[ob].append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                obses_t, actions, dvs, rewards, dv_rewards, obses_tp1, dones = replay_buffer.sample(batch_size)

                weights, batch_idxes = {}, {}
                for ob in priorities:
                    weights[ob], batch_idxes[ob] = np.ones_like(rewards[ob]), None

                train_threads = []
                td_errors = {}

                def train_wrap(ob, session, args):
                    with session.as_default():
                        td_error = train[ob](*args)
                        return td_error

                for ob in priorities:
                    args = (obses_t, actions, dvs[ob], rewards[ob], dv_rewards[ob], obses_tp1, dones, weights[ob])
                    train_wrap(ob, tf.get_default_session(), args)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                for ob in priorities:
                    update_target[ob]()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_5ep_reward = round(np.mean(episode_rewards[-6:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 5 episode reward", mean_5ep_reward)
                for ob in priorities:
                    obj_mean = round(np.mean(objective_rewards[ob][-6:-1]), 1)
                    logger.record_tabular(ob + " mean 5ep", obj_mean)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    return multi_act
