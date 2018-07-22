import gym

import models as dqn_dv_models
import multiobjective as dqn_dv
from baselines.common.atari_wrappers import ScaledFloatFrame
import time
import tensorflow as tf

import eater

from baselines import logger

RUN_NAME = "run_" + str(time.time())[-4:]
PRIORITIES = ["charge", "collision", "clean"]

logger.configure(dir="logs/" + str(int(time.time())) + RUN_NAME, format_strs=['stdout', 'tensorboard'])

RENDER = "state"

def main():

    # env = gym.make("EaterNoFrameskipMultiObjective-v1")
    env = gym.make("EaterNoFrameskipMultiObjectiveRandomized-v1")
    # env = gym.make("EaterNoFrameskipMultiObjectiveRandomizedDeterministic-v1")
    # env = gym.make("EaterNoFrameskipSingleObjective-v1")
    env = ScaledFloatFrame(env)

    models = {}

    objectives = env.env.get_objectives()
    print(objectives)
    num_objectives = len(objectives)

    for o in objectives:
        model = dqn_dv_models.cnn_to_mlp_with_dv(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[128],
            dueling=True,
            num_dvs=1,
            reuse_conv=None,
        )
        models[o] = model

    act = dqn_dv.learn(
        env,
        q_func_dict=models,
        priorities=PRIORITIES,
        lr=1e-4,
        max_timesteps=1000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        print_freq=1,
        flat_decision_values=True,
        disable_dv=True,
    )
    act.save(RUN_NAME)
    env.close()

if __name__ == '__main__':
    main()

