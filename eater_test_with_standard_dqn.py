import gym

from baselines import deepq
from baselines.common.atari_wrappers import ScaledFloatFrame

import eater

import time
from baselines import logger

RUN_NAME = "randomized_single" + str(time.time())[-4:]

logger.configure(dir="logs/" + str(int(time.time())) + RUN_NAME, format_strs=['stdout', 'tensorboard'])


RENDER = "state"

def main():
    # env = gym.make("EaterNoFrameskipSingleObjectiveRandomized-v1")
    env = gym.make("EaterNoFrameskipSingleObjective-v1")
    env = ScaledFloatFrame(env)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[128],
        dueling=True
    )


    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=1000000,
        buffer_size=10000,
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        print_freq=1,
    )
    act.save("original_dqn_model.pkl")
    env.close()

if __name__ == '__main__':
    main()

