import gym
import models as subsumption_models
import multiobjective as subsumption
from baselines.common.atari_wrappers import ScaledFloatFrame
import time
import numpy as np
import eater
import sys


def run_dqn(model, priorities, weights, disable_dvs, episodes_count):
    # env = gym.make("PongNoFrameskip-v4")
    # env = gym.make("EaterNoFrameskipMultiObjective-v1")
    env = gym.make("EaterNoFrameskipMultiObjectiveRandomizedDeterministic-v1")
    # env = gym.make("EaterNoFrameskipSingleObjective-v1")
    env = ScaledFloatFrame(env)

    print("WEIGHTS: ", weights)

    objectives = env.env.get_objectives()
    print(objectives)

    act = subsumption.load(model)
    act.flat_dvs = True
    act.priorities = priorities
    act.weights = weights
    act.disable_dvs = disable_dvs
    print("setting priorities to: ", act.priorities)

    all_rews = []
    episodes = 0
    while episodes < episodes_count:
        obs, done = env.reset(), False
        episode_rew = np.array([0.0, 0.0, 0.0])
        while not done:
            action, q_vals_sum, dvs, selected_dvs, extra_indicators = act(obs[None])
            env.env.set_extra_indicators(extra_indicators)
            obs, rew, done, _ = env.step(action)
            r = np.array([rew['collision'], rew['clean'], rew['charge']])
            episode_rew += r
        print("[" + str(episodes) + "]Episode reward", episode_rew)
        all_rews.append(episode_rew)
        episodes += 1

    score = np.mean(np.array(all_rews), axis=0)
    print("TOTAL SCORE:")
    print("collision, clean, charge")
    print(score[0], score[1], score[2])

    env.close()

    return score

if __name__ == '__main__':
    model, weights, episodes_count = sys.argv[1], sys.argv[2:5], 10
    priorities = ['collision', 'clean', 'charge']
    W = dict([(x[0], float(x[1])) for x in zip(priorities, weights)])
    run_dqn(model, ['collision', 'clean', 'charge'], W, False, episodes_count)

