import gym
import cleaner
import time, random

if __name__ == "__main__":

    eater = gym.make("CleanerNoFrameskipMultiObjectiveRandomized-v1")
    print(eater.observation_space.shape)
    fps = 0
    t = int(time.time())
    a = 0
    while(True):
        eater.step(a)
        if random.random() < 0.2:
            a = random.randint(0,3)
        s, r, d, _ = eater.step(a)
        if d:
            print("RESET")
            eater.reset()

        eater.render()

        fps += 1
        if int(time.time()) != t:
            print("FPS", fps)
            t = int(time.time())
            fps = 0