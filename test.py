import pyglet
from pyglet.gl import *
import gym




if __name__=="__main__":
    env = gym.make("CartPole-v0")
    env.reset()

    while True:
        state, reward, done, _ = env.step(env.action_space.sample())
        print(reward)
        if done:
            print("done", reward)
            env.reset()


