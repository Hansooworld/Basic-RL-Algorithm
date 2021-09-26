import gym
import pybulletgym
import numpy as np
# 'AntPyBulletEnv-v0'
def main():
    env = gym.make('AntPyBulletEnv-v0')
    gym.logger.set_level(40)
    dim_of_states = env.observation_space.shape[0]
    dim_of_actions = env.action_space.shape[0]
    print(dim_of_states,dim_of_actions)
    env.render()
    s = env.reset()
    for i in range(100):
        env.render()
        b = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        obs = env.step(b)
        # print(obs)
main()

