import gym

env = gym.make('CartPole-v1')
obs = env.reset()
done = False
while done is not True:
    env.render()
    action = input()    
    if action == "d":
        action = 0
    else:
        action = 1
    obs, r, done, _ = env.step(action)
    
    if done is True:
        print(r)
        break
