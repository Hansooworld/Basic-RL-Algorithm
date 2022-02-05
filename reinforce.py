import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.data = []
        self.gamma = 0.99
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,2)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0003)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 0)
        return x

    def put_data(self,item):
        self.data.append(item)

    def train(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + R * self.gamma
            loss = -prob * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    global pi
    env = gym.make('CartPole-v1')
    pi = Policy()
    
    for n_episode in range(1000):
        avg_t = 0
        # 원래 obs는 numpy 형태
        obs = env.reset()
        for t in range(600):
            obs = torch.tensor(obs,dtype = torch.float)
            # out은 확률 형태
            out = pi(obs)
            m = Categorical(out)
            # action은 텐서 형태
            action = m.sample()
            obs, r, done, _ = env.step(action.item())
            pi.put_data((r,torch.log(out[action])))
            if done:
                break
            avg_t += 1
        if avg_t < 450:
            pi.train()
        else:
            print("Avg timestep:{}".format(avg_t))
            break

        if n_episode % 20 == 0 and n_episode != 0:
            print("num of episode:{}, Avg timestep:{}".format(n_episode,avg_t))
    
    obs = env.reset()
    done = False
    while done is not True:
        out = pi(torch.tensor(obs,dtype=torch.float))
        m = Categorical(out)
        action = m.sample()
        obs, r, done, _ = env.step(action.item())
        print(action)
        env.render(mode='human')
        
        if done is True:
            env.close()

main()