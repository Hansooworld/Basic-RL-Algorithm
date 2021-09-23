import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

class PPO(nn.Module):
    def __init__(self):
        super(PPO,self).__init__()
        self.data = []
        self.gamma = 0.98
        # GAE 논문에서 필요한 하이퍼파라미터 lambda
        self.lmbda = 0.95
        self.eps = 0.1
        self.k = 3

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0005)

    # policy (input:4 -> 256 -> 2 -> Softmax)
    def pi(self, x, softmax_dim = 0):
        # 단일 sample에 대해서는 softmax_dim = 0
        # batch들이 들어가게 되면 softmax_dim = 1
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob

    # value (input:4 -> 256 -> 1)
    def v(self,x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for item in self.data:
            s, a, r, s_prime, prob_a, done = item
            s_lst.append(s)
            # s는 numpy array형태인데, a와 r은 number라 shape을 맞춰주기 위해 [] 이용
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1 
            done_lst.append([done_mask])
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype = torch.float), torch.tensor(a_lst), torch.tensor(r_lst),\
                                              torch.tensor(s_prime_lst, dtype = torch.float),torch.tensor(done_lst, dtype = torch.float),torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.k):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype = torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            # td_target.detach()를 해줘야함
            # detach()의 의미는 gradient 를 계산하지 않는 것 -> 즉, 학습하지 않는 것
            # td_target과 v를 둘 다 학습하는 것이 아니라 v만 학습하는 것! 
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(td_target.detach(),self.v(s))

            self.optimizer.zero_grad() 
            # loss가 scalar값이 아닐 때 mean() 
            loss.mean().backward()
            self.optimizer.step()
        wandb.log({'loss':loss})

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    # T : 몇 step 동안 data를 모으고, policy update를 할지 
    T = 20
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                
                score += r
                if done:
                    break
            
            model.train()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

if __name__ == '__main__':
    main()
