import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import deque
import pyglet

# Instruction:
# 1. In this model, we constructed a 3-layers neuron network class for the input state.
# 2. To ensure the stability of the model, we used two network instances including evlt_net for training and 
# tgt_net net work as the target Q-value. At the start of each time for replaying, we imported the parameters in 
# evlt_net to tgt_net, the latter will not backward but only for the evaluation.  
# 3. Acturally, there are two methods to replay the parameters of network: iteration by for-loop and input 
# the matrix of batch size of Q-values. However, it is more obvious for for-loop to perform the difference between two 
# categories of network during iteration. Additionally, for-loop is more convenient for us to watch the change
# in each step during replaing.
# 4. After 1000 epoches training, the model can achieve more than 200 average score.

# Reference:
# 1. COMP532 module slide and the code from professor Angrosh
# 2. https://blog.csdn.net/hustqb/article/details/78648556 "the reason why DQN never learning"
# 3. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?highlight=dqn
# 4. https://stackoverflow.com/questions/49840892/deep-q-network-is-not-learning
# 5. https://github.com/Guillem96/space-invaders-jax/tree/main/spaceinv

# Code:

# construct a neuron network (prepare for step1, step3.2 and 3.3)
class DQN(nn.Module):
    def __init__(self, s_space, a_space) -> None:

        # inherit from DQN class in pytorch
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(s_space, 360)
        self.fc2 = nn.Linear(360, 360)
        self.fc3 = nn.Linear(360, a_space)

    # NN operation architecture
    def forward(self, input):
        out = self.fc1(input)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

# define the action selection according to epsilon using neuron network (prepare for step3.2)
def select(net, epsilon, env, state):
    actions = net(th.Tensor(state)).data.numpy()
    # randomly select an action if not greedy
    if(np.random.rand() <= epsilon):
        a = env.action_space.sample()
        Q_value = actions[0][a]
        return a, Q_value
    # select the maximum reward action by NN and the given state if greedy
    else:
        a = np.argmax(actions)
        Q_value = actions[0][a]
        return a, Q_value


# using loss function to improve neuron network (prepare for step3.3)

def al_backprbgt(evl_net, tgt_net, store, batch_size, gamma, optimal):

    # step 1: define loss function and optimizer

    judge_F = nn.MSELoss()
    optimal = optimal

    # step 2: import memory sample

    materials = random.sample(store, batch_size)

    # step3: calculation loss value

    tgt_net.load_state_dict(evl_net.state_dict())   # reference 3

    for t in materials: # reference 1,3 and 5, considering which method to replay net parameters

        # step3.1 calculate tgt_Q_value according to greedy next action  # reference 1
        if(t[4] == True):
            tgt = t[3]
        else:
            tgt = t[3] + gamma * max(tgt_net(th.Tensor(t[2])).data.tolist()[0])

        tgt_Q_value = th.Tensor([tgt]).unsqueeze(1)

        # step3.2 calculate evlt_Q_value
        a_v = th.LongTensor([t[1]]).unsqueeze(1)
        evlt_Q_value = evl_net(th.Tensor(t[0])).gather(1,a_v)

        # step4: back propagation with loss value
        optimal.zero_grad()
        loss = judge_F(evlt_Q_value, tgt_Q_value)
        # print(loss)
        loss.backward()
        optimal.step()

# show the statistic data (average score and average Q_value)
def out_image(epi, avg_scores, avg_Q_values):
    plt.plot(epi, avg_scores, label = "average score")
    plt.xlabel('episodes')
    plt.ylabel('average score')
    plt.gcf()
    plt.show()

    plt.plot(epi, avg_Q_values, label = "average Q-value")
    plt.xlabel('episodes')
    plt.ylabel('average Q-value')
    plt.gcf()
    plt.show()

# training progression

# step 1: set supreme parameters

episode = 1000
epsilon = 0.1
min_epsilon = 0.01
dr = 0.995
gamma = 0.9
lr = 0.0001 
batch_size = 128    # reference 2 and 4, consider about reducing learning rate and expend the batch size
memory_store = deque(maxlen=1000)

epi = []
avg_scores = []
avg_Q_values = []

score = 0
Q_values = 0

# step 2: define game category, NN instances, optimizer, and associated states and actions
env = gym.make("CartPole-v1")
s_space = env.observation_space.shape[0]
a_space = env.action_space.n

evl_net = DQN(s_space, a_space)
tgt_net = DQN(s_space, a_space)
optimal = th.optim.Adam(evl_net.parameters(),lr=lr)

# step 3: trainning
for e in range(0, episode):

    Q_value = 0
    avg_Q_value = 0
    avs = 0
    # step3.1: at the start of each episode, the current result should be refreshed

    # set initial state matrix
    s = env.reset().reshape(-1, s_space)
    # step3.2: iterate the state and action
    for run in range(500):

        env.render()
        # select action and get the next state according to current state "s"
        a, Q_value = select(evl_net, epsilon, env, s)
        obs, reward, done, _ = env.step(a)
        if(done == True):
            reward = -10.0
        
        next_s = obs.reshape(-1, s_space)

        # save the record of each iteration, we use the latest 1000 steps instead of the more previous
        memory_store.append((s, a, next_s, reward, done))
        s = next_s

        score += 1
        Q_values += Q_value

        if(done == True):
            avs = score / (e+1)
            avg_Q_value = Q_values / (run+1)
            print("episode:", e+1, "score:", run+1, "avarage score: {:.2f}".format(avs), "epsilon: {:.2}".format(epsilon))
            break

        if(run == 499):
            print("episode:", e+1, "score:", run+1, "avarage score:", avs)

    # step3.3 whenever the episode reach the integer time of batch size, 
    # we should backward to improve the NN
    if(len(memory_store) > batch_size):
        al_backprbgt(evl_net, tgt_net, memory_store, batch_size, gamma, optimal) # here we need a backprbgt function to backward
        if(epsilon > min_epsilon):
            epsilon = epsilon * dr

    epi.append(e+1)
    avg_scores.append(avs)
    avg_Q_values.append(avg_Q_value)

# step 4 output the statistic data by two images after finishing training
out_image(epi, avg_scores, avg_Q_values)
env.close()
        