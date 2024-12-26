import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment import Env
import rclpy
from rclpy.node import Node
import csv
import copy
import time

# Define hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQUENCY = 2
BUFFER_SIZE = 1000000
BATCH_SIZE = 256

# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for storing experiences
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim,max_action):
        self.actor = Actor(state_dim, action_dim,max_action)
        self.actor_target = Actor(state_dim, action_dim,max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_target_1 = Critic(state_dim, action_dim)
        self.critic_target_2 = Critic(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=LEARNING_RATE)

        
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().cpu().numpy().flatten()




    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample()

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)


        # Compute target Q-values
        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = reward + (1 - done) * GAMMA * torch.min(target_Q1, target_Q2)

        # Optimize Critic 1
        current_Q1 = self.critic_1(state, action)
        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q.detach())
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        # Optimize Critic 2
        current_Q2 = self.critic_2(state, action)
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q.detach())
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        torch.autograd.set_detect_anomaly(True)

        # Delayed policy updates
        if self.total_it % POLICY_FREQUENCY == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        self.total_it += 1

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, filename):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_optimizer_1.state_dict(), filename + "_critic_optimizer_1")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_optimizer_2.state_dict(), filename + "_critic_optimizer_2")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_optimizer_1.load_state_dict(torch.load(filename + "_critic_optimizer_1"))
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
        self.critic_optimizer_2.load_state_dict(torch.load(filename + "_critic_optimizer_2"))
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)



def train_agent():

    # Training Loop
    env = Env(True)
    state_dim = 16
    action_dim = 2
    max_action = 0.5
    prev_action = [0.0,0.0]

    agent = TD3Agent(state_dim, action_dim,max_action)
    total_episodes = 4000
    total_steps = 80000
    meanReward = []
    allReward = []
    episode = 0

    while episode <= total_episodes:
        state, done = env.reset(), False
        episode_reward = 0
        step = 0
        while not done and step<=total_steps:
            action = agent.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action,prev_action)
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            prev_action = action
            agent.train()
            step += 1
            time.sleep(0.01)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        allReward.append(episode_reward)
        meanScore = np.mean(allReward[-100:])
        meanReward.append(meanScore)
        episode += 1

        if episode % 100 == 0 :
            print("Saving model.....")
            agent.save("td3")

            with open('Rewards.csv', mode='w') as dataFile:
                rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                rewardwriter.writerow(meanReward)

    # env.close()


    
def test_agent():

    env = Env(True)
    state_dim = 16
    action_dim = 2
    max_action = 0.5
    prev_action = [0.0,0.0]

    agent = TD3Agent(state_dim, action_dim,max_action)
    episode = 0
    total_episodes = 100
    total_steps = 80000

    # load model
    agent.load("/home/ameya/turtlebot_ws2/src/td3_planning/model/td3")

    while episode <= total_episodes:
        state, done, arrive = env.reset2(), False, False
        # episode_reward = 0
        step = 0
        while not (done or arrive):
            action = agent.select_action(np.array(state))
            next_state, reward, done, arrive = env.step(action,prev_action)
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            prev_action = action
            # time.sleep(0.01)
            # step += 1

        episode += 1


def resume_training():

    env = Env(True)
    state_dim = 16
    action_dim = 2
    max_action = 0.5
    prev_action = [0.0,0.0]

    agent = TD3Agent(state_dim, action_dim,max_action)
    total_episodes = 25000
    total_steps = 80000
    meanReward = []
    allReward = []
    episode = 11101

    # load model
    agent.load("/home/ameya/turtlebot_ws2/src/td3_planning/run7/td3")

    print("Resuming training with saved model ....")

    while episode <= total_episodes:
        state, done = env.reset(), False
        episode_reward = 0
        step = 0
        while not done and step<=total_steps:
            action = agent.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action,prev_action)
            agent.add_experience(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            prev_action = action
            agent.train()
            step += 1
            time.sleep(0.01)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        allReward.append(episode_reward)
        meanScore = np.mean(allReward[-100:])
        meanReward.append(meanScore)
        episode += 1

        if episode % 100 == 0 :
            print("Saving model.....")
            agent.save("td3")

            with open('Rewards.csv', mode='w') as dataFile:
                rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                rewardwriter.writerow(meanReward)




def main(args=None):
    rclpy.init(args=args)

    # print("Enter Option \n 1. Train \n 2. Test\n")
    option = int(input("Enter Option \n 1. Train \n 2. Test\n 3. Resume Training\n"))

    if option == 1:
        train_agent()

    elif option == 2:
        test_agent()

    elif option == 3:
        resume_training()


    rclpy.shutdown()


if __name__ == '__main__':
    main()
