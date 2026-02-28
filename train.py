import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出动作概率分布
        return F.softmax(self.fc2(x), dim=-1)


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


def train():
    env = gym.make('CartPole-v1')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = PolicyNetwork(n_states, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99
    
    episodes = 500
    for ep in range(episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        

        for t in range(500):
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            saved_log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break
        

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        

        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 计算损失函数: Loss = - ∑ log_prob * G_t
        policy_loss = []
        for log_prob, G in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}\t Last reward: {len(rewards)}")

    env.close()
    print("训练完成！")

if __name__ == "__main__":
    train()