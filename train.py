# import gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Bernoulli
from environment import Water_Distribution

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
        return F.sigmoid(self.fc2(x))


def select_action(policy, state):
    
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    # 假设 policy 输出 5 个独立的概率，每个都在 [0, 1] 之间
    # 通常网络最后层用 Sigmoid 激活函数

    
    # 建立伯努利分布
    m = Bernoulli(probs)
    
    # 采样得到一个向量，例如 tensor([1., 0., 1., 1., 0.])
    actions = m.sample()
    
    
    # 计算这组动作组合的 log_prob
    # 对于独立事件，总概率的对数等于各项对数概率之和
    # print(actions)
    
    return actions, m.log_prob(actions).sum()


def train():
    
    
    env_kwargs = {
        "channel_num": 5,
        "channel_pos": [0.1, 0.3, 0.5, 0.7, 0.9],
        "branch_width": 3,
        "main_road_width": 5,
        "wall_height": 1.0,
        "gate_thickness": 1,
        "channel_state": [1, 0, 1, 0, 1],
        "x_max": 100,
        "y_max": 20,
        "sim_name": "env_test",
        "script_path": __file__,
        "parallel": 6
    }

    
    
    env = Water_Distribution(sim_name = "env_test")
    # n_states = env.observation_space.shape[0]
    # n_actions = env.action_space.n
    n_states = env_kwargs["channel_num"]
    n_actions = env_kwargs["channel_num"]
    
    policy = PolicyNetwork(n_states, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99
    
    episodes = 500
    for ep in range(episodes):
        saved_log_probs = []
        rewards = []

        env_kwargs["channel_state"] = np.random.randint(0, 2, env_kwargs['channel_num']).tolist()
        
        state, _ = env.reset(**env_kwargs)

        print("###### Episode: ", ep)
        

        for t in range(500):
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated

            print("action:", action)
            print("state:", state)
            print("reward:", reward)
            
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
        
        if (ep + 1) % 1 == 0:
            print(f"Episode {ep+1}\t Last reward: {len(rewards)}")

    env.close()
    print("训练完成！")

if __name__ == "__main__":
    train()