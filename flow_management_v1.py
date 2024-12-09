import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from collections import deque
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TABLE_SIZE = 50
NUM_FLOWS = 1000

def generate_flows(num_flows=NUM_FLOWS):
    flows = []
    for i in range(num_flows):
        flows.append({
            'priority': np.random.randint(0, 100),
            'age': np.random.randint(0, 100),
            'packet_count': np.random.randint(0, 100),
            'bytes_count': np.random.randint(0, 100),
        })
    return flows
table_flow = generate_flows(TABLE_SIZE)
new_flows = generate_flows()

class FlowTableEnvironment(gym.Env):
    def __init__(self, table_size=TABLE_SIZE):
        super(FlowTableEnvironment, self).__init__()
        self.table_size = table_size
        self.action_space = gym.spaces.Discrete(table_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(table_size * 4,), dtype=np.float32)

        self.flow_table = table_flow
        self.flows = new_flows
        self.current_flow_index = 0
        self.step_count = 0
        self.print_flow_table()

    def reset(self):
        self.flows = new_flows
        self.flow_table = table_flow
        self.step_count = 0
        self.current_flow_index = 0

        return self._get_state()

    def _get_state(self):
        flow_info = []
        
        # Find max values for normalization
        max_priority = max(flow['priority'] for flow in self.flow_table)
        max_age = max(flow['age'] for flow in self.flow_table)
        max_packets = max(flow['packet_count'] for flow in self.flow_table)
        max_bytes = max(flow['bytes_count'] for flow in self.flow_table)
        
        # Normalize each feature to [0,1] range for each flow
        for flow in self.flow_table:
            flow_info.append(flow['priority'] / max_priority if max_priority > 0 else 0)
            flow_info.append(flow['age'] / max_age if max_age > 0 else 0)
            flow_info.append(flow['packet_count'] / max_packets if max_packets > 0 else 0)
            flow_info.append(flow['bytes_count'] / max_bytes if max_bytes > 0 else 0)

        return np.array(flow_info, dtype=np.float32)

    # Return max age of the flow table and its index
    def max_age(self):
        max_age = max(flow['age'] for flow in self.flow_table)
        max_age_index = [flow['age'] for flow in self.flow_table].index(max_age)
        return max_age, max_age_index

    def step(self, action):
        self.step_count += 1
        reward = self._calculate_reward(action)
        max_age, max_age_index = self.max_age()
        self.flow_table[action] = self.flows[self.current_flow_index]
        # print(f"New flow added to index {action} with age {self.flow_table[action]['age']}. Max age is {max_age} at index {max_age_index}")
        self.current_flow_index += 1
            
        done = self.current_flow_index >= len(self.flows)
        return self._get_state(), reward, done

    def _calculate_reward(self, action):
        flow_stats = self.flow_table[action]
         # Find max values for normalization
        max_priority = max(flow['priority'] for flow in self.flow_table)
        max_age = max(flow['age'] for flow in self.flow_table)
        max_packets = max(flow['packet_count'] for flow in self.flow_table)
        max_bytes = max(flow['bytes_count'] for flow in self.flow_table)

        age_reward = (flow_stats['age'] / max_age)
        priority_reward = 1 - (flow_stats['priority'] / max_priority)
        packet_reward = 1 - (flow_stats['packet_count'] / max_packets)
        bytes_reward = 1 - (flow_stats['bytes_count'] / max_bytes)

        reward = (age_reward + priority_reward + packet_reward + bytes_reward) / 4

        if flow_stats['age'] < max_age * 0.2:
            reward -= 0.5
        
        if flow_stats['priority'] > max_priority * 0.8:
            reward -= 0.5
            
        if flow_stats['packet_count'] > max_packets * 0.8:
            reward -= 0.3
            
        if flow_stats['bytes_count'] > max_bytes * 0.8:
            reward -= 0.3

        if reward < -1:
            reward = -1

        return reward
    def print_flow_table(self):
        indexed_table = []
        for i, flow in enumerate(self.flow_table):
            flow_with_index = {'index': i}
            flow_with_index.update(flow)
            indexed_table.append(flow_with_index)

        print(tabulate(indexed_table, headers="keys", tablefmt="grid"))

class QNetwork(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DoubleDQNAgent:
    """Double DQN agent implementation with dynamic sampling strategy"""
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_q_network = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.step_count = 0
        self.training_phase = "exploration"

    def act(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return np.argmax(q_values.cpu().data.numpy())  
        
    def learn(self, replay_buffer, batch_size=256):
        """Update networks using dynamic prioritized experience replay"""
        if len(replay_buffer) < batch_size:
            return
        
        # Update training phase based on step count
        if self.step_count > 5000: 
            self.training_phase = "exploitation"
        
        # Adjust sampling strategy based on training phase
        if self.training_phase == "exploration":
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = \
                replay_buffer.sample(batch_size, prioritized=False)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = \
                replay_buffer.sample(batch_size, prioritized=True)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
        done_batch = torch.FloatTensor(np.array(done_batch)).unsqueeze(1).to(device)
        weights = torch.FloatTensor(np.array(weights)).unsqueeze(1).to(device)

        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)
        next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]

        target_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values

        q_values = q_values.gather(1, action_batch)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Update priorities with higher emphasis during exploitation
        if self.training_phase == "exploitation":
            td_errors = td_errors * 1.25
        
        replay_buffer.update_priorities(indices, td_errors)

        unweighted_loss = nn.MSELoss(reduction='none')(q_values, target_q_values)
        loss = (weights * unweighted_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform soft update every step
        self.soft_update_target_network()
        self.step_count += 1
        return loss.item()

    def soft_update_target_network(self):
        """Soft update target network parameters using tau"""
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
class ReplayBuffer:
    """Prioritized Experience Replay Buffer with dynamic sampling"""
    def __init__(self, buffer_size=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, prioritized=True):
        """Sample batch using dynamic strategy"""
        if len(self.buffer) < batch_size:
            return
        
        if prioritized:
            # Prioritized sampling for exploitation phase
            priorities = np.array([float(p) for p in self.priorities], dtype=np.float64)
            probs = (priorities + self.epsilon) ** self.alpha
            probs /= np.sum(probs)
        else:
            # Uniform sampling for exploration phase
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        if prioritized:
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
        else:
            weights = np.ones(batch_size)
        
        samples = [self.buffer[idx] for idx in indices]
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*samples)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, 
                weights, indices)

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)
    

def train_agent(env, agent, episodes=500, batch_size=256, epsilon_decay=0.995):
    wandb.init(project="flow-table-management-v1", 
               config={
                   "episodes": episodes,
                   "batch_size": batch_size,
                   "epsilon_decay": epsilon_decay,
               })
    
    epsilon = 1.0
    epsilon_min = 0.01
    replay_buffer = ReplayBuffer()
    episode_rewards = []
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            if len(replay_buffer) >= batch_size:
                loss = agent.learn(replay_buffer, batch_size)
                if loss is not None:
                    episode_loss += loss

        avg_loss = episode_loss / steps if steps > 0 else 0
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)
        loss_history.append(avg_loss)
        
        wandb.log({
            "episode": episode + 1,
            "total_reward": total_reward,
            "average_loss": avg_loss,
            "epsilon": epsilon
        })
        
        print(f"Episode {episode + 1} finished with total reward {total_reward}, average loss {avg_loss:.4f}, end in {steps} steps")

    wandb.finish()
    return agent, episode_rewards, loss_history

if __name__ == "__main__":
    env = FlowTableEnvironment()
    agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)
    train_agent(env, agent)
