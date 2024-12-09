# Flow Table Environment
# ---------------------
# - Table size: 100 entries
# - State representation:
#   - Flow entry: priority, age, packet count, bytes count
#   - Global characteristics: max age, table utilization, duplicate rate
# - Actions:
#   - Delete flow
#   - Increase/decrease priority
#   - Increase/decrease timeout

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_flow():
    """Generate a random flow entry with realistic network parameters"""
    return {
        "priority": np.random.randint(0, 100),
        "age": np.random.randint(0, 100),  # seconds
        "packet_count": np.random.randint(0, 1000),
        "bytes_count": np.random.randint(0, 100000),
        "match": {
            "in_port": np.random.randint(1, 5),
            "eth_type": 0x0800,  # IPv4
            "ipv4_src": f"192.168.1.{np.random.randint(1, 255)}",
            "ipv4_dst": f"192.168.2.{np.random.randint(1, 255)}",
            "ip_proto": 6,  # TCP
            "tcp_src": np.random.randint(1024, 65535),
            "tcp_dst": np.random.choice([80, 443, 8080])  # Common ports
        },
        "actions": [
            {"type": "OUTPUT", "port": np.random.randint(1, 5)}
        ],
        "idle_timeout": np.random.randint(0, 300),  # seconds
        "hard_timeout": np.random.randint(300, 600)  # seconds
    }

class FlowTableEnvironment(gym.Env):
    """OpenAI Gym environment for flow table management"""
    def __init__(self, table_size=50):
        super(FlowTableEnvironment, self).__init__()
        self.table_size = table_size
        self.action_space = gym.spaces.Discrete(5)
        # State space: 4 flow entry features (priority, age, packet_count, bytes_count) + 7 global features (age, avg_age, occupancy, duplicate, traffic, priority)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        self.flow_table = [generate_flow() for _ in range(table_size)]
        self.current_flow_index = 0
        self.print_flow_table()

    def reset(self):
        """Reset environment to initial state"""
        self.flow_table = [generate_flow() for _ in range(self.table_size)]
        self.current_flow_index = 0
        # self.print_flow_table()
        return self._get_state()
    
    def flow_similarity(self, index):
        """Calculate similarity score of current flow with all flows in the table"""
        current_flow = self.flow_table[index]
        similarity_score = 0
        
        for i, other_flow in enumerate(self.flow_table):
            if i == index:
                continue
                
            match1 = current_flow['match']
            match2 = other_flow['match']
            
            # Calculate similarity based on match fields
            field_matches = 0
            total_fields = 4  # Number of fields we check
            
            if match1['ipv4_src'] == match2['ipv4_src']:
                field_matches += 1
            if match1['ipv4_dst'] == match2['ipv4_dst']:
                field_matches += 1
            if match1['tcp_src'] == match2['tcp_src']:
                field_matches += 1
            if match1['tcp_dst'] == match2['tcp_dst']:
                field_matches += 1
                
            # Add normalized similarity score for this flow
            similarity_score += field_matches / total_fields
            
        # Return average similarity across all other flows
        return similarity_score / (len(self.flow_table) - 1) if len(self.flow_table) > 1 else 0
    
    def _get_state(self):
        """Get current state representation"""
        # Get current flow features
        flow_info = []
        flow_info.append(float(self.flow_table[self.current_flow_index]['priority']) / 100.0) # Normalize priority to 0-1
        flow_info.append(min(float(self.flow_table[self.current_flow_index]['age']) / 3600.0, 1.0)) # Cap age at 1 hour
        flow_info.append(min(float(self.flow_table[self.current_flow_index]['packet_count']) / 1000.0, 1.0)) # Cap packets at 1000
        flow_info.append(min(float(self.flow_table[self.current_flow_index]['bytes_count']) / 1000000.0, 1.0)) # Cap bytes at 1MB
        flow_info.append(self.flow_similarity(self.current_flow_index))

        # Get global features
        global_info = []
        
        # Age statistics
        max_age = max(float(flow['age']) for flow in self.flow_table)
        avg_age = sum(float(flow['age']) for flow in self.flow_table) / len(self.flow_table)
        global_info.append(min(max_age / 3600.0, 1.0)) # Cap max age at 1 hour
        global_info.append(min(avg_age / 3600.0, 1.0)) # Cap avg age at 1 hour
        
        # Table occupancy
        global_info.append(float(len(self.flow_table)) / float(self.table_size)) # Already 0-1
        
        # Traffic statistics
        total_packets = sum(float(flow['packet_count']) for flow in self.flow_table)
        total_bytes = sum(float(flow['bytes_count']) for flow in self.flow_table)
        global_info.append(min(total_packets / 10000.0, 1.0)) # Cap at 10k packets
        global_info.append(min(total_bytes / 10000000.0, 1.0)) # Cap at 10MB
        
        # Priority distribution
        priorities = [float(flow['priority']) for flow in self.flow_table]
        avg_priority = sum(priorities) / len(priorities)
        global_info.append(avg_priority / 100.0) # Normalize to 0-1

        # Convert to numpy array and ensure float32
        state = np.array(flow_info + global_info, dtype=np.float32)
        
        # Clip to ensure all values are between 0 and 1
        return np.clip(state, 0.0, 1.0)

    def step(self, action):
        """Execute one environment step"""
        # Check if current_flow_index is valid before getting state
        if self.current_flow_index + 1 == len(self.flow_table):
            return self._get_state(), 0, True
    
        # Get state and reward
        state = self._get_state()
        reward = self._calculate_reward(action)
        
        return state, reward, False
    
    def _calculate_reward(self, action):
        """Calculate reward based on action taken"""
        if action == 0:  # Delete
            return self.delete_flow(self.current_flow_index)
        elif action == 1:  # Increase priority
            reward = self.increase_priority(self.current_flow_index)
        elif action == 2:  # Decrease priority
            reward = self.decrease_priority(self.current_flow_index)
        elif action == 3:  # Increase timeout
            reward = self.increase_timeout(self.current_flow_index)
        elif action == 4:  # Decrease timeout
            reward = self.decrease_timeout(self.current_flow_index)
        self.current_flow_index += 1
        return reward
    
    def delete_flow(self, index):
        """Delete flow and calculate reward based on flow characteristics
        
        The reward is calculated considering multiple factors:
        1. Age: Normalized age of the flow
           - Older flows may be less relevant
        2. Activity metrics: Normalized packet and byte counts
           - Less active flows are better candidates for deletion
        3. Priority: Normalized priority value
           - Lower priority flows are better candidates for deletion
        4. Similarity: Similarity score of the flow with other flows in the table
           - More similar flows are better candidates for deletion
        
        The final reward combines these factors to determine if deletion was beneficial:
        - Higher reward for deleting old, low activity, low priority flows
        - Lower reward for deleting young, high activity, high priority flows
        """
        # Get flow metrics and normalize them
        flow = self.flow_table[index]
        
        # Priority component
        flow_priority = flow['priority']
        if flow_priority > 70:
            priority_component = -1
        elif flow_priority < 30:
            priority_component = 1
        else:
            priority_component = 0
        
        # Efficiency component
        # TODO: Consider age and activity high when deleting
        # Case old age and high activity
        # Case young age and low activity
        flow_age = flow['age']
        flow_packet_count = flow['packet_count']
        flow_bytes_count = flow['bytes_count']
        mean_age = sum(f['age'] for f in self.flow_table) / len(self.flow_table)
        mean_packets = sum(f['packet_count'] for f in self.flow_table) / len(self.flow_table)
        mean_bytes = sum(f['bytes_count'] for f in self.flow_table) / len(self.flow_table)
        if flow_age > mean_age * 0.8 and (flow_packet_count < mean_packets * 0.8 or flow_bytes_count < mean_bytes * 0.8):
            efficiency_component = 1
        elif flow_age < mean_age * 0.2 and (flow_packet_count > mean_packets * 0.8 or flow_bytes_count > mean_bytes * 0.8):
            efficiency_component = -1
        else:
            efficiency_component = 0
        

        # Recent activity component
        # TODO: Consider logic of timeout reward
        flow_idle_timeout = flow['idle_timeout']
        flow_hard_timeout = flow['hard_timeout']
        mean_idle_timeout = sum(f['idle_timeout'] for f in self.flow_table) / len(self.flow_table)
        mean_hard_timeout = sum(f['hard_timeout'] for f in self.flow_table) / len(self.flow_table)
        if flow_idle_timeout > mean_idle_timeout * 0.8 and flow_hard_timeout > mean_hard_timeout * 0.8:
            activity_component = 1
        elif flow_idle_timeout < mean_idle_timeout * 0.3 and flow_hard_timeout < mean_hard_timeout * 0.3:
            activity_component = -1
        else:
            activity_component = 0

        # Table utilization component
        # TODO: Flowtable will be full all time ???
        table_utilization = len(self.flow_table) / self.table_size
        if table_utilization > 0.9:
            table_utilization_component = 1
        elif table_utilization < 0.5:
            table_utilization_component = -1
        else:
            table_utilization_component = 0
        
        similarity_component = self.flow_similarity(index)
        
        # Calculate final weighted reward
        reward = (priority_component * 0.2 + 
                 efficiency_component * 0.3 + 
                 activity_component * 0.3 + 
                 table_utilization_component * 0.1 + 
                 similarity_component * 0.1)

        # Delete the flow and return reward
        del self.flow_table[index]
        return reward
    
    def priority_reward(self, index):
        """Calculate reward based on flow priority and activity alignment.
        
        Args:
            index (int): Index of flow in flow table
            
        Returns:
            float: Reward value between -2 and 1
                  1.0: Priority level matches activity level
                  Negative: Priority mismatches activity, scaled by mismatch size
                  
        The reward compares the flow's priority level with its activity level:
        - Activity level combines packet count and byte count into low/med/high
        - Priority level splits into low/med/high based on thresholds
        - Reward is positive when priority appropriately matches activity
        - Reward is negative when priority is misaligned with activity
        """
        # Get current flow metrics
        current_packets = self.flow_table[index]['packet_count'] 
        current_bytes = self.flow_table[index]['bytes_count']
        current_priority = self.flow_table[index]['priority']
        
        # Define level thresholds
        packet_levels = [100, 500, 1000]  # Low, Medium, High thresholds 120
        byte_levels = [1000, 5000, 10000]  # Low, Medium, High thresholds 5500
        priority_levels = [30, 60, 90]  # Low, Medium, High thresholds 10
        
        # Determine packet level (0=Low, 1=Medium, 2=High)
        packet_level = 0 #1
        for i, threshold in enumerate(packet_levels):
            if current_packets > threshold:
                packet_level = i + 1
                
        # Determine byte level (0=Low, 1=Medium, 2=High)
        byte_level = 0 #2
        for i, threshold in enumerate(byte_levels):
            if current_bytes > threshold:
                byte_level = i + 1
                
        # Determine priority level (0=Low, 1=Medium, 2=High)
        priority_level = 0
        for i, threshold in enumerate(priority_levels):
            if current_priority > threshold:
                priority_level = i + 1
                
        # Calculate activity level as average of packet and byte levels
        activity_level = (packet_level + byte_level) / 2
        
        # Calculate reward
        if abs(priority_level - activity_level) <= 0.5:
            # Priority matches activity - positive reward
            reward = 1.0
        else:
            # Priority doesn't match activity - negative reward
            reward = -1.0 * abs(priority_level - activity_level)
            
        return reward
    
    def increase_priority(self, index):
        self.flow_table[index]['priority'] = min(100, self.flow_table[index]['priority'] + 10)
        reward = self.priority_reward(index)
        return reward
    
    def decrease_priority(self, index):
        self.flow_table[index]['priority'] = max(1, self.flow_table[index]['priority'] - 10)
        reward = self.priority_reward(index)
        return reward
    
    def timeout_reward(self, index):
        """Calculate reward based on timeout settings alignment.
        
        Args:
            index (int): Index of flow in flow table
            
        Returns:
            float: Reward value between -1 and 1
                  Higher rewards for:
                  - Longer timeouts for active flows when table has space
                  - Shorter timeouts for inactive flows when table is full
                  Lower rewards for:
                  - Long timeouts when table is full
                  - Short timeouts for active flows
                  
        The reward considers three main components:
        1. Priority alignment: Higher priority flows should have longer timeouts
        2. Activity alignment: More active flows should have longer timeouts
        3. Table utilization: Fuller tables should encourage shorter timeouts
        
        Components are weighted:
        - Priority: 30%
        - Activity: 40%  
        - Table utilization: 30%
        """
        # Priority component calculation
        current_priority = self.flow_table[index]['priority']
        current_timeout = self.flow_table[index]['idle_timeout']
        
        if current_priority > 70 and current_timeout < 20:
            priority_component = 1
        elif current_priority < 30 and current_timeout > 40:
            priority_component = -1
        else:
            priority_component = 0
            
        # Activity component calculation
        # Calculate mean packet and byte counts across all flows
        mean_packet_count = sum(flow['packet_count'] for flow in self.flow_table) / len(self.flow_table)
        mean_bytes_count = sum(flow['bytes_count'] for flow in self.flow_table) / len(self.flow_table)
        
        # Normalize current flow counts relative to means
        # Calculate packet and byte ratios relative to means
        packet_ratio = self.flow_table[index]['packet_count'] / mean_packet_count
        bytes_ratio = self.flow_table[index]['bytes_count'] / mean_bytes_count
        
        # Cap ratios at 2.0 to avoid extreme values
        packet_count = min(packet_ratio, 2.0)
        bytes_count = min(bytes_ratio, 2.0)

        mean_timeout = sum(flow['idle_timeout'] for flow in self.flow_table) / len(self.flow_table)
        
        # Weight bytes more heavily since it's a better indicator of flow importance
        activity = (0.3 * packet_count + 0.7 * bytes_count) / 2.0
        
        # Adjust thresholds and add intermediate case
        if activity > 0.8 and current_timeout < mean_timeout:
            activity_component = 1
        # elif 0.4 <= activity <= 0.8:
        #     activity_component = 0.5 if current_timeout < mean_timeout else -0.5
        elif activity < 0.4 and current_timeout > mean_timeout:
            activity_component = -1
        else:
            activity_component = 0
            
        # Table utilization component calculation
        table_utilization = len(self.flow_table) / self.table_size
        
        if table_utilization > 0.9:
            table_utilization_component = -1
        elif table_utilization < 0.1:
            table_utilization_component = 1
        else:
            table_utilization_component = 0
            
        # Calculate final weighted reward
        reward = priority_component * 0.3 + activity_component * 0.4 + table_utilization_component * 0.3
        return reward
    
    def increase_timeout(self, index):
        self.flow_table[index]['idle_timeout'] = min(60, self.flow_table[index]['idle_timeout'] + 5)
        reward = self.timeout_reward(index)
        return reward
    
    def decrease_timeout(self, index):
        self.flow_table[index]['idle_timeout'] = max(5, self.flow_table[index]['idle_timeout'] - 5)
        reward = self.timeout_reward(index)
        return reward
    
    def print_flow_table(self):
        """Print formatted flow table for debugging"""
        # Create list of dictionaries with all fields flattened
        table_data = []
        for i, flow in enumerate(self.flow_table):
            row = {
                'index': i,
                'priority': flow['priority'],
                'age': flow['age'],
                'packet_count': flow['packet_count'],
                'bytes_count': flow['bytes_count'],
                **flow['match'],
                'actions': flow['actions'],
                'idle_timeout': flow['idle_timeout'], 
                'hard_timeout': flow['hard_timeout']
            }
            table_data.append(row)
        
        # Get headers from first row keys
        if table_data:
            # headers = list(table_data[0].keys())
            print(tabulate(table_data, headers='keys', tablefmt="grid"))
        else:
            print("Flow table is empty")
    
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
        self.tau = tau  # Soft update parameter
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
        self.training_phase = "exploration"  # Start with exploration phase

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
        if self.step_count > 10000:  # Transition to exploitation after 10k steps
            self.training_phase = "exploitation"
        
        # Adjust sampling strategy based on training phase
        if self.training_phase == "exploration":
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = \
                replay_buffer.sample(batch_size, prioritized=False)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = \
                replay_buffer.sample(batch_size, prioritized=True)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        # Get current Q values and next Q values
        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)
        next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]

        target_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values

        q_values = q_values.gather(1, action_batch)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Update priorities with higher emphasis during exploitation
        if self.training_phase == "exploitation":
            td_errors = td_errors * 1.5  # Increase priority scaling during exploitation
        
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
            priorities = np.array([float(p) for p in self.priorities])
            probs = (priorities + self.epsilon) ** self.alpha
            probs /= probs.sum()
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
    

def train_agent(env, agent, episodes=5000, batch_size=256, epsilon_decay=0.999):
    """Train agent using Double DQN with prioritized experience replay"""
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
        step_count = 0
        
        # Track actions and rewards for this episode
        action_counts = {}
        action_rewards = {}

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            
            # Update action statistics
            if action not in action_counts:
                action_counts[action] = 0
                action_rewards[action] = 0
            action_counts[action] += 1
            action_rewards[action] += reward
            
            total_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            step_count += 1

            if len(replay_buffer) >= batch_size:
                loss = agent.learn(replay_buffer, batch_size)
                if loss is not None:
                    episode_loss += loss

        # Calculate average reward per action
        avg_rewards_per_action = {}
        for action in action_counts:
            avg_rewards_per_action[action] = action_rewards[action] / action_counts[action]
            
        avg_loss = episode_loss / step_count
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)
        loss_history.append(avg_loss)
        
        print(f"Episode {episode + 1:4d} finished with total reward {total_reward:5.3f}, average loss {avg_loss:5.3f}")
        print("Action statistics:")
        for action in sorted(action_counts.keys()):
            print(f"Action {action}: count = {action_counts[action]}, average reward = {avg_rewards_per_action[action]:.3f}")
        print("-" * 50)

    return agent, episode_rewards, loss_history
if __name__ == "__main__":
    env = FlowTableEnvironment()
    agent = DoubleDQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent, episode_rewards, loss_history = train_agent(env, agent)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.show()
