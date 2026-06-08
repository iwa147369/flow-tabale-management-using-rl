import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128


class QNetwork(nn.Module):
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
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.step_count = 0
        self.training_phase = "exploration"

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())

    def learn(self, replay_buffer, batch_size=BATCH_SIZE):
        if len(replay_buffer) < batch_size:
            return

        if self.step_count > 5000:
            self.training_phase = "exploitation"

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

        # Double DQN: online network selects action, target network evaluates it
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_q_network(next_state_batch).gather(1, next_actions)

        target_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values

        q_values = q_values.gather(1, action_batch)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        if self.training_phase == "exploitation":
            td_errors = td_errors * 1.25

        replay_buffer.update_priorities(indices, td_errors)

        unweighted_loss = nn.MSELoss(reduction='none')(q_values, target_q_values)
        loss = (weights * unweighted_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target_network()
        self.step_count += 1
        return loss.item()

    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)


class ReplayBuffer:
    def __init__(self, buffer_size=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, prioritized=True):
        if len(self.buffer) < batch_size:
            return

        if prioritized:
            priorities = np.array([float(p) for p in self.priorities], dtype=np.float64)
            probs = (priorities + self.epsilon) ** self.alpha
            probs /= np.sum(probs)
        else:
            probs = np.ones(len(self.buffer)) / len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

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
        for idx, td_error in zip(indices, td_errors.flatten()):
            self.priorities[idx] = float(td_error) + self.epsilon

    def __len__(self):
        return len(self.buffer)
