import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import cv2
from collections import deque
from config import Config

import game


class ThreeConvTwoFCNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=4):
        super(ThreeConvTwoFCNetwork, self).__init__()
        self.conv1 = self.layer_init(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4))
        self.conv2 = self.layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2))
        self.conv3 = self.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))
        self.fc4 = self.layer_init(nn.Linear(7 * 7 * 64, 512))
        self.fc5 = self.layer_init(nn.Linear(512, action_dim))

        self.set_gpu(0)

    def forward(self, x):
        x = self.tensor(x)
        y = functional.relu(self.conv1(x))
        y = functional.relu(self.conv2(y))
        y = functional.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = functional.relu(self.fc4(y))
        y = self.fc5(y)
        return y

    def predict(self, x, to_numpy=False):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

    def layer_init(self, layer):
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def set_gpu(self, gpu):
        if gpu >= 0 and torch.cuda.is_available():
            gpu = gpu % torch.cuda.device_count()
            self.device = torch.device('cuda:{}'.format(gpu))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class Task:
    def __init__(self, frame_skip=4, history_length=4):
        self.frame_skip = frame_skip
        self.history_length = history_length

        self.env = game.AttackOnBall()
        self.action_dim = 2
        self.state_dim = (84, 84, 1)
        self.warp_width = 84
        self.warp_height = 84

        self.skip_buffer = np.zeros((2,) + self.state_dim, dtype=np.uint8)
        self.stack_buffer = deque([], maxlen=history_length)

    def reset(self):
        state = self.env.reset()
        state = self.warp(state)
        state = state.transpose(2, 0, 1)
        for _ in range(self.history_length):
            self.stack_buffer.append(state)
        return LazyFrames(list(self.stack_buffer))

    def step(self, action):
        # skip step
        total_reward = 0.0
        done = None
        for i in range(self.frame_skip):
            next_state, reward, done = self.env.step(action)
            next_state = self.warp(next_state)
            if i == self.frame_skip - 2:
                self.skip_buffer[0] = next_state
            if i == self.frame_skip - 1:
                self.skip_buffer[1] = next_state
            total_reward += reward
            if done:
                break
        max_frame = self.skip_buffer.max(axis=0)
        max_frame = max_frame.transpose(2, 0, 1)
        self.stack_buffer.append(max_frame)
        return LazyFrames(list(self.stack_buffer)), total_reward, done

    def warp(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.warp_width, self.warp_height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class Policy:
    def __init__(self, epsilon, epsilon_decay, min_epsilon):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def sample(self, action_value):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(action_value))
        return np.argmax(action_value)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def sample(self):
        sample_indices = [np.random.randint(0, len(self.data)) for _ in range(self.batch_size)]
        sample_data = [self.data[index] for index in sample_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sample_data)))
        return batch_data

    def size(self):
        return len(self.data)


class Agent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def episode(self, deterministic=False):
        state = self.task.reset()
        total_reward = 0.0
        steps = 0

        while True:
            value = self.network.predict(self.network.tensor(state).unsqueeze(0), True).squeeze(0)
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done = self.task.step(action)
            total_reward += reward
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next, _ = self.target_network.predict(next_states, False).detach().max(1)

                terminals = self.network.tensor(terminals)
                rewards = self.network.tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = self.network.tensor(actions).unsqueeze(1).long()
                q = self.network.predict(states, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
            if done:
                break

        return total_reward, steps

    def test_episode(self):
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = np.argmax(self.target_network.predict(self.network.tensor(state).unsqueeze(0), True).squeeze(0))
            state, reward, done = self.task.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        return total_reward, steps


if __name__=="__main__":
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: Task(frame_skip=4, history_length=config.history_length)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: ThreeConvTwoFCNetwork(action_dim=2, in_channels=4)
    config.policy_fn = lambda: Policy(epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
    config.discount = 0.99
    config.target_network_update_freq = 1000
    config.exploration_steps = 5000
    config.episode_limit = 1000

    #run episodes
    agent = Agent(config)
    np.random.seed()
    torch.manual_seed(np.random.randint(int(1e6)))

    ep = 0
    rewards = []
    steps = []
    while True:
        ep += 1
        reward, step = agent.episode()
        print("ep{}: reward: {}, step: {}".format(ep, reward, step))
        rewards.append(reward)
        steps.append(step)

        if agent.config.episode_limit and ep > agent.config.episode_limit:
            break
        if agent.config.max_steps and agent.total_steps > agent.config.max_steps:
            break

    print()
    for _ in range(1000):
        reward, step = agent.test_episode()
        print(reward)
