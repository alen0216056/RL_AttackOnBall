
class Config:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = 0.99
        self.target_network_update_freq = 0
        self.max_episode_length = 0
        self.exploration_steps = 0
        self.history_length = 1
        self.update_interval = 1
        self.gradient_clip = 0.5
        self.entropy_weight = 0.01
        self.min_memory_size = 200
        self.min_epsilon = 0
        self.save_interval = 0
        self.max_steps = 0
        self.iteration_log_interval = 30
        self.optimization_epochs = 4
        self.num_mini_batches = 32
        self.evaluation_env = None
        self.evaluation_episodes_interval = 0
        self.evaluation_episodes = 0
        self.tau = 0.001
