import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import time

# =========================== Replay Buffer ===========================
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# =========================== Neural Network Model ===========================
def create_model(lr, n_actions, input_dims, fc_dims):
    model = Sequential([
        Dense(fc_dims, input_shape=(input_dims,), activation='relu'),
        Dense(fc_dims, activation='relu'),
        Dense(fc_dims, activation='relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(learning_rate=lr, decay=0.001), loss='mse')
    return model

# =========================== DDQN Agent ===========================
class DDQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims=9, epsilon_dec=0.995, epsilon_end=0.1,
                 mem_size=10000, fname='Model', replace_target=50):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = create_model(alpha, n_actions, input_dims, 32)
        self.q_target = create_model(alpha, n_actions, input_dims, 32)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def act(self, state):
        state = np.array(state)[np.newaxis, :]
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        actions = self.q_eval.predict(state)
        return np.argmax(actions)

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)
            q_target = q_pred
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                self.gamma * q_next[batch_index, max_actions.astype(int)] * done

            self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        timestr = time.strftime("-%d-%m-%Y-%H-%M")
        self.q_eval.save(f"Models/{self.model_file}{timestr}.h5")

    def load_model(self, path):
        self.q_eval = load_model(path)
        self.q_eval.summary()
        self.q_target = load_model(path)
        if self.epsilon == 0.0:
            self.update_network_parameters()

# =========================== End of Code ===========================
