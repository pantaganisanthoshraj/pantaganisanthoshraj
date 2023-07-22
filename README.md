# Import the necessary libraries
import numpy as np

# Define the environment
class Environment:
    def __init__(self):
        self.state = 0
        self.reward = 0
        self.done = False
        
    def step(self, action):
        if action == 0:
            self.state += 1
        elif action == 1:
            self.state -= 1
        
        if self.state == 10:
            self.reward = 1
            self.done = True
        elif self.state == 0:
            self.reward = -1
            self.done = True
        else:
            self.reward = 0
            self.done = False
        
        return self.state, self.reward, self.done

# Define the agent
class Agent:
    def __init__(self):
        self.q_table = np.zeros((11, 2))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(2)
        else:
            action = np.argmax(self.q_table[state])
        
        return action
    
    def learn(self, state, action, reward, next_state):
        td_error = reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# Train the agent
env = Environment()
agent = Agent()

for i in range(1000):
    state = env.state
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    agent.learn(state, action, reward, next_state)
    
    if done:
        env = Environment()
        
# Test the agent
state = env.state
while not env.done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    state = next_state
    
print(f'Total reward: {env.reward}')
