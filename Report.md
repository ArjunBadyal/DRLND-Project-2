[//]: # (Image References)

[image1]: https://github.com/arjunlikesgeometry/DRLND-Project-2/blob/master/P2.png
[image2]: https://github.com/arjunlikesgeometry/DRLND-Project-2/blob/master/DDPG.png

### Introduction
In this project the DDPG algorithm was used to solve the second version of the reacher environemnt outlined in the readme. Previously the DQN algorithm was used to solve the environment in Project 1, however this was a discrete action space. DDPG allows us to generate an action value function for a continuous action space.

### Algorithm and Network Architecture
![DDPG][image2]
The algorithm above was taken from this <cite><a href="https://arxiv.org/abs/1509.02971"><i>paper</i></a></cite>. This is an actor-critic method, meaning the actor updates the policy whilst the critic estimates the value function. One of the key features of this algorithm is the blending of the weights of the actor and critic neural networks into their respective target networks. Another feature is the use of a replay buffer as seen in the DQN algorithm in the first project.

Overall, this code was modified from the ddpg-pendulum code provided by Udacity.

Both the actor and critic neural networks are made up of three linear layers with the relu fuction used in between layers. The actor network however returns with the tanh function due to the continuous nature of this task. This can be seen in the model.py file: 
```python
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
return self.fc3(x)
```
The hyperparmeters were as follows:
```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0001 # L2 weight decay
```
Initially it was taking too long for the agents to train so I changed the learning rate of the critic from 1e-3 to 1e-4. This seemed to make a difference, proabably because the actor depends on the critic. Reducing the minibatch size to 64 may have also made a difference, however this would require more work looking into how the combination of problem dimensionality and minibatch size affect overall performance of the algorithm.  

The last thing to note is that noise was added using the Ornstein–Uhlenbeck process in order to aid exploration of the agents in the environment:
```python
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
return self.state
```

### Results

The results show that the environment was solved in 169 espisodes i.e. this was the point after which the average score was greater than or equal than 30 for the next 100 episodes. The weights used to solve the environment have been saved in the checkpoint_actor.pth and checkpoint_critic.pth files and may be loaded to see the performance of the trained model.

Episode 100	Average Score: 1.07

Episode 200	Average Score: 7.62

Episode 269	Average Score: 30.27

Environment solved in 169 episodes!	Average Score: 30.27

![Trained Agent][image1]

### Conclusion and Future Work
To conclude, with the right hyperparameters it appears that DDPG works well for continuous tasks of this nature. Future work could include further experimentation with the hyperparameters such as the effects of minibatch size highlighted above. Other algorithms could also be tested such as PPO, A3C and D4PG. Long term stability and convergence of the algorithm could also be tested, since the training stopped as soon as it reached the target threshold in this case. 
