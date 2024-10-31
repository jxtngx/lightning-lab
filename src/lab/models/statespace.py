# demo of a selective state space model
# generated with Meta's Llama 70B Instruct variant using Hugging Chat

import torch
import torch.nn as nn

class SimpleStateSpaceModel(nn.Module):
    def __init__(self, state_dim, obs_dim, 
                 transition_matrix=None, observation_matrix=None,
                 process_noise_var=0.1, observation_noise_var=0.5):
        """
        Initialize the Simple State Space Model.
        
        :param state_dim: Dimension of the state.
        :param obs_dim: Dimension of the observation.
        :param transition_matrix: A (state_dim x state_dim) matrix. If None, defaults to an identity matrix.
        :param observation_matrix: A (obs_dim x state_dim) matrix. If None, defaults to a matrix that simply selects the first 'obs_dim' states.
        :param process_noise_var: Variance of the process noise.
        :param observation_noise_var: Variance of the observation noise.
        """
        super(SimpleStateSpaceModel, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Initialize Transition Matrix (A)
        self.A = nn.Parameter(torch.tensor(transition_matrix) if transition_matrix is not None 
                               else torch.eye(state_dim), requires_grad=True)
        
        # Initialize Observation Matrix (C)
        self.C = nn.Parameter(torch.tensor(observation_matrix) if observation_matrix is not None 
                               else torch.cat([torch.eye(obs_dim), torch.zeros(obs_dim, state_dim-obs_dim)], dim=1), 
                               requires_grad=True)
        
        # Initialize Noise Variances
        self.process_noise_var = nn.Parameter(torch.tensor(process_noise_var), requires_grad=False)
        self.observation_noise_var = nn.Parameter(torch.tensor(observation_noise_var), requires_grad=False)
        
    def forward(self, initial_state, steps=1):
        """
        Generate a sequence of observations from the model.
        
        :param initial_state: The initial state (tensor of shape (state_dim,)).
        :param steps: Number of steps to generate.
        :return: A tensor of shape (steps, obs_dim) containing the generated observations.
        """
        observations = torch.zeros((steps, self.obs_dim))
        state = initial_state
        
        for t in range(steps):
            # Generate Observation
            observation = torch.matmul(self.C, state) + torch.randn(self.obs_dim) * torch.sqrt(self.observation_noise_var)
            observations[t] = observation
            
            # Update State
            state = torch.matmul(self.A, state) + torch.randn(self.state_dim) * torch.sqrt(self.process_noise_var)
        
        return observations