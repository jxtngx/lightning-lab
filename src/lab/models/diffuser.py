# generated with Meta's Llama 70B Instruct variant using Hugging Chat

import torch

# Define the model architecture
class DiffusionModel:
  def __init__(self, num_steps, beta_schedule, image_size):
    self.num_steps = num_steps
    self.beta_schedule = beta_schedule
    self.image_size = image_size
    self.encoder = Encoder(image_size)  # Encoder network
    self.decoder = Decoder(image_size)  # Decoder network

  def forward(self, x):
    # Forward process
    x_noisy = x
    for i in range(self.num_steps):
      beta = self.beta_schedule[i]
      eps = torch.randn_like(x_noisy)  # Sample noise
      x_noisy = x_noisy + beta * eps
    return x_noisy

  def reverse(self, x_noisy):
    # Reverse process
    x_recon = x_noisy
    for i in range(self.num_steps - 1, -1, -1):
      beta = self.beta_schedule[i]
      eps_recon = self.encoder(x_recon)
      x_recon = (x_recon - beta * eps_recon) / (1 - beta**2)
    return x_recon

  def loss(self, x, x_noisy):
    # Loss function
    x_recon = self.reverse(x_noisy)
    return torch.mean((x - x_recon)**2)

  def train_step(self, x):
    # Training step
    x_noisy = self.forward(x)
    loss = self.loss(x, x_noisy)
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Define the encoder and decoder networks
class Encoder(torch.nn.Module):
  def __init__(self, image_size):
    super(Encoder, self).__init__()
    self.fc1 = torch.nn.Linear(image_size, 128)
    self.fc2 = torch.nn.Linear(128, 128)
    self.fc3 = torch.nn.Linear(128, image_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class Decoder(torch.nn.Module):
  def __init__(self, image_size):
    super(Decoder, self).__init__()
    self.fc1 = torch.nn.Linear(image_size, 128)
    self.fc2 = torch.nn.Linear(128, 128)
    self.fc3 = torch.nn.Linear(128, image_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Define the beta schedule
def beta_schedule(num_steps):
  beta = torch.linspace(0.0001, 0.02, num_steps)
  return beta
