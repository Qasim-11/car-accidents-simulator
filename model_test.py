import torch.optim as optim
from simulation_model import *
# Initialize model, loss, and optimizer
model = AccidentSimulator()
# get the number of model parameters
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

# save the model
out = model(torch.randn(1, 8, 3, 224, 224), "This is a test text")
print(out)
print(out.shape)