import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from PIL import Image
import math
import simulation_model


def extract_yaw_from_quaternion(q):
            q_x, q_y, q_z, q_w = q
            yaw = math.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
            return yaw
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

# Create the dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, max_length=768):
        self.root_path = path        
        self.max_length = max_length
        

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        idx = (idx % 20) + 1  

        images = self.root_path + f"\\Accidents\\accident{idx}\\"
        images = [Image.open(images + f"IMG{i}.png").convert("RGB") for i in range(1, 9)]
        images = [img.resize((224, 224)) for img in images]  # Resize to 224x224
        images = [torch.tensor(numpy.array(img)).permute(2, 0, 1) for img in images]  # Convert to tensor and change to (C, H, W)
        images = [img.float() / 255.0 for img in images]  # Normalize to [0, 1]
        images = [img.unsqueeze(0) for img in images]  # Add batch dimension
        images = torch.stack(images, dim=0) # dim=0 is the batch dimension, shape(1, 8, 3, 224, 224)

        text = self.root_path + f"\\Accidents\\accident{idx}\\"
        text = [load_text_file(text + f"V{i}.txt") for i in range(1, 3)]
        # text = [list(t.encode('utf-8')) for t in text]
        # text = [t[:self.max_length] if len(t) > self.max_length else t + [0] * (self.max_length - len(t)) for t in text]
        # text = [torch.tensor(t) for t in text]
        # text = torch.stack(text, dim=0)  # shape(1, 2, 768)
        
        car1_spawn = torch.tensor(torch.load(self.root_path + f"\\Labels\\pos1_{idx}.pt"))
        car2_spawn = torch.tensor(torch.load(self.root_path + f"\\Labels\\pos2_{idx}.pt"))

        start_orientation1 = torch.tensor(torch.load(self.root_path + f"\\Labels\\start_orientation1_{idx}.pt"))
        start_orientation2 = torch.tensor(torch.load(self.root_path + f"\\Labels\\start_orientation2_{idx}.pt"))

        yaw1 = extract_yaw_from_quaternion(start_orientation1)
        yaw2 = extract_yaw_from_quaternion(start_orientation2)

        # replace the z with yaw
        car1_spawn[2] = yaw1
        car2_spawn[2] = yaw2
        cars_params = torch.load(self.root_path + f"\\Labels\\cars_parameters{idx}.pt")

        return images, text, car1_spawn, car2_spawn, cars_params

# Test the dataset
dataset = Dataset("DB")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# for i, (images, text, car1_spawn, car2_spawn, cars_params) in enumerate(dataloader):
#     print(f"Batch {i+1}:")
#     print(f"Images shape: {images.shape}")
#     print(f"Text shape: {len(text)}")
#     print(f"Car 1 spawn: {car1_spawn}")
#     print(f"Car 2 spawn: {car2_spawn}")
#     print(f"Cars parameters shape: {cars_params.shape}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = simulation_model.AccidentSimulator()                          # your model
model.to(device)

# Freeze all the layers except the decoder layer
for name, param in model.named_parameters():
    if 'decoder' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Training parameter: {name}")


# loss should work for both images and text, so use a custom loss function. But if we want to use a pre-defined loss function, we can use nn.MSELoss()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    model.train()

    for i, (images, text, car1_spawn, car2_spawn, cars_params) in enumerate(dataloader):
        images = images.to(device)
        # text = text.to(device)
        # car1_spawn = car1_spawn.to(device)
        # car2_spawn = car2_spawn.to(device)
        # cars_params = cars_params.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, text, car1_spawn, car2_spawn)

        # Compute loss
        loss_value = loss(outputs, cars_params)

        # Backward pass and optimization
        loss_value.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss_value.item():.4f}")

