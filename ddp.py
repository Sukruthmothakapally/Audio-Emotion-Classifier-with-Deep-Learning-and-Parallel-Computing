import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import torch.nn.functional as F

# Define the model architecture
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 63 * 63, 32)
        self.fc2 = nn.Linear(32, num_classes)  # Use num_classes here

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 63 * 63)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(rank, world_size, train_loader, test_loader, num_classes, num_cpus=4):
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    # Initialize the distributed environment
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Set device for GPU
    device = torch.device(f'cuda:{rank}')

    # Determine devices for data parallelism (GPU and CPUs)
    devices = [device]  # Add GPU to devices

    # Add CPUs to devices
    if rank == 0:
        devices += [f'cpu:{i}' for i in range(num_cpus)]

    # Create the model and move it to GPU
    model = Net(num_classes).to(device)

    # Make the model DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    start_time = time.time()
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}, accuracy: {accuracy}')
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    return training_time

if __name__ == "__main__":
    train(rank, world_size, train_loader, test_loader, num_classes)
