# train_test.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import time
import torch.nn.functional as F

def train_and_test(X_train, y_train, X_test, y_test, num_cpus):
    
    # Check the number of unique classes
    num_classes = len(set(y_train.tolist() + y_test.tolist()))
    
    # Create PyTorch datasets and data loaders with num_workers
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_cpus)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=num_cpus)

    # Define the model architecture inside the function
    class Net(nn.Module):
        def __init__(self, num_classes):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 63 * 63, 32)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(-1, 16 * 63 * 63)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create the model
    model = Net(num_classes)

    # Create the optimizer with the model parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store loss and accuracy for each epoch
    training_time_list = []

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Start the timer for training
    start_time_train = time.time()

    # Train the model
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Test the model after each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}, accuracy: {accuracy}')
        
    # End the timer for training
    end_time_train = time.time()
    training_time_list.append(end_time_train - start_time_train)
    print("Training time with {num_cpus}:",end_time_train - start_time_train)
        
    return training_time_list

if __name__ == '__main__':
    train_and_test(X_train, y_train, X_test, y_test, 1)  # Run with a single CPU for testing
