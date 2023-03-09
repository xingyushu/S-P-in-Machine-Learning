import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Set up the training and test data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc = []
test_accs = []


# Train the model for 10 epochs
'''
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # Evaluate the model on the test data
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))


# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pt')
'''

# FGSM AE Attack
epsilon = 0.1  # perturbation strength
alpha = 0.05   # step size for attack iterations
steps = 10     # number of attack iterations

# define loss function for the attack
# define the FGSM AE loss function
def fgsm_ae_loss(output, target, perturbed_data, criterion, epsilon):
    # calculate the loss
    # first, calculate the original loss between the output and target
    original_loss = criterion(output, target)

    # calculate the perturbation as the sign of the gradient of the loss w.r.t the input
    perturbation = torch.sign(torch.autograd.grad(original_loss, perturbed_data, 
                                                   retain_graph=False, create_graph=False)[0])

    # perturb the data by adding the perturbation scaled by epsilon
    perturbed_data = perturbed_data + epsilon * perturbation

    # clamp the perturbed data between 0 and 1
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    # calculate the loss between the perturbed data and the original input data
    ae_loss = criterion(perturbed_data, data)

    # combine the losses and return
    total_loss = original_loss + ae_loss
    return total_loss

# load the MNIST test set

# test_data = MNIST('data', train=False, download=True, transform=transform)
# test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)


# test_dataset =MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# load the pre-trained model
model = torch.load('mnist_cnn.pt')

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# define the criterion
# criterion = nn.CrossEntropyLoss()

# set the epsilon for the FGSM AE attack
epsilon = 0.1

# perform the FGSM AE attack on a sample image from the test set
# model.eval()
data, target = next(iter(test_loader))
data = data.to(device)
target = target.to(device)
data.requires_grad = True
output = model(data)
loss = criterion(output, target)
loss.backward()
perturbed_data = data + epsilon * torch.sign(data.grad)
perturbed_data = torch.clamp(perturbed_data, 0, 1)

# evaluate the model on the perturbed data
# model.eval()
output = model(perturbed_data)
test_loss = criterion(output, target).item()
pred = output.argmax(dim=1, keepdim=True)
correct = pred.eq(target.view_as(pred)).sum().item()

# calculate the success rate of the victim model
success_rate = 100. * (1 - correct / len(target))

# print the results
print('Original image:')
plt.imshow(data[0][0].cpu(), cmap='gray')
plt.show()

print('Perturbed image:')
plt.imshow(perturbed_data[0][0].cpu(), cmap='gray')
plt.show()

print('Prediction on original image:', output.argmax(dim=1)[0].item())
print('Prediction on perturbed image:', pred[0].item())
print('Test loss on perturbed image:', test_loss)
print('Test accuracy on perturbed image: {:.2f}%'.format(100. * correct / len(target)))
print('Success rate of the victim model: {:.2f}%'.format(success_rate))
