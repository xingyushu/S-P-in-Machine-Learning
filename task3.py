import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from __future__ import print_function



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
        return nn.functional.log_softmax(x, dim=1), x

# Set up the training and test data loaders
# Define transformation to convert images to tensors and normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST train and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)






train_acc = []
test_accs = []


# define adversarial training function
def adversarial_train(model,train_loader, optimizer, criterion, epsilon):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Generate adversarial example
        data.requires_grad = True
        output,_  = model(data)
        loss = nn.functional.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon*sign_data_grad
        # Train with adversarial example
        optimizer.zero_grad()
        adv_output, _ = model(perturbed_data)
        adv_loss = criterion(adv_output, target)
        adv_loss.backward()
        optimizer.step()
    

                    # compute training accuracy and print it on the shell
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        total = target.size(0)
        accuracy = 100. * correct / total
        # train_acc.append(accuracy)
        

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item(), accuracy))
        
        
        if (batch_idx + 1) == len(train_loader):
            # Compute training accuracy only after last batch of the epoch
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            total = target.size(0)
            accuracy = 100. * correct / total
            train_acc.append(accuracy)

    # Save the model after each epoch
    torch.save(model.state_dict(), 'mnist_model_defense_045.pt')

    return model





# Define the training parameters,loss function and optimizer
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 200
learning_rate = 0.001
epsilon = 0.45

# epsilons = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
use_cuda = False
# criterion = nn.MSELoss()
model = Net()
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




# Train the model with adversarial examples
for epoch in range(1, epochs+1):
    adversarial_train(model,train_loader, optimizer, criterion, epsilon)
     # Evaluate the model on the test data
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output,_ = model(data)  # move data to GPU
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))

    print(f'Epoch {epoch} completed.')

torch.save(model.state_dict(), 'mnist_model_defense_045_2.pt')


# Plot the training accuracy
plt.plot(range(1, len(train_acc)+1), train_acc, 'b')
plt.legend(['Training Accuracy'],loc='lower right')
# plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
# plt.xticks(range(1, len(train_acc)+1))
plt.savefig('defense_attack_train.pdf')
# plt.show()


# plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_accs,'r')
plt.legend(['Training Accuracy','Testing Accuracy'],loc='lower right')
# plt.title('Testing accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
# plt.legend()
plt.savefig('defense_attack_test.pdf')
# plt.show()
