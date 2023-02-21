'''********************************************
-----------------------------------------------
* Course: COMP 4600/5800 Homework: Part I
* Name : Xingyu Lyu
* ID: 02011552
* DATE: Feb-19-2023
* Python Version: Python 3.9.12
* Torch Version: 1.12.1+cu102
* Code: An CNN model training on MNIST DATASET
------------------------------------------------
********************************************'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

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
for epoch in range(200):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
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


# Save the training and testing accuracy to a file
# with open('accuracy.txt', 'w') as f:
#     f.write('Epoch\tTraining Accuracy\tTest Accuracy\n')
#     for i in range(len(train_acc)):
#         f.write('{}\t{}\t{}\n'.format(i, train_acc[i], test_accs[i]))

# Plot the training and testing accuracy
import matplotlib.pyplot as plt

# Plot the training accuracy
plt.plot(range(1, len(train_acc)+1), train_acc, 'b')
plt.legend(['Training Accuracy'],loc='lower right')
# plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
# plt.xticks(range(1, len(train_acc)+1))
plt.savefig('train_accuracy_1.pdf')
plt.show()


# plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_accs,'r')
plt.legend(['Training Accuracy','Testing Accuracy'],loc='lower right')
# plt.title('Testing accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
# plt.legend()
plt.savefig('test_accuracy_1.pdf')
plt.show()