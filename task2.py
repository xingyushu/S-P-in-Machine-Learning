import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from __future__ import print_function
from torchvision import datasets, transforms


epsilons = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]

use_cuda = False


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

    # Save the model after each epoch
    torch.save(model.state_dict(), 'mnist_model.pt')



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

import matplotlib.pyplot as plt

# Plot the training accuracy
plt.plot(range(1, len(train_acc)+1), train_acc, 'b')
plt.legend(['Training Accuracy'],loc='lower right')
# plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
# plt.xticks(range(1, len(train_acc)+1))
plt.savefig('train_accuracy_task2_without_attack.pdf')
# plt.show()


# plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_accs,'r')
plt.legend(['Training Accuracy','Testing Accuracy'],loc='lower right')
# plt.title('Testing accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
# plt.legend()
plt.savefig('test_accuracy_task2_without_attack.pdf')
# plt.show()


# Save the training and testing accuracy to a file
# with open('accuracy.txt', 'w') as f:
#     f.write('Epoch\tTraining Accuracy\tTest Accuracy\n')
#     for i in range(len(train_acc)):
#         f.write('{}\t{}\t{}\n'.format(i, train_acc[i], test_accs[i]))

# Plot the training and testing accuracy
import matplotlib.pyplot as plt


# Evaluate the success rate of the FGSM attack for each epsilon value
success_rates = []
for epsilon in epsilons:
    num_correct = 0
    num_total = 0
    for images, labels in test_loader:
        # Generate adversarial examples using FGSM attack
        images.requires_grad = True
        output = model(images)
        loss = nn.functional.nll_loss(output, labels)
        loss.backward()

        # Compute the sign of the gradient with respect to the input data
        images_grad = images.grad.data.sign()

        # Perturb the input data with the sign of the gradient to generate adversarial examples
        perturbed_images = images + epsilon * images_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Evaluate the victim model on the adversarial examples
        output = model(perturbed_images)
        _, predicted = torch.max(output.data, 1)
        num_correct += (predicted == labels).sum().item()
        num_total += labels.size(0)

    success_rate = num_correct / num_total
    success_rates.append(success_rate)
    print('Epsilon: {}\tSuccess rate: {:.2f}%'.format(epsilon, 100. * success_rate))

# Plot the success rate as a function of epsilon
plt.plot(epsilons, success_rates)
plt.xlabel('Epsilon')
plt.ylabel('Success rate (%)')
plt.savefig('success_rate_task2.pdf')
# plt.show()


# Plot the training accuracy
# plt.plot(range(1, len(train_acc)+1), train_acc, 'b')
# plt.legend(['Training Accuracy'],loc='lower right')
# # plt.title('Training accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# # plt.xticks(range(1, len(train_acc)+1))
# plt.savefig('train_accuracy_task2.pdf')
# # plt.show()


# # plt.plot(train_acc, label='Training Accuracy')
# plt.plot(test_accs,'r')
# plt.legend(['Training Accuracy','Testing Accuracy'],loc='lower right')
# # plt.title('Testing accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy(%)')
# # plt.legend()
# plt.savefig('test_accuracy_task2.pdf')
# # plt.show()


print("-------------------------------------Begin ATTACK----------------------------------------------------")






# download the dataset
loader = torch.utils.data.DataLoader(
    datasets.MNIST('data2/', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1,
    shuffle=True
)


# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc = []
test_accs = []


# Load the victim model
model = Net()
model.load_state_dict(torch.load('mnist_model.pt'))
model.eval()


cuda_available = torch.cuda.is_available()
device = torch.device('cude' if (use_cuda and cuda_available) else 'cpu')
print('CUDA is available: ', cuda_available)


# FGSM Attack Module
def fgsm_attack_module(image, epsilon, data_grad):
 
    sign_data_grad = data_grad.sign()
    adversarial_image = image + epsilon * sign_data_grad
    # set range to [0,1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    return adversarial_image



def test( model, device, test_loader, epsilon ):

    correct = 0
    # adverserial examples
    adv_examples = []

    for data, target in test_loader:

        # send the data and labels to device
        data, target = data.to(device), target.to(device)

        # set  requires_grad in tensor
        data.requires_grad = True

        # forward data 
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # check the init  predict == target ? If not,continue
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)

        model.zero_grad()

        loss.backward()

        # collect datagrad
        data_grad = data.grad.data

        #launch the fgsm_attack_module
        perturbed_data = fgsm_attack_module(data, epsilon, data_grad)

        # run the model on perturbed_data
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))


    return final_acc, adv_examples



accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device,loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig('test_accuracy_task2_with_Epsilon.pdf')
# plt.show()


# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.savefig('examples_task2.pdf')
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

