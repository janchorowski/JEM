import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)

batch_size = 100


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        layer_in = nn.Linear(input_size, hidden_size)
        self.layers.append(layer_in)
        for i in range(0):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layer_out = nn.Linear(hidden_size, output_size)
        # self.layers.append(layer_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        # output = self.layer1(x)
        # output = self.relu(output)
        # output = self.layer2(output)
        output = self.layer_out(x)
        return output


input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 5

learning_rate = 0.002

model = NeuralNet(input_size, hidden_size, output_size)

# CE loss below has logsoftmax (logsumexp?) and neg log lik already together
lossFunction = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)
        out = model(images)
        loss = lossFunction(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_size == 0:
            print(out[0])
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        out = model(images)
        _, predicted = torch.max(out.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
