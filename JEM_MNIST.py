import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np


train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)

batch_size = 100


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extra_layers=0):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        layer_in = nn.Linear(input_size, hidden_size)
        self.layers.append(layer_in)
        for i in range(extra_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layer_out = nn.Linear(hidden_size, output_size)
        # self.layers.append(layer_out)

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        logits = self.layer_out(x)
        # if y is None:
        #     output = logits.logsumexp(dim=1)
        #     print(logits.shape)
        #     print(output.shape)
        # else:
        #     print(y[:,None])
        #     output = torch.gather(logits, 1, y[:,None])
        output = logits
        return output


input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 5
image_size = 28

learning_rate = 0.002

model = NeuralNet(input_size, hidden_size, output_size)
f = model

# CE loss below has logsoftmax (logsumexp?) and neg log lik already together
p_y_given_x_loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#

def sgld(steps: int, step_size: int = 2, ):
    # Draws a sample according to SGLD, starting from uniform distribution sample
    # 0,1 as uniform bounds for MNIST
    alpha = step_size
    x_0 = torch.FloatTensor(batch_size, image_size * image_size).uniform_(0, 1)
    x_i = x_0
    for step in range(steps):
        eps = torch.FloatTensor(batch_size, image_size * image_size).normal_(0, np.sqrt(alpha))
        x_i = torch.tensor(x_i, requires_grad=True)
        # print(f(x_i))
        # print(f(x_i, y=torch.randint(0, 10, (batch_size,))))
        de_dtheta = torch.autograd.grad(outputs=-f(x_i).logsumexp(dim=1).sum(), inputs=x_i)[0]
        # print(alpha)
        # print(de_dtheta)
        x_i1 = x_i - alpha/2 * de_dtheta + eps #logsumexp gradient here, with respect to (input?)
        x_i = x_i1
    return x_i

sgld(10)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, image_size * image_size)
        out = model(images)
        loss = p_y_given_x_loss(out, labels)
        # print(images[0])

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
        images = images.reshape(-1, image_size*image_size)
        out = model(images)
        _, predicted = torch.max(out.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
