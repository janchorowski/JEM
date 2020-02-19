import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np


input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 20
image_size = 28

learning_rate = 1e-5 # with warmup

sigma = 1e-2

batch_size = 64

sgld_steps = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose(
            [transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),
             lambda x: x + sigma * torch.randn_like(x)]
        )
transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((.5,), (.5,)),
             lambda x: x + sigma * torch.randn_like(x)]
        )

train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transform_train, download=True)

# train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transform_test, download=True)

# test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(), download=True)



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




# Maybe a more powerful network needed to be able to handle the JEM training
# Huh but yeah it can diverge all of a sudden
model = NeuralNet(input_size, hidden_size, output_size, extra_layers=2).to(device)
f = model

# CE loss below has logsoftmax (logsumexp?) and neg log lik already together
p_y_given_x_loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def energy(x):
    return -(f(x).logsumexp(dim=1))

def sgld(steps: int, step_size: int = 1, ):
    # Draws a sample according to SGLD, starting from uniform distribution sample
    # 0,1 as uniform bounds for MNIST
    alpha = step_size
    x_0 = torch.FloatTensor(batch_size, image_size * image_size).uniform_(0, 1)
    x_i = x_0
    for step in range(steps):
        # Smaller noise for more accurate SGLD was all I needed to make it work?
        eps = torch.FloatTensor(batch_size, image_size * image_size).normal_(0, np.sqrt(alpha)/100).to(device)
        x_i = torch.tensor(x_i, requires_grad=True).to(device)
        # print(f(x_i))
        # print(f(x_i, y=torch.randint(0, 10, (batch_size,))))
        # de_dx = torch.autograd.grad(outputs=-f(x_i).logsumexp(dim=1).sum(), inputs=x_i)[0]
        de_dx = torch.autograd.grad(outputs=energy(x_i).sum(), inputs=x_i)[0].to(device)

        # print(alpha)
        # print(de_dx)
        x_i1 = x_i - alpha/2 * de_dx + eps #logsumexp gradient here, with respect to (input?)
        x_i = x_i1
    return x_i.detach()


def test_network():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, image_size * image_size).to(device)
            out = model(images).cpu()
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(
            100 * correct / total))

# def test_batch():
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.reshape(-1, image_size * image_size).to(device)
#             out = model(images)
#             _, predicted = torch.max(out.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             break;
#
#         print('Accuracy of the network on test images: {} %'.format(
#             100 * correct / total))


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        loss = 0
        images = images.reshape(-1, image_size * image_size).to(device)
        x = images
        # f_x = f(x)

        # p(x)
        x_prime = sgld(sgld_steps)
        # surrogate loss, in that loss.backward() gives us the deriv we want
        # expectation approximated by sgld, with equal weighting on samples drawn
        # Their code has it "backward" because of the logsumexp not having the negative sign
        # mean takes average over batch, we get equal weighting. Note that I guess
        # the expectation is approximated from a single sample in this case.
        log_p_x = (energy(x_prime).mean() - energy(x).mean()).cpu()
        # print(log_p_x)

        # and of course, here we need negative log likelihood for the loss
        loss += -log_p_x
        # print(-log_p_x)

        # p(y|x)
        out = model(images).cpu()
        # out = f_x
        loss += p_y_given_x_loss(out, labels)
        # print(images[0])
        # print(p_y_given_x_loss(out, labels))
        # print(loss)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_size == 0:

            # Hacky warmup
            learning_rate += 1e-5 / 4
            learning_rate = min(learning_rate, 1e-4)
            print(learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            print(-log_p_x)
            print(p_y_given_x_loss(out, labels))

            # print(out[0])
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            _, predicted = torch.max(out.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            print('Train Accuracy: {:.4f}'.format(correct/total))
            # test_batch()


test_network()


samples = sgld(sgld_steps).cpu()
print(samples)
np.save("samples" + str(num_epochs), np.array(samples))




