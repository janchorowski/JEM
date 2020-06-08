import torch
import numpy as np
import sys
from urllib import request
from torch.utils.data import Dataset
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

sys.path.append("../semi-supervised")
n_classes = 10
cuda = torch.cuda.is_available()
n_examples_per_class = 10


mnist_train = MNIST(root="./", transform=transforms.ToTensor(),download=True, train=True)
mnist_test = MNIST(root="./", transform=transforms.ToTensor(),download=True, train=False)

# print(mnist_train[0][0].shape)
# for i in range(20):
#     print(mnist_train[i][1])

train_labels = mnist_train.train_labels.numpy()
test_labels = mnist_test.test_labels.numpy()
print(len(mnist_train))

# def get_subset(labels):
#     indices = np.arange(len(labels))
#     label_inds = labels.argsort()
#     sorted_examples = indices[label_inds]
#     return sorted_examples

def get_subset_sampler(labels):
    indices = np.arange(len(labels))
    selected_indices = np.hstack(
        [list(filter(lambda idx: labels[idx] == i, indices))[:n_examples_per_class] for i in
         range(n_classes)])
    selected_indices = torch.from_numpy(selected_indices)
    sampler = SubsetRandomSampler(selected_indices)
    return sampler

get_subset_sampler(train_labels)

labelled = torch.utils.data.DataLoader(mnist_train, batch_size=100, sampler=get_subset_sampler(
                                               mnist_train.train_labels.numpy(),
                                               ))

print(len(labelled))

unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                         num_workers=2, pin_memory=cuda,
                                         sampler=get_sampler(
                                             mnist_train.train_labels.numpy()))


def get_mnist(transform_set, location="./", batch_size=64, labels_per_class=100):
    # mnist_train = MNIST(location, train=True, download=True,
    #                     transform=transform_set,
    #                     )
    # mnist_valid = MNIST(location, train=False, download=True,
    #                     transform=transform_set,
    #                     )
    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(
            reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in
             range(n_labels)])

        print(indices)
        print(indices.shape)

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                           num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(
                                               mnist_train.train_labels.numpy(),
                                               labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                             num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(
                                                 mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size,
                                             num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(
                                                 mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation


# get_mnist(None)
