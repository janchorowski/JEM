import torch
import pandas as pd
import numpy as np
import pickle


def get_data(dataset, test_frac=.1, seed=1234, n_classes=10):
    if dataset == "concrete":
        training_data_x = pd.read_excel("data/concrete.xls")
        data_dim = 8
        x = training_data_x.to_numpy()
    elif dataset == "protein":
        training_data_x = pd.read_csv("data/protein.csv")
        data_dim = 9
        x = training_data_x.to_numpy()
    elif dataset == "power_plant":
        training_data_x = pd.read_excel("data/power_plant/Folds5x2_pp.xlsx")
        data_dim = 4
        x = training_data_x.to_numpy()
    elif dataset == "navy":
        with open("data/navy.txt", 'r') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                ls = line.split()
                ls = [float(v) for v in ls]
                vals.append(ls)
        x = np.array(vals).astype(np.float)
        x = x[:, :-1]  # take out last thing
        data_dim = 16
    elif dataset == "year":
        # with open("data/year.txt", 'r') as f:
        #     lines = f.readlines()
        #     vals = []
        #     for line in lines:
        #         ls = line.strip().split(',')
        #         ls = [float(v) for v in ls]
        #         vals.append(ls)
        # x = np.array(vals).astype(np.float)
        with open("data/year.pkl", 'rb') as f:
            x = pickle.load(f)
        #import pickle
        #pickle.dump(x, open("data/year.pkl", 'wb'))
        #1/0
        data_dim = 90
    else:
        raise ValueError

    mu = x.mean(0)
    std = x.std(0)


    x = (x - mu[None]) / (std[None] + 1e-6)


    n_test = int(x.shape[0] * test_frac)
    if dataset == "year":
        n_test = 51630
    inds = list(range(x.shape[0]))
    np.random.seed(seed)
    np.random.shuffle(inds)

    train_inds = np.array(inds[:-n_test])
    test_inds = np.array(inds[-n_test:])
    train, test = x[train_inds], x[test_inds]

    # concrete and power-plant is [x, y]
    if dataset == "concrete" or dataset == "power_plant" or dataset == "navy" or dataset == "year":
        xtr, ytr = train[:, :-1], train[:, -1]
        xte, yte = test[:, :-1], test[:, -1]
    # protein is [y, x]
    elif dataset == "protein":
        xtr, ytr = train[:, 1:], train[:, 0]
        xte, yte = test[:, 1:], test[:, 0]
    else:
        raise ValueError

    # compute bins
    y_sorted = sorted(ytr)

    n_per_class = len(y_sorted) // n_classes
    buckets = []
    for i in range(n_classes - 1):
        ind = (i + 1) * n_per_class
        b = y_sorted[ind]
        buckets.append(b)

    def reg_to_clf(y):
        for i in range(n_classes - 1):
            if y < buckets[i]:
                return i
        return n_classes - 1

    ytr_clf = np.array([reg_to_clf(y) for y in ytr])
    yte_clf = np.array([reg_to_clf(y) for y in yte])
    # import matplotlib.pyplot as plt
    # plt.hist(ytr_clf)
    # plt.hist(yte_clf)
    # plt.show()

    xtr, xte = [torch.from_numpy(v).float() for v in [xtr, xte]]
    ytr, yte = [torch.from_numpy(v).long() for v in [ytr_clf, yte_clf]]

    dset_train = torch.utils.data.TensorDataset(xtr, ytr)
    dset_test = torch.utils.data.TensorDataset(xte, yte)
    return dset_train, dset_test, data_dim


if __name__ == "__main__":
    tr, te, ddim = get_data("year")
    print(ddim, tr[0][0].size())
