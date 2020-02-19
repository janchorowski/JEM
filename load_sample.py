import numpy as np
import matplotlib.pyplot as plt

samples = np.load("samples20.npy")
# print(samples)
print(samples.shape)

for i in range(11, 20):
    plt.imshow(samples[i].reshape(28,28), cmap="viridis")
    plt.show()
