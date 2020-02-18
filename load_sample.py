import numpy as np
import matplotlib.pyplot as plt

samples = np.load("samples.npy")
# print(samples)
print(samples.shape)

plt.imshow(samples[1].reshape(28,28), cmap="viridis")
plt.show()
