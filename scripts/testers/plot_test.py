import matplotlib.pyplot as plt
import random
import numpy as np

def plot(labels, x, y,tile):
    plt.subplot(5, 5, tile)
    plt.scatter(x, y,
                s=20, alpha=0.8, cmap='Set1', c=labels)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    return plt

y = np.random.uniform(0, 1000, (100,))
labels = np.random.randint(0, 2, (100,))

for row in range(1, 11):
    x = np.random.uniform(0, 1000, (100,))
    plt_ = plot(labels, x, y, row)

plt_.show()
